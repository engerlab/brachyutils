from pathlib import Path
from typing import Literal, Optional, Union, List, Dict
# from collections import defaultdict
import numpy as np

from brachyutils.geometry.phantom_utils import BrachyPhantom
from brachyutils.geometry.phantom_utils import phantom_with_empty_image_like
from opentps.core.data._transform3D import Transform3D
from opentps.core.data.images import Deformation3D, VectorField3D
from opentps.core.data.images import ROIMask
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D
from brachyutils.geometry.registration_utils.reg_utils import BrachyPhantomRegistration

class Registration_Plastimatch(BrachyPhantomRegistration):
    def __init__(
        self,
        pth_plastimatch: Path | str,
        static_phantom: BrachyPhantom,
        moving_phantom: BrachyPhantom,
        register_on_contour: Union[Literal["common"], Optional[str]] = None,
        deformable: bool = False,
        # algorithm: Literal["demons", "bspline"] = None,
        backend = "plastimatch",
        tryGPU: bool = False,
        **kwargs
        ):
        r"""
        ### Purpose:
        - A class to wrap around the Plastimatch image registration method.
        ### Inputs:
        - pth_plastimatch: Path | str: The path to the plastimatch executable or the URL where
        the plastimatch server is running.
        - static_phantom: BrachyPhantom: The static phantom object.
        - moving_phantom: BrachyPhantom: The phantom object that is transformed to match the static phantom.
        - register_on_contour: Optional[str] | "common" = None: The name of the contour to be used in the registration process.
        if this input is provided, contour based registration is used. If "common" is provided, registeration is done
        based on all the common structures.
        - deforemable: bool = False: A flag to indicate whether the registration is deformable or not.
        - algorithm: Literal["Demons", "Morphons", ...] = None The type of registration algorithm.
        - backend: Literal["plastimatch"] = "opentps" The backend package used to handle 
        the registration process.
        - dir_phantom_export: Union[Path, str]: The path to the geometry setup directory.
        ### Outputs:
        - None
        ### Functions:
        - register: Register the moving phantom to the static phantom.
        ### Dependencies:
        - Plastimatch
        """

        super().__init__(
            static_phantom,
            moving_phantom,
            register_on_contour,
            deformable,
            # algorithm,
            backend,
            tryGPU
            )
        self.pth_plastimatch = pth_plastimatch if pth_plastimatch else kwargs.popitem("pth_plastimatch", None)

    def register(
        self,
        stage_params_list: List[Dict[str, str]] = None,
        dir_phantom_export: Path | str = None,
        ) -> tuple[BrachyPhantom, Transform3D]:
        r"""
        ### Purpose:
        - Register the moving phantom to the static phantom using the Plastimatch package.
        ### Inputs:
        - stage_params_list: List[Dict[str, str]] := a list of dictionaries containing the stage parameters for the registration.
        please look at the plastimatch documentation for the full list of possible stage parameters.
        - dir_phantom_export := directory where the registered phantom is exported to.
        ### Outputs:
        - BrachyPhantom: The registered phantom object.
        """
        # leave some space to figure out the rigidness and options for the registration.

        # need to write out the images for plastimatch to read them.
        # first sort out the paths to the images
        if dir_phantom_export is None:
            dir_temp_data = Path(__file__).resolve().parent.parent.parent.parent.joinpath("temp_data/registration/temp")

        elif "temp_data/registration" not in str(dir_phantom_export.resolve()):
            dir_temp_data = Path(__file__).resolve().parent.parent.parent.parent.joinpath("temp_data/registration")
        else:
            dir_temp_data = dir_phantom_export.resolve().joinpath("temp/"+self.moving_phantom.pth_image.stem)                        

        pth_static = dir_temp_data.joinpath("static.nrrd")
        pth_moving = dir_temp_data.joinpath("moving.nrrd")
        pth_output = dir_temp_data.joinpath("registered.nrrd")

        # create phantoms with empty image data. remember, the image data could be structure masks
        # or actual image data.
        for data, pth in zip([self._static_data, self._moving_data], [pth_static, pth_moving]):
            empty_phant = phantom_with_empty_image_like(
                self.moving_phantom,
                new_pth_image=pth
            )
            empty_phant.image_obj = data
            empty_phant.write_image_to_nrrd(
                pth_output=pth
            )

        global_params = {
            "fixed" : f"{str(pth_static)}",
            "moving" : f"{str(pth_moving)}",
            "image_out" : f"{str(pth_output)}",
            "vf_out" : f"{str(dir_temp_data.joinpath('vf.nrrd'))}",
        }

        stage_params_list = stage_params_list if stage_params_list else[
            {
                "xform": "bspline",
                # "optim": "versor",
                # "max_its": "50",
            }
        ]

        if "http" in self.pth_plastimatch:
            import requests
            response = requests.post(
                url=self.pth_plastimatch+"/plastimatch_register",
                json={
                    "global_params": global_params,
                    "stage_params_list": stage_params_list,
                    },
                timeout=None
            )
            # get the registered image
            if not pth_output.exists():
                raise ValueError("The registered image was not generated.")
        else:
            raise NotImplementedError("The local plastimatch registration is not implemented yet.")

        # load the registered image
        self._registered_data = BrachyPhantom(
            pth_phantom_file=pth_output
        ).image_obj
        self._registered_data = resampleImage3DOnImage3D(
            self._registered_data,
            self._static_data,
            )
        # self.deformation = _load_deformation_field(
        #     dir_temp_data.joinpath("vf.nrrd")
        #     )
        self.synch_registered_phantom_with_data(
            pth_vector_field=Path(global_params["vf_out"])
            )
        if dir_phantom_export is not None:
            self.export_to(dir_phantom_export)
        return self.registered_phantom, self.deformation

    def export_to(
        self,
        dir_registered_phantom: Path | str,
        output_type: Literal[".nrrd", ".dcm"] = ".nrrd") -> None:
        super().export_to(dir_registered_phantom, output_type)

    def synch_registered_phantom_with_data(
        self,
        pth_vector_field: Path = None
        ) -> None:
        r"""
        ### Purpose:
        - To match the image and the contours of the registered phantom with the registered data.
        by applying the vector field to the image and the contours using plastimatch convert.
        ### Inputs:
        - pth_vector_field: Path: The path to the vector field file.
        ### Outputs:
        - None
        """        
        # we have the path to the vf file.
        # we need to apply this deformation to the image and the contours.
        # each data needs to be written out to a file and given to plastimatch warp along with vf.
        # the output of the warp will be the registered image and contours.

        # load the registered data into registered phantom
        self.registered_phantom = phantom_with_empty_image_like(
            self.moving_phantom,
            new_pth_image=f"reg_{self.moving_phantom.pth_image.stem}"
            )
        
        # if registration based on image:
        # load the image into the registered phantom
        if self.register_on_contour is None:
            self.registered_phantom.image_obj = self._registered_data
        # registration based on contour
        # create a new contour based on the registered mask.
        else:
            # pass the moving image to the registered phantom image
            self.registered_phantom.image_obj = self.moving_phantom.image_obj
            # create a new contour based on the registered mask.
            new_contour = ROIMask(
                name=self.register_on_contour,
                imageArray=self._registered_data.imageArray,
                origin=self._registered_data.origin,
                spacing=self._registered_data.spacing,
            )
            self.registered_phantom.set_structure_set({self.register_on_contour: new_contour})

        structure_mask_dict = self.registered_phantom.get_structure_mask(
            self.registered_phantom.structure_names,
            mask_type=ROIMask
        )
        all_data = structure_mask_dict | {"image": self.registered_phantom.image_obj}
        for data_name in all_data:
            # write out the data to be warped by plastimatch
            pth_in = pth_vector_field.parent.joinpath(f"{data_name}.nrrd")
            pth_warped = pth_vector_field.parent.joinpath(f"{data_name}_warped.nrrd")

            empty_phant = phantom_with_empty_image_like(
                self.moving_phantom,
                new_pth_image=pth_in
            )
            empty_phant.image_obj = all_data.get(data_name)
            empty_phant.write_image_to_nrrd(
                pth_output=pth_in
            )
            # call plastimatch warp to deform the image and the contours.
            if "http" in self.pth_plastimatch:
                import requests
                response = requests.post(
                    url=self.pth_plastimatch+"/plastimatch_convert",
                    json={
                        "pth_input": str(pth_in),
                        "pth_output": str(pth_warped),
                        "xf": str(pth_vector_field),
                        },
                    timeout=None
                )
            else:
                raise NotImplementedError("The local plastimatch registration is not implemented yet.")

            # load the deformed image and contours back into the registered phantom.
            if data_name == self.register_on_contour:
                continue
            deformed_data = BrachyPhantom(
                pth_phantom_file=pth_warped,
            ).image_obj
            deformed_data = resampleImage3DOnImage3D(
                deformed_data,
                self._static_data
            )
            if data_name == "image":
                self.registered_phantom.image_obj = deformed_data
            else:
                structure_mask_dict[data_name] = ROIMask(
                    deformed_data.imageArray,
                    name=data_name,
                    spacing=self.registered_phantom.image_obj.spacing,
                    origin=self.registered_phantom.image_obj.origin
                    )

        self.registered_phantom.set_structure_set(structure_mask_dict)

    def evaluate_on_contours(self):
        return super().evaluate_on_contours()

def _load_deformation_field(pth_transform_nrrd: Path) -> Deformation3D:
    import nrrd
    data, header = nrrd.read(str(pth_transform_nrrd), index_order="C")
    spacing = header.get("space directions", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if spacing.shape[0] == 4:
        spacing = spacing[1:, :]
        spacing = np.diag(spacing)

    displacement= VectorField3D(
        imageArray=data,
        # name=pth_transform_nrrd,
        origin=header.get("space origin", [0, 0, 0]),
        spacing=spacing
    )
    return Deformation3D(
        displacement=displacement,
        name=pth_transform_nrrd.stem
    )