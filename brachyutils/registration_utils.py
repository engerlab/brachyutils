from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Literal, Optional, Union, List, Dict
from collections import defaultdict
import numpy as np

from brachyutils.geometry_utils import BrachyPhantom, phantom_with_empty_image_like
from opentps.core.data._transform3D import Transform3D
from opentps.core.data.images import Deformation3D, VectorField3D
from opentps.core.data.images import ROIMask
# from opentps.core.data import 

class PhantomRegistration(ABC):
    def __init__(
        self,
        static_phantom: Union[BrachyPhantom, str],
        moving_phantom: Union[BrachyPhantom, str],
        register_on_contour: Union[Literal["common"], Optional[str]] = None,
        deformable: bool = False,
        algorithm: Literal["demons", "morphons"] = None,
        backend: Literal["elastix", "plastimatch", "opentps"] = None,
        tryGPU: bool = False,
    ) -> None:
        r"""
        Purpose:
            - A generic class to wrap around all sorts of image registration methods and algorithms. 
            Each registration method should support the attributes of this class and implements its abstract methods.
        Attributes:
            - static_phantom: BrachyPhantom: The static phantom object.
            - moving_phantom: BrachyPhantom: The phantom object that is transformed to match the static phantom.
            - register_on_contour: Optional[str] | "common" = None: The name of the contour to be used in the registration process.
            if this input is provided, contour based registration is used. If "common" is provided, registeration is done
            based on all the common structures.
            - deforemable: bool = False: A flag to indicate whether the registration is deformable or not.
            - algorithm: Literal["Demons", "Morphons", ...] = None The type of registration algorithm.
            - backend: Literal["elastix", "plastimatch", "opentps"] = "opentps" The backend package used to handle 
            the registration process.

        Inputs:
            - dir_plan_export: Union[Path, str]: The path to the dose setup directory.
            - pth_dose_executable: Union[Path, str]: The path to the dose executable.

        Outputs:
            - None

        Functions:
            - register: Register the moving phantom to the static phantom.
            - export_to: Export the registered phantom to a given path.
            - synch_registered_phantom_with_data: Match the image and the contours of the registered phantom.
            - evaluate_on_contours: Evaluate the registration quality by comparing the contours in the registered

        """

        self.static_phantom = static_phantom
        self.moving_phantom = moving_phantom
        self.register_on_contour = register_on_contour
        self.deformable = deformable
        self.algorithm = algorithm
        self.backend = backend
        self.tryGPU = tryGPU
        # the following attributes will be computed during the registration process
        self.registered_phantom: BrachyPhantom = None
        self.deformation: Transform3D = None
        self._static_data = None
        self._moving_data = None
        self._registered_data = None
        # depending on the registration target we will set the static and moving data
        if self.register_on_contour is None:
            self._static_data = self.static_phantom.image_obj
            self._moving_data = self.moving_phantom.image_obj
        elif self.register_on_contour == "common":
            def get_contour_intersections(
                input_phantom:BrachyPhantom,
                common_structures: List[str]
                ) -> ROIMask:
                r"""
                Purpose:
                    - To get the intersection of the common structures in the input phantom.
                """
                mask_dict = input_phantom.get_structure_mask(
                    common_structures,
                    mask_type=np.ndarray)

                all_mask_ndarray = np.stack([
                    mask_dict[structure] for structure in common_structures
                    ])
                all_mask_intersection = np.sum(all_mask_ndarray, axis=0)
                return ROIMask(
                    name=f"common_{input_phantom.image_obj.name}",
                    imageArray=np.swapaxes(all_mask_intersection, 0, 2),
                    origin=input_phantom.image_obj.origin,
                    spacing=input_phantom.image_obj.spacing
                )
                
            common_structures = set(self.static_phantom.structure_names).intersection(  
                set(self.moving_phantom.structure_names)
            )
            if not common_structures:
                raise ValueError("No common structures found in the phantoms.")
            self._static_data = get_contour_intersections(self.static_phantom, common_structures)
            self._moving_data = get_contour_intersections(self.moving_phantom, common_structures)

        else:
            self._static_data = self.static_phantom.get_structure_mask(
                [self.register_on_contour],
                mask_type=ROIMask
            ).get(self.register_on_contour)
            self._moving_data = self.moving_phantom.get_structure_mask(
                [self.register_on_contour],
                mask_type=ROIMask
            ).get(self.register_on_contour)
        if self._static_data is None and self._moving_data is None:
            raise ValueError("The registration target is not defined. If registering based on images, do not provide contour name. else ensure contour is loaded in phantom.")

    @abstractmethod
    def register(self, pth_phantom_export: str | Path = None) -> tuple[BrachyPhantom, Transform3D]:
        r"""
        Purpose:
            - Register the moving phantom to the static phantom.
        Inputs:
            - pth_phantom_export := if provided, the registered phantom will be exported to this directory.
        Outputs:
            - BrachyPhantom: The registered phantom object.
        """
        if self._static_data is None or self._moving_data is None:
            raise ValueError("The static or moving phantom is not defined.")
        pass

    @abstractmethod
    def export_to(
        self,
        dir_registered_phantom: Path | str,
        output_type: Literal[".nrrd", ".dcm"] = ".nrrd") -> None:
        """
        Purpose:
            - To export the obtained registered phantom to a given directory with a given output type.
        Inputs:
            - dir_phantom_export: Union[Path, str]: 
            - output_type: Literal[".nrrd", ".dcm"] = ".nrrd": The output type of the registered phantom.
        Output:
            - None     
        """
        assert self.registered_phantom is not None
        if output_type == ".nrrd":
            self.registered_phantom.export_to(
                dir_nrrd_out=dir_registered_phantom
                )
        elif output_type == ".dcm":
            self.registered_phantom.export_to(
                dir_dicom_out=dir_registered_phantom
            )
        else:
            raise ValueError(f"The output type {output_type} is not supported. please specify .nrrd or .dcm")
    
    @abstractmethod
    def synch_registered_phantom_with_data(self) -> None:
        """
        Purpose:
            - To match the image and the contours of the registered phantom with the registered data.
            If the registration was based on the image, the same deformation will be applied to the contours.
            If the registration was based on the contours, the deformation will be applied to the image
            and the contours will be resampled on the deformed image.  
        Inputs:
            - None
        Output:
            - None
        """
        if self._registered_data is None:
            raise ValueError("The registered data is not defined.")

        # load the registered data into registered phantom
        self.registered_phantom = phantom_with_empty_image_like(
            self.moving_phantom,
            new_pth_image=f"reg_{self.moving_phantom.pth_image.stem}"
            )

        # registration based on image
        if self.register_on_contour is None:
            self.registered_phantom.image_obj = self._registered_data
        # registration based on contour
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

        # deform the image based on the registered structure
        if self.register_on_contour is not None:
            self.registered_phantom.image_obj = self.deformation.deformImage(
                self.registered_phantom.image_obj
                )
            self.registered_phantom.image_obj = resampleImage3DOnImage3D(
                self.registered_phantom.image_obj,
                self._static_data
            )
            # apply the deformation to the image and the rest of the contours.
            if self.registered_phantom.image_obj.name.endswith("_copy"):
                self.registered_phantom.image_obj.name = (
                    self.registered_phantom.image_obj.name.replace("_copy", "")
                )

        structure_mask_dict = self.registered_phantom.get_structure_mask(
            self.registered_phantom.structure_names,
            mask_type=ROIMask
        )

        if not structure_mask_dict:
            print("No structure masks found in the registered phantom.")
            return

        for contour_name in structure_mask_dict:
            # skip the contour that was transformed
            if contour_name == self.register_on_contour:
                continue
            if structure_mask_dict[contour_name] is None:
                continue
            new_mask = self.deformation.deformImage(structure_mask_dict[contour_name])
            new_mask = resampleImage3DOnImage3D(
                new_mask,
                self._static_data
            )
            if new_mask.name.endswith("_copy"):
                new_mask.name = new_mask.name.replace("_copy", "")
            structure_mask_dict[contour_name] = new_mask

        self.registered_phantom.set_structure_set(structure_mask_dict)

    @abstractmethod
    def evaluate_on_contours(self) -> Dict[str, Dict[str, float]]:
        """
        Purpose:
            - To evaluate the registratin quality by comparing the contours in the registered
            phantom with contours in static phantom. The evaluation metrics are Dice score and
            Hausdorff distance.
            Note: This function assumes that there are structures with exactly the same names
            in both registered and static phantoms.
            Note: The decision to resample the registered contours on static contours is made
            by register() function.

        Inputs:
            - None
            expects self.registered_phantom.structure_set and self.static_phantom.structure_set to be defined.

        Output:
            - results: Dict[str, Dict[str, float]]: A dictionary containing the evaluation metrics for each contour.
            in the format below:
            {
                Dice: {
                    "contour_name": dice_score
                    ...
                    "mean": mean_dice_score
                    "std": std_dice_score
                },
                Hausdorff: {
                    "contour_name": hausdorff_distance
                    ...
                    "mean": mean_hausdorff_distance
                    "std": std_hausdorff_distance
                }
            }

        Dependencies:
            - Scipy
        """
        if self.registered_phantom.structure_set is None:
            raise ValueError("The registered phantom structure set is not defined.")
        if self.static_phantom.structure_set is None:
            raise ValueError("The static phantom structure set is not defined.")
        from scipy.spatial.distance import dice
        from monai.metrics import compute_hausdorff_distance

        Dice = defaultdict(list)
        Hausdorf = defaultdict(list)
        
        # find common structures in both phantoms
        common_structures = set(self.registered_phantom.structure_names).intersection(
            set(self.static_phantom.structure_names)
        )

        registered_contours = self.registered_phantom.get_structure_mask(
            common_structures,
            mask_type=ROIMask
            )
        static_contours = self.static_phantom.get_structure_mask(
            common_structures,
            mask_type=ROIMask
            )

        for reg, static in zip(registered_contours, static_contours):
            if reg == self.register_on_contour:
                continue
            dice_score = 1 - dice(
                registered_contours.get(reg).imageArray.flatten(),
                static_contours.get(static).imageArray.flatten()
            )
            hausdorff_distance = float(
                compute_hausdorff_distance(
                    registered_contours.get(reg).imageArray[None, None, ...],
                    static_contours.get(static).imageArray[None, None, ...],
                    percentile=95
                )
                )
            Dice[reg] = (dice_score)
            Hausdorf[reg] = (hausdorff_distance)

        Dice["mean"] = np.array(list(Dice.values()), dtype=float).mean()
        Dice["std"] = np.array(list(Dice.values()), dtype=float).std()
        Hausdorf["mean"] = np.array(list(Hausdorf.values()), dtype=float).mean()
        Hausdorf["std"] = np.array(list(Hausdorf.values()), dtype=float).std()

        return {"Dice": Dice, "Hausdorff": Hausdorf}

from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D
class Registration_OpenTPS(PhantomRegistration):
    def __init__(
        self,
        static_phantom: BrachyPhantom,
        moving_phantom: BrachyPhantom,
        register_on_contour: Union[Literal["common"], Optional[str]] = None,
        deformable: bool = False,
        algorithm: Literal["demons", "morphons", "quick"] = None,
        backend = "opentps",
        tryGPU: bool = False,
        **kwargs
        ):
        r"""
        Purpose:
            - A class to wrap around the OpenTPS image registration method.
        Inputs:
            - static_phantom: BrachyPhantom: The static phantom object.
            - moving_phantom: BrachyPhantom: The phantom object that is transformed to match the static phantom.
            - register_on_contour: Optional[str] | "common" = None: The name of the contour to be used in the registration process.
            if this input is provided, contour based registration is used. If "common" is provided, registeration is done
            based on all the common structures.
            - deforemable: bool = False: A flag to indicate whether the registration is deformable or not.
            - algorithm: Literal["Demons", "Morphons", ...] = None The type of registration algorithm.
            - backend: Literal["opentps"] = "opentps" The backend package used to handle 
            the registration process.
            - dir_phantom_export: Union[Path, str]: The path to the geometry setup directory.
        Outputs:
            - None
        Functions:
            - register: Register the moving phantom to the static phantom.
        Dependencies:
            - OpenTPS
        """

        super().__init__(
            static_phantom,
            moving_phantom,
            register_on_contour if register_on_contour else kwargs.get("register_on_contour", None),
            deformable if deformable else kwargs.get("deformable", False),
            algorithm if algorithm else kwargs.get("algorithm", None),
            backend,
            tryGPU if tryGPU else kwargs.get("tryGPU", False)
            )
        
    def register(
        self,
        pth_phantom_export: Path | str = None,
        **kwargs
        ) -> tuple[BrachyPhantom, Transform3D]:
        r"""
        Purpose:
            - Register the moving phantom to the static phantom using the OpenTPS package.
        Inputs:
            - pth_phantom_export := directory where the registered phantom is exported to.
            
            kwargs entries could include:
            - baseResolution: float = 2.0: The base resolution of the registration algorithm in mm.
            - tryGPU: bool = False: A flag to indicate whether to use the GPU for the registration process.
            - multimodal: bool = False: A flag to indicate whether the registration is multimodal or not.
        Outputs:
            - BrachyPhantom: The registered phantom object.
        """
        if self.deformable:
            assert self.algorithm is not None, "The registration algorithm is not defined."
        
        if self.deformable:

            if self.algorithm == "demons":
                from opentps.core.processing.registration.registrationDemons import RegistrationDemons

                reg = RegistrationDemons(
                    fixed=self._static_data,
                    moving=self._moving_data,
                    baseResolution=kwargs.get("baseResolution", 2.0),
                    tryGPU=self.tryGPU
                )
                self.deformation = reg.compute()

            elif self.algorithm == "morphons":
                from opentps.core.processing.registration.registrationMorphons import RegistrationMorphons
    
                reg = RegistrationMorphons(
                    fixed=self._static_data,
                    moving=self._moving_data,
                    baseResolution=kwargs.get("baseResolution", 2.0),
                    tryGPU=self.tryGPU
                )
                self.deformation = reg.compute()

            elif self.algorithm == "quick":
                from opentps.core.processing.registration.registrationQuick import RegistrationQuick

                reg = RegistrationQuick(
                    fixed=self._static_data,
                    moving=self._moving_data,
                )
                self.deformation = reg.compute(tryGPU=self.tryGPU)
            else:
                raise ValueError("The registration algorithm is not supported. Please choose between 'demons' and 'morphons'.")

        else:
            from opentps.core.processing.registration.registrationRigid import RegistrationRigid
            reg = RegistrationRigid(
                fixed=self._static_data,
                moving=self._moving_data,
                multimodal=kwargs.get("multimodal", False)
            )
            self.deformation = reg.compute()
        # resample the registered image/contour on the static iamge to match the coordinates and contours.
        self._registered_data = resampleImage3DOnImage3D(
                reg.deformed,
                self._static_data,
            )

        self.synch_registered_phantom_with_data()
        if pth_phantom_export is not None:
            self.export_to(pth_phantom_export)

        return self.registered_phantom, self.deformation

    def export_to(
        self,
        dir_registered_phantom: Path | str,
        output_type: Literal[".nrrd", ".dcm"] = ".nrrd") -> None:
        super().export_to(dir_registered_phantom, output_type)

    def synch_registered_phantom_with_data(self) -> None:
        super().synch_registered_phantom_with_data()
        
    def evaluate_on_contours(self):
        return super().evaluate_on_contours()

class Registration_Plastimatch(PhantomRegistration):
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
        Purpose:
            - A class to wrap around the Plastimatch image registration method.
        Inputs:
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
        Outputs:
            - None
        Functions:
            - register: Register the moving phantom to the static phantom.
        Dependencies:
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
        pth_phantom_export: Path | str = None,
        **kwargs
        ) -> tuple[BrachyPhantom, Transform3D]:
        r"""
        Purpose:
            - Register the moving phantom to the static phantom using the Plastimatch package.
        
        Inputs:
            - stage_params_list: List[Dict[str, str]] := a list of dictionaries containing the stage parameters for the registration.
            please look at the plastimatch documentation for the full list of possible stage parameters.
            - pth_phantom_export := directory where the registered phantom is exported to.
        
        Outputs:
            - BrachyPhantom: The registered phantom object.
        """
        # leave some space to figure out the rigidness and options for the registration.

        # need to write out the images for plastimatch to read them.
        # first sort out the paths to the images
        if "temp_data/registration" in str(pth_phantom_export.resolve()):
            dir_temp_data = pth_phantom_export.joinpath("temp/"+self.moving_phantom.pth_image.stem)
        else:
            dir_temp_data = Path(__file__).resolve().parent.parent.joinpath("temp_data/registration")
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
            "vf_out" : f"{str(dir_temp_data.joinpath("vf.nrrd"))}",
        }

        stage_params_list = stage_params_list if stage_params_list else[
            {
                "xform": "bspline"
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
        if pth_phantom_export is not None:
            self.export_to(pth_phantom_export)
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
        Purpose:
            - To match the image and the contours of the registered phantom with the registered data.
            by applying the vector field to the image and the contours using plastimatch convert.
        
        Inputs:
            - pth_vector_field: Path: The path to the vector field file.
        
        Output:
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