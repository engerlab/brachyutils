from brachyutils.geometry.registration_utils.reg_utils import (
    BrachyPhantomRegistration, phantom_with_empty_image_like
)
from brachyutils.geometry.phantom_utils import BrachyPhantom
from pathlib import Path
from typing import Literal, Optional, Union
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D

class Registration_SimpleElastix(BrachyPhantomRegistration):
    def __init__(
        self,
        pth_simple_elastix: Path | str,
        static_phantom: BrachyPhantom,
        moving_phantom: BrachyPhantom,
        register_on_contour: Union[Literal["common"], Optional[str]] = None,
        deformable: bool = False,
        backend = "simple_elastix",
        tryGPU: bool = False,
        **kwargs
        ):
        r"""
        Purpose:
            - A class to wrap around the simple_elastix image registration method.
        Inputs:
            - pth_simple_elastix: Path | str: The path to the simple_elastix executable or the URL where
            the simple_elastix server is running.
            - static_phantom: BrachyPhantom: The static phantom object.
            - moving_phantom: BrachyPhantom: The phantom object that is transformed to match the static phantom.
            - register_on_contour: Optional[str] | "common" = None: The name of the contour to be used in the registration process.
            if this input is provided, contour based registration is used. If "common" is provided, registeration is done
            based on all the common structures.
            - deformable: bool = False: A flag to indicate whether the registration is deformable or not.
            - algorithm: Literal["Demons", "Morphons", ...] = None The type of registration algorithm.
            - backend: Literal["simple_elastix"] = "opentps" The backend package used to handle 
            the registration process.
            - dir_phantom_export: Union[Path, str]: The path to the geometry setup directory.
        Outputs:
            - None
        Functions:
            - register: Register the moving phantom to the static phantom.
        Dependencies:
            - simple_elastix
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
        self.pth_simple_elastix = pth_simple_elastix if pth_simple_elastix else kwargs.popitem("pth_simple_elastix", None)

    def register(
        self,
        parameter_map: str = "translation",
        dir_phantom_export: str | Path = None,
        ):
        r"""
        ### Purpose:
        - Register the moving phantom to the static phantom using the simple_elastix package.
        ### Inputs:
        - parameter_map: str: not sure what this is, but is a string that is passed to the simple_elastix executable.
        - dir_phantom_export: str | Path: The path to the directory where the registered phantom is exported to.
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
            dir_temp_data = dir_phantom_export.joinpath("temp/"+self.moving_phantom.pth_image.stem)                        
        # if "temp_data/registration" in str(dir_phantom_export.resolve()):
        #     dir_temp_data = dir_phantom_export.joinpath("temp/"+self.moving_phantom.pth_image.stem)
        # else:
        #     dir_temp_data = Path(__file__).resolve().parent.parent.joinpath("temp_data/registration")

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
        
        # now we have the paths to the images, we can register them.
        if "http" in self.pth_simple_elastix:
            import requests
            http_pth_static = str(pth_static).split("temp_data/registration/")[-1]
            http_pth_moving = str(pth_moving).split("temp_data/registration/")[-1]
            http_pth_output = str(pth_output).split("temp_data/registration/")[-1]
            response = requests.post(
                self.pth_simple_elastix+"/elastix_register",
                json={
                    "pth_fixed_image": http_pth_static,
                    "pth_moving_image": http_pth_moving,
                    "parameter_map": parameter_map,
                    "pth_output_image": http_pth_output
                }
            )
        else:
            raise NotImplementedError("The local simple_elastix registration is not implemented yet.")
        if response.status_code != 200:
            raise RuntimeError(f"Registration failed with status code {response.status_code}: {response.text}")

        # now we load the registered image and create a new phantom object.
                # load the registered image
        self._registered_data = BrachyPhantom(
            pth_phantom_file=pth_output
        ).image_obj
        self._registered_data = resampleImage3DOnImage3D(
            self._registered_data,
            self._static_data,
            )
        self.deformation = _load_deformation_field(
            dir_temp_data.joinpath("vf.nrrd")
            )
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
        r"""
        See `BrachyPhantomRegistration.export_to` for more details.
        """
        super().export_to(dir_registered_phantom, output_type)

    def synch_registered_phantom_with_data(self):
        r"""
        Synchronize the registered phantom with the original data.
        """
        super().synch_registered_phantom_with_data()

    def evaluate_on_contours(self):
        r"""
        See `BrachyPhantomRegistration.evaluate_on_contours` for more details.
        """
        return super().evaluate_on_contours()