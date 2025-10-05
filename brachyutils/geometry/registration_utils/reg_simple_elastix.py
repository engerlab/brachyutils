import numpy as np
from brachyutils.geometry.registration_utils.reg_utils import (
    BrachyPhantomRegistration, phantom_with_empty_image_like
)
from brachyutils.geometry.phantom_utils import BrachyPhantom
from pathlib import Path
from typing import Literal, Optional, Union, List, Dict
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D
from opentps.core.data.images import ROIMask

class Registration_SimpleElastix(BrachyPhantomRegistration):
    def __init__(
        self,
        pth_executable: Path | str,
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
            - pth_executable: Path | str: The path to the simple_elastix executable or the URL where
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
        self.pth_executable = pth_executable if pth_executable else kwargs.popitem("pth_executable", None)

    def register(
        self,
        parameter_maps: List[Dict | str] = None,
        dir_phantom_export: str | Path = None,
        **kwargs
        ):
        r"""
        ### Purpose:
        - Register the moving phantom to the static phantom using the simple_elastix package.
        ### Inputs:
        - parameter_map: list[dict | str] := list of parameter maps for the transformations.
            For strings, we get the default maps from sitk, which are any combination of:
            "translation", "affine", "bspline", "groupwise", "rigid".
            If a dictionary is provided, it can contain the key "default_parameter_map" to specify
            which default map to use. If "default_parameter_map" matched a name of a default map,
            we would load the default map, then override the remaining provided keys and values.
            If the dictionary does not specify "default_parameter_map", we create a parameter map from scratch.
        - dir_phantom_export: str | Path: The path to the directory where the registered phantom is exported to.
        ### Outputs:
        - BrachyPhantom: The registered phantom object.
        """
        parameter_maps = (parameter_maps if parameter_maps is not None
                          else kwargs.pop("parameter_maps", None))
        dir_phantom_export = (dir_phantom_export if dir_phantom_export is not None
                              else kwargs.pop("dir_phantom_export", None))

        # for mask-based registration, we need to use nearest neighbor interpolation.
        if self.register_on_contour is not None and parameter_maps is None:
            parameter_maps = [
                {
                    "default_parameter_map": "translation",
                    "ResampleInterpolator": "FinalNearestNeighborInterpolator",
                }
            ]
        elif self.register_on_contour is not None and parameter_maps is not None:
            for param_map in parameter_maps:
                if isinstance(param_map, dict):
                    param_map["ResampleInterpolator"] = "FinalNearestNeighborInterpolator"
                elif isinstance(param_map, str):
                    param_map = {
                        "default_parameter_map": param_map,
                        "ResampleInterpolator": "FinalNearestNeighborInterpolator",
                    }

        # need to write out the images for simple elastix to read them.
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
        # now we have the paths to the images, we can register them.
        if "http" in self.pth_executable:
            import requests
            http_pth_static = str(pth_static).split("temp_data/registration/")[-1]
            http_pth_moving = str(pth_moving).split("temp_data/registration/")[-1]
            http_pth_output = str(pth_output).split("temp_data/registration/")[-1]
            # Prepare request data for SimpleElastix server
            http_json = {
                "pth_fixed_image": http_pth_static,
                "pth_moving_image": http_pth_moving,
                "pth_output_image": http_pth_output
            }
            # Add parameter maps if provided
            if parameter_maps is not None:
                http_json["parameter_maps"] = parameter_maps
            # Send registration request to SimpleElastix server
            response = requests.post(
                url=f"{self.pth_executable}/elastix_register",
                json=http_json,
                timeout=None  # No timeout, registration might take a while
            )
            if response.status_code != 200:
                raise RuntimeError(f"Registration failed with status code {response.status_code}: {response.text}")
        else:
            raise NotImplementedError("The local simple_elastix registration is not implemented yet.")
        pth_transform_maps = list(dir_temp_data.glob("transform_parameter_*.txt"))
        if not pth_transform_maps:
            raise FileNotFoundError(f"Registration failed, no parameter map files were created.")
        # now we load the registered image and create a new phantom object.
                # load the registered image
        self._registered_data = BrachyPhantom(
            pth_phantom_file=pth_output
        ).image_obj
        # do not resample the registered data on static data. the registration is already done that.
        # self._registered_data = resampleImage3DOnImage3D(
            # self._registered_data,
            # self._static_data,
            # )

        self.synch_registered_phantom_with_data(
            transform_params=pth_transform_maps
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

    def synch_registered_phantom_with_data(self, transform_params:list[str | Path]):
        r"""
        Synchronize the registered phantom with the original data. 
        """
        # we have the path to the parameter map. we make an api call to simple_elastix to 
        # to apply this transformation to the registered phantom (image or contours).
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
            # self._registered_data.imageArray = extract_mask_from_image(self._registered_data)
            # create a new contour based on the registered mask.
            new_contour = ROIMask(
                name=self.register_on_contour,
                imageArray=self._registered_data.imageArray.astype(bool),
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
            # write out the data to be warped by simple elastix
            pth_in = transform_params[0].parent.joinpath(f"{data_name}.nrrd")
            pth_warped = transform_params[0].parent.joinpath(f"{data_name}_warped.nrrd")

            empty_phant = phantom_with_empty_image_like(
                self.moving_phantom,
                new_pth_image=pth_in
            )
            empty_phant.image_obj = all_data.get(data_name)
            empty_phant.write_image_to_nrrd(
                pth_output=pth_in
            )
            if data_name == self.register_on_contour:
                # skip the contour that is used for registration
                continue
            # call simple elastix warp to deform the image and the contours.
            if "http" in self.pth_executable:
                import requests
                pth_in_http = str(pth_in).split("temp_data/registration/")[-1]
                pth_output_http = str(pth_warped).split("temp_data/registration/")[-1]
                pth_transform_http = [str(pth).split("temp_data/registration/")[-1] for pth in transform_params]

                response = requests.post(
                    url=self.pth_executable+"/elastix_warp",
                    json={
                        "pth_input": pth_in_http,
                        "pth_output": pth_output_http,
                        "pth_transform_maps": pth_transform_http,
                        },
                    timeout=None
                )
                if response.status_code != 200:
                    raise RuntimeError(f"Registration failed with status code {response.status_code}: {response.text}")
            else:
                raise NotImplementedError("The local simple elastix registration is not implemented yet.")

            deformed_data = BrachyPhantom(
                pth_phantom_file=pth_warped,
            ).image_obj
            if data_name == "image":
                self.registered_phantom.image_obj = deformed_data
            else:
                structure_mask_dict[data_name] = ROIMask(
                    imageArray=deformed_data.imageArray.astype(bool),
                    name=data_name,
                    spacing=deformed_data.spacing,
                    origin=deformed_data.origin
                    )
        self.registered_phantom.set_structure_set(structure_mask_dict)

    def evaluate_on_contours(self):
        r"""
        See `BrachyPhantomRegistration.evaluate_on_contours` for more details.
        """
        return super().evaluate_on_contours()
    
from opentps.core.data.images import Image3D

def extract_mask_from_image(image_obj: Image3D) -> np.ndarray:
    r"""
    Simple Elastix generates a floating point image as output, even if the input is a binary mask.
    This function extracts a binary mask from the floating point image by applying a threshold.
    We only take the top 15% of the intensity values to create the mask.
    """
    if not hasattr(image_obj, 'imageArray'):
        raise ValueError("The image object does not have an 'imageArray' attribute.")
    # Assuming the imageArray is a numpy array, we can create a mask based on a threshold.
    from opentps.core.processing.segmentation.segmentation3D import applyThreshold
    import numpy as np
    threshold_min = 0.5
    # threshold_min = np.percentile(image_obj.imageArray[image_obj.imageArray!=0], [90])
    # threshold_min = image_obj.imageArray[image_obj.imageArray!=0].max()-image_obj.imageArray[image_obj.imageArray!=0].std()
    threshold_max = np.percentile(image_obj.imageArray[image_obj.imageArray!=0], [100])
    mask = applyThreshold(image_obj, thresholdMin=threshold_min, thresholdMax=threshold_max)
    return mask.imageArray.astype(bool)
