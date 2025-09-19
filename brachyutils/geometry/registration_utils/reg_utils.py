from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Union, List, Dict
from collections import defaultdict
import numpy as np

from brachyutils.geometry.phantom_utils import BrachyPhantom
from brachyutils.geometry.phantom_utils import phantom_with_empty_image_like
from opentps.core.data._transform3D import Transform3D
from opentps.core.data.images import ROIMask
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D

class BrachyPhantomRegistration(ABC):
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
    def register(self, dir_phantom_export: str | Path = None) -> tuple[BrachyPhantom, Transform3D]:
        r"""
        Purpose:
            - Register the moving phantom to the static phantom.
        Inputs:
            - dir_phantom_export := if provided, the registered phantom will be exported to this directory.
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
            if data_name == self.register_on_contour:
                continue
            deformed_data = self.deformation.deformImage(all_data[data_name])
            deformed_data = resampleImage3DOnImage3D(
                deformed_data,
                self._static_data,
            )
            if deformed_data.name.endswith("_copy"):
                deformed_data.name = deformed_data.name.replace("_copy", "")
            if data_name == "image":
                self.registered_phantom.image_obj = deformed_data
            else:
                structure_mask_dict[data_name] = ROIMask(
                    name=data_name,
                    imageArray=deformed_data.imageArray.astype(bool),
                    origin=deformed_data.origin,
                    spacing=deformed_data.spacing,
                )
        # update the registered phantom structure set with the deformed contours                
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
            # if reg == self.register_on_contour:
            #     continue
            dice_score = 1 - dice(
                u=registered_contours.get(reg).imageArray.flatten(),
                v=static_contours.get(static).imageArray.flatten()
            )
            hausdorff_distance = float(
                compute_hausdorff_distance(
                    registered_contours.get(reg).imageArray[None, None, ...],
                    static_contours.get(static).imageArray[None, None, ...],
                    percentile=95,
                    spacing=self.static_phantom.image_obj.spacing
                )
                )
            Dice[reg] = (dice_score)
            Hausdorf[reg] = (hausdorff_distance)

        Dice["mean"] = np.array(list(Dice.values()), dtype=float).mean()
        Dice["std"] = np.array(list(Dice.values()), dtype=float).std()
        Hausdorf["mean"] = np.array(list(Hausdorf.values()), dtype=float).mean()
        Hausdorf["std"] = np.array(list(Hausdorf.values()), dtype=float).std()

        return {"Dice": Dice, "Hausdorff": Hausdorf}