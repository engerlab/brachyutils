from pathlib import Path
from typing import Tuple, Literal, Dict, List
import SimpleITK as sitk
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.catheter_setup import CatheterSetUp
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.contour.creator import CatheterContourCreator
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.contour_digitizer import DwellPositionCreator, CatheterTableTimesFiller
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.contour.postprocess import CatheterPostProcessor

import numpy as np
import json

def dicom_to_catheter_table(
    dir_dicom: Path | str,
    ) -> Tuple[dict, CatheterSetUp]:
    r"""
    ### Purpose:
    - To extract the catheter setup from a DICOM plan file.

    ### Inputs:
    - dir_dicom: Path to the DICOM directory that contains a plan file.

    ### Outputs:
    - catheter_table: dict := {
        "step_size": float := the step size between the dwell position (a dicom property),
        "channel_length": float := the length of the catheter channel (a dicom property),
        "catheter_list": list := [
            {
                'index': int,
                'points': list := list of digitization coordinates
                'fit_function' := catheter.digitization.pw_linear_interpolator.PiecewiseLinear3D
                'tip_position': list := coordinates of the firs dwell position.
            } ...
        ],
    }
    - CatheterSetUp: Catheter setup object containing the following catheter information.  
        - catheter_table: list
        - digitization_points: dict := {
            - "Needle_1": {
                list([x, y, z])
                } ...
            }
        - piece_wise_lines: dict := {
            - "Needle_1": catheter.digitization.pw_linear_interpolator.PiecewiseLinear3D :=
                point_pairs : list = [
                        [
                            [x1, y1, z1], [x2, y2, z2]
                        ], ...
                    ],
                points: list = [[x, y, z], ...],
                segment_lengths: list = [float, ...], XXX in mm?
                segment_ranges: list = [... (float_i, float_i+1), ...]
                total_length: float
                }

    """
    dir_dicom = Path(dir_dicom)
    # Create a CatheterSetUp object from the DICOM plan
    cat_setup = CatheterSetUp(dir_dicom)
    return _update_catheter_table(
        catheter_table = cat_setup.catheter_table,
        digitization_points=cat_setup.digitization_points,
        fit_function=cat_setup.piece_wise_lines,
        tips=cat_setup.get_tips_coords(),
        step_size=cat_setup.step_size,
    ), cat_setup

def catheter_setup_to_contour(
        catheter_setup: CatheterSetUp | str | Path,
        processed_folder: str | Path,
        patient_volume_path: str | Path = None,
        dilation: int = 0,
        add_tip_marker_contour: bool = True,
        extend_catheters_to_body: bool = False,
        body_contour_mask: sitk.Image = None,
        catheter_diameter: float = 2.0,
        multiprocess: bool =True,
        write_contours: bool =False,
    ) -> Tuple[sitk.Image, CatheterContourCreator]:
        """
        ### Purpose:
        - Creates a CatheterContourCreator from either a CatheterSetUp object or a path.

        ### Inputs:
        - catheter_setup: Either a CatheterSetUp object or path to patient dicom data
        - processed_folder: The folder where the processed files will be saved, used for 
        nnunet raw dataset creation as well as catheter contours.
        - patient_volume_path: Path to patient volume data if not using dicom data
        - dilation: The number of times the catheter will be dilated.
        - add_tip_marker_contour: Whether to add tip marker contours
        - extend_catheters_to_body: Whether to extend catheters to body surface
        - body_contour_mask: Body contour mask image
        - catheter_diameter: Diameter of catheters in mm
        - multiprocess: boolean indicating whether to use multiprocessing
        - write_contours: boolean indicating whether to write contours to disk. the location
        will be the same as the processed_folder.

        ### Outputs:
        - CatheterContourCreator: Object that contains the following functional information:
        """
        if not isinstance(catheter_setup, (CatheterSetUp, str, Path)):
            raise TypeError(
                f"catheter_setup must be a CatheterSetUp object, str or Path, not {type(catheter_setup)}"
            )
        if isinstance(catheter_setup, (str, Path)):
            catheter_setup = Path(catheter_setup)
        processed_folder = Path(processed_folder)
        if patient_volume_path is None:
            patient_volume_path = processed_folder.joinpath("ct.nrrd")

        creator_kwargs = {
            "patient_volume_path": patient_volume_path,
            "processed_folder": processed_folder,
            "dilation": dilation,
            "add_tip_marker_contour": add_tip_marker_contour,
            "extend_catheters_to_body": extend_catheters_to_body,
            "body_contour_mask": body_contour_mask,
            "catheter_diameter": catheter_diameter,
        }

        if isinstance(catheter_setup, (str, Path)):
            creator_kwargs["patient_path"] = catheter_setup
        else:
            creator_kwargs["catheter_setup"] = catheter_setup

        contour_creator = CatheterContourCreator(**creator_kwargs)
        return contour_creator.create_catheter_contour(
            multiprocess = multiprocess,
            write = write_contours,
            out_path = (
                processed_folder.joinpath("catheters.seg.nrrd") if
                write_contours else None
                ),
        ), contour_creator

def contour_to_catheter_table(
    catheter_contours: sitk.Image | str | Path,
    fit_function: Literal["spline", "linear", "piecewise_linear"] = "spline",
    tip_distal: bool = True,
    multi_class: bool = True,
    tip_class_idx: int = 3,
    catheter_core_class_idx: int = 1, 
    contour_dilation: int = 0,
    step_dwell_position: float = 1.0, # XXX in mm?
    ) -> dict:
    r"""
    ### Purpose:
    - To extract the catheter setup from catheter contours.

    ### Inputs:
    - catheter_contours: Path to the catheter contours file or a SimpleITK image.
    - fit_function: The type of fitting function to use for the catheter contours.
    - tip_distal: Whether to consider the distal end as the tip.
    - multi_class: Whether to use multiple classes for catheter segmentation.
    - tip_class_idx: The class index for the tip.
    - catheter_core_class_idx: The class index for the catheter core.
    - contour_dilation: The number of times to dilate the contour.
    - step_dwell_position: The step size for dwell position creation.

    ### Outputs:
    - catheter_table: list := see dicom_to_catheter_table for details
    - CatheterSetUp: Catheter setup object containing the following catheter information.
    """

    if isinstance(catheter_contours, (str, Path)):
        catheter_contours = Path(catheter_contours)
    elif isinstance(catheter_contours, sitk.Image):
        # since dwell position creator requires a path, we write the image to
        # temp location on disk
        temp_catheter_path = Path("/tmp/catheter_contours.seg.nrrd")
        sitk.WriteImage(
            catheter_contours,
            temp_catheter_path,
            useCompression=True,
        )
        catheter_contours = temp_catheter_path

    dwell_pos_creator = DwellPositionCreator(
    sitk_needles_contour_path=catheter_contours,
    fit_function=fit_function,
    tip_distal=tip_distal,
    multi_class=multi_class,
    tip_class_idx=tip_class_idx,
    catheter_core_class_idx=catheter_core_class_idx,
    contour_dilation=contour_dilation,
    )
    created_needle_dict, solo_components = dwell_pos_creator.create_points_from_contours(
        step_dwell_position,
        for_viz=True
        )
    # Create a CatheterTableTimesFiller object from needle_dict and solo_components
    catheter_table = CatheterTableTimesFiller(created_needle_dict).zerosec_table
    return _update_catheter_table(
        catheter_table=catheter_table,
        digitization_points=created_needle_dict.get("Points from segmentation"),
        fit_function=created_needle_dict["Fitted function params"],
        tips=created_needle_dict["Distal tips"],
        channel_length=None, # contour does not give channel length
        step_size=step_dwell_position,
    )

def ct_to_catheter_table(
    image: sitk.Image | str | Path,
    model_path: str | Path,
    pth_out_contours: str | Path = None,
    fold: List[int] = None,
):
    r"""
    ### Purpose:
    - To extract the catheter setup from a CT image. The catheters will be contoured on
    CT images and the catheter table will be created from the contours.

    ### Inputs:
    - image: Path to the CT image .nrrd file or a SimpleITK image.
    - model_path: Path to the trained model for catheter segmentation.
    - pth_out_contours: Path to the output contours file. If None, the contours will be saved
    to parent directory of image.

    ### Outputs:
    - catheter_table: list := see dicom_to_catheter_table for details
    """
    from ai_assisted_brachy.inference.catheters import CatheterDLSegmentor
    import torch

    # take care of image pathing if needed
    if isinstance(image, (str, Path)):
        ct_path = Path(image)
        if not str(ct_path).endswith(".nrrd"):
            raise ValueError("image must a .nrrd file containing a patient ct image.")
        if not ct_path.exists():
            raise FileNotFoundError(
                f"image file {ct_path} does not exist."
            )
    elif isinstance(image, sitk.Image):
        # since dwell position creator requires a path, we write the image to
        # temp location on disk
        temp_ct_path = Path(pth_out_contours) if pth_out_contours else Path("/tmp/ct_image.nrrd")
        sitk.WriteImage(
            image,
            temp_ct_path,
            useCompression=True,
        )
        ct_path = temp_ct_path
    else:
        raise TypeError(
            f"image must be a sitk.Image, str or Path, not {type(image)}"
        )
    # take care of pth_out_contours
    if pth_out_contours is None:
        pth_out_contours = ct_path.parent.joinpath("catheters.seg.nrrd")
    else:
        pth_out_contours = Path(pth_out_contours)
        if not str(pth_out_contours).endswith("seg.nrrd"):
            raise ValueError("pth_out_contours must be a .seg.nrrd file.")
    pth_out_contours.parent.mkdir(parents=True, exist_ok=True)

    if fold is not None:
        fold = fold if isinstance(fold, list) else [fold]
    else:
        fold = [0, 1, 2, 3, 4] if torch.cuda.is_available() else [0]

    # raise NotImplementedError("this function is not implemented yet")
    predictor = CatheterDLSegmentor(
        model_path=model_path,
        fold_num=fold
    )
    catheter_contours = predictor.predict(
        ct_path=ct_path,
        output_file_name=pth_out_contours
        )
    # post process the contours
    prossed_contours = CatheterPostProcessor(
        reference_ct=None,
        contour_dilation=1, # XXX is it a good default value? 
        log_path=pth_out_contours
        )
    # Post process the AI-generated catheter contours
    results = prossed_contours.postprocess_catheters(
        catheters_contour_path=pth_out_contours
    )
    # Unpack the returned values
    post_processed_ai_contours = results[0]  # Processed contour image
    single_postproc_cat = results[1]         # Individual processed catheters
    post_processed_infos = results[2]        # Post-processing information
    separator_infos = results[3]             # Separator information

    ## Flagging if catheter is abnormally small
    _flag_unsually_small_catheters(single_postproc_cat, pth_out_contours.parent)

    sitk_needles_contour_path = sitk_needles_contour_path.replace(
        ".seg.nrrd",
        "_postprocessed.seg.nrrd"
        )
    sitk.WriteImage(
        post_processed_ai_contours,
        sitk_needles_contour_path,
        useCompression=True
        )

    with open(pth_out_contours.parent.joinpath(
        "ai_generated_catheters_postprocessed_infos.json"), "w") as f:
        json.dump(post_processed_infos, f, indent=4)

    with open(pth_out_contours.parent.joinpath(
        "ai_generated_catheters_separator_infos.json"), "w") as f:
        json.dump(separator_infos, f, indent=4)

    return contour_to_catheter_table(
        catheter_contours=post_processed_ai_contours,
    )

def _update_catheter_table(
    catheter_table: dict | list,
    digitization_points: dict = None,
    fit_function: dict = None,
    tips: dict = None,
    step_size: float = 5.0,
    ) -> dict:
    r"""
    ### Purpose:
    - To update the catheter table with digitization points, fitting lines and tips.
    we assume that the catheters, digitization points per catheter, and fitting lines
    per catheter are all in the same order. Loop through the catheters and add each
    attribute to each catheter.

    ### Inputs:
    - catheter_table: dict
    - digitization_points: dict
    - fit_function: dict
    - tips: dict
    - channel_length

    ### Outputs:
        - catheter_table: dict
    """
    if isinstance(catheter_table, dict):
        catheter_list = catheter_table.get("catheter_list")
    elif isinstance(catheter_table, list):
        catheter_list = catheter_table
    else:
        raise TypeError(
            f"catheter_table must be a list, not {type(catheter_table)}"
        )
    if len(catheter_list) == 0:
        raise ValueError(
            f"catheter_table is empty. Please check the input."
        )
    input_name_list = ["digitization_points", "fit_function", "tips"]
    input_list = [digitization_points, fit_function, tips]
    for i, input_ in enumerate(input_list):
        if input_ is not None:
            if not isinstance(input_, dict):
                raise TypeError(
                    f"{input_name_list[i]} must be a dict, not {type(input_)}"
                )
            if len(catheter_list) != len(list(input_.keys())):
                raise ValueError(
                    f"Number of catheters in {input_name_list[i]} is {len(list(input_.keys()))}\
                        and in catheter table is ({len(catheter_list)}). They do not match."
                )
    list_digitization_points = list(digitization_points.values())
    list_fit_function = list(fit_function.values())
    list_tips = list(tips.values())
    for j, catheter in enumerate(catheter_list):
        catheter["index"] = j
        catheter["points"] = list_digitization_points[j]
        catheter["fit_function"] = list_fit_function[j]
        catheter["tip_position"] = list_tips[j]

    catheter_dict = {}
    catheter_dict["catheter_list"] = catheter_list
    catheter_dict["step_size"] = step_size
    return catheter_dict

class CreatedSetUp:
    """
    A class to represent a created catheter setup with all the necessary information.
    This is mainly here to add a loading functionaility to brachyutils, see CatheterTable
    class initilization in catehetr_utils.py in brachyutils.
    """
    def __init__(self, created_needle_dict:dict, catheter_table:List):
        """
        Args:
            created_needle_dict (dict): Dictionary with the created dwell positions from DwellPositionCreator class.
            catheter_table (List): Catheter table prepared with the CatheterTableTimesFiller class.
        """
        self.created_needle_dict = created_needle_dict
        self.catheter_table = catheter_table

    def to_brachyutils_CatheterTable_format(self) -> dict:
        """
        Convert the catheter table to a brachyutils CatheterTable compatible format.
        Returns:
            dict: Catheter table in the brachyutils CatheterTable format.
        """
        return _update_catheter_table(
            catheter_table=self.catheter_table,
            digitization_points=self.created_needle_dict.get("Points from segmentation"),
            fit_function=self.created_needle_dict["Fitted function params"],
            tips=self.created_needle_dict["Distal tips"],
            step_size=self.created_needle_dict["Step size"],
        )

    def get_non_zero_dwell_positions(self)-> dict:
        """
        Gathering the non 0s dwell piositions in a dict.
        Needed for brachyutils CatheterTable initialization.
        """
        non_zeros_dp = {}
        for key, catheter in zip(self.created_needle_dict["Dwell positions"].keys(), self.catheter_table):
            
            assert np.all(np.array(self.created_needle_dict["Distal tips"][key]) == np.array(catheter["tip_position"])), \
                f"Tip position in created needle dict {self.created_needle_dict['Distal tips'][key]} \
                    does not match catheter table {catheter['tip_position']} for catheter {key}."

            non_zeros_dp[key] = [
                dp["position"] for dp in catheter["dwells"]
                if dp["time"] > 0.0
            ]
        return non_zeros_dp
    
def catheter_table_to_catheter_setup(catheter_table: dict) -> CatheterSetUp:
    r"""
    ### Purpose:
        - To convert a catheter table to a CatheterSetUp object that could be used for contour creation.

    ### Inputs:
        - catheter_table: dict

    ### Outputs:
        - CatheterSetUp: Catheter setup object containing the following catheter information.
            - catheter_table: list
            - digitization_points: dict
            - piece_wise_lines: dict
    """
    # initialize the required attributes for the CatheterSetUp object
    all_digi_points: Dict[list] = {} # used to generate the catheter line repersentation
    step_size: float = None # the step size between the dwell position (a dicom property)
    all_dwell_positions_per_cath: Dict[list] = {} # if a catheter does not have dwells, [] is appended
    catheter_table_with_only_activated_dps: List[dict] = []
    channel_length: float = None
    # loop through cathter table and extract the required attributes
    for i, catheter in enumerate(catheter_table):
        if i == 0:
            channel_length = catheter.get("channel_length")
        needle_id = catheter.get('needle_key', f"Needle_{i+1}")
        digitization_points = catheter.get("digitization_points") or catheter.get("points")
        if digitization_points is None:            
            raise ValueError(
            f"Catheter {needle_id} does not have digitization points."
            )
        all_digi_points[needle_id] = digitization_points
        step_size = catheter.get("step_size", 5.0)

        # Extract dwell positions for current catheter
        dwell_info = catheter.get("dwells", [])
        if not dwell_info:
            print(f"Catheter {needle_id} does not have dwell positions.")
            all_dwell_positions_per_cath[needle_id] = []
            continue
        # Extract positions from dwell info 
        dwell_positions = [dwell.get("position") for dwell in dwell_info]
        all_dwell_positions_per_cath [needle_id] = dwell_positions
        catheter_table_with_only_activated_dps.append(catheter)

    # initialize the CatheterSetUp object and set the required attributes
    cat_setup = CatheterSetUp(setup=False)
    cat_setup.digitization_points = all_digi_points
    cat_setup.step_size = step_size
    cat_setup.channel_length = channel_length
    cat_setup.non_zero_dwell_positions = all_dwell_positions_per_cath
    # compute the remaning attributes
    cat_setup.get_dwell_positions(catheter_table_with_only_activated_dps)
    cat_setup.catheter_table = cat_setup._add_zero_treatment_times(
        catheter_table_with_only_activated_dps
        )
    return cat_setup

def _flag_unsually_small_catheters(
    catheters:List[sitk.Image], 
    processed_patient_out_folder:str | Path,
    ) -> List[int]:
    """
    We identify unusally small catheters by looking at potential outliers
    in catheter volume sizes for a single patient. We identify outliers
    the same way a boxplot would, by taking the interquartile range and
    considering anything outside of 1.5 times the IQR as an outlier.
    """
    processed_patient_out_folder = Path(processed_patient_out_folder)
    catheters_to_check = processed_patient_out_folder.joinpath("catheters_to_check")
    catheter_volumes = _get_catheter_sizes(catheters)
    catheter_volumes = np.array(catheter_volumes)
    q1 = np.percentile(catheter_volumes, 25)
    q3 = np.percentile(catheter_volumes, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    unusual_catheters = []
    for cat_idx, catheter in enumerate(catheters):
        if catheter_volumes[cat_idx] < lower_bound or catheter_volumes[cat_idx] > upper_bound:
            unusual_catheters.append(cat_idx)
            # replaced logger with print
            print([f"WARNING:: Please check for catheter {cat_idx}! Its size is unsual : it is {catheter_volumes[cat_idx]} mm3. \n"])
            catheters_to_check.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(
                catheter, 
                catheters_to_check.joinpath(f"catheter_to_check_{cat_idx}.nrrd"), 
                useCompression=True
                )

    return unusual_catheters

def _get_catheter_sizes(catheters:List[sitk.Image]):
        catheter_volumes = []
        for cat_idx, catheter in enumerate(catheters):
            catheter_array = sitk.GetArrayFromImage(catheter)
            nb_voxels_in_catheter = np.sum(catheter_array != 0)
            volume_catheter = nb_voxels_in_catheter * np.prod(catheter.GetSpacing())
            if volume_catheter < 600:
                print([f"Catheter {cat_idx} size is small : it is {volume_catheter} mm3. \n"])
            catheter_volumes.append(volume_catheter)
        return catheter_volumes

def test_dicom_to_catheter_table():
    r"""
    ### Purpose:
        - To test the dicom_to_catheter_table function.
    """
    pth_dicom_test = "/root/Software/brachyutils/data_test/prostate-glen-p1-dcm"
    _ , catheter_setup = dicom_to_catheter_table(pth_dicom_test)
    
    # Print the catheter setup information
    print(catheter_setup.catheter_table)
    print(catheter_setup.digitization_points)
    print(catheter_setup.piece_wise_lines)

def test_catheter_setup_to_contour():
    r"""
    ### Purpose:
        - To test the catheter_setup_to_contour function.
    """
    pth_dicom_test = "/root/Software/brachyutils/data_test/prostate-glen-p1-dcm"
    pth_out_data = "/root/Software/brachyutils/data_test/test_export_plan/prostate"
    write = True

    _ , catheter_setup = dicom_to_catheter_table(pth_dicom_test)
    # Create a CatheterContourCreator object
    catheter_contour, _ = catheter_setup_to_contour(
        catheter_setup,
        pth_out_data,
        write_contours=write
        )

    # Print the contour information
    print(catheter_contour)

def test_contour_to_catheter_table():
    r"""
    ### Purpose:
        - To test the contour_to_catheter_table function.
    """
    pth_dicom_test = "/root/Software/brachyutils/data_test/prostate-glen-p1-dcm"
    pth_out_data = "/root/Software/brachyutils/data_test/test_export_plan/prostate"
    write = True

    catheter_setup = dicom_to_catheter_table(pth_dicom_test)
    # Create a CatheterContourCreator object
    catheter_contour, _ = catheter_setup_to_contour(
        catheter_setup,
        pth_out_data,
        write_contours=write
        )
    # Create a CatheterSetUp object from the contours
    # pth_contours = pth_out_data.joinpath("catheters.seg.nrrd")
    contour_to_catheter_table(catheter_contours=catheter_contour)

def test_catheter_table_to_catheter_setup():
    r"""
    ### Purpose:
        - To test the catheter_table_to_catheter_setup function.
    """
    pth_dicom_test = "/root/Software/brachyutils/data_test/prostate-glen-p1-dcm"
    _ , catheter_setup = dicom_to_catheter_table(pth_dicom_test)
    catheter_table = catheter_setup.catheter_table
    cat_setup = catheter_table_to_catheter_setup(catheter_table)
    print(cat_setup.catheter_table)
    print(cat_setup.digitization_points)
    print(cat_setup.piece_wise_lines)

if __name__ == "__main__":
    # test_dicom_to_catheter_table()
    # test_catheter_setup_to_contour()
    # test_contour_to_catheter_table()
    test_catheter_table_to_catheter_setup()