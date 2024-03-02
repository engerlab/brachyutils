import os

# from dicompylercore import dicomparser
from glob import glob

import numpy as np
from DicomRTTool.ReaderWriter import DicomReaderWriter  # , ROIAssociationClass


def get_structure_index_range(pth_dir_dicom: str, query_structure_list: list):
    r"""
    Purpose:
        to find the index extent of the structure voxels along each axis using dicom RT structure file.
    Inputs:
        - pth_dir_dicom := path to the directory with the dicom files of a patient.
            it should contain both images and RTSTRUCT file
        - query_structure_list := list of structure names to find the index range of.
    Outputs:
        - structure_index_range:np.array :=  a 3 x 2 array holding the min and max on x, y and axis
            [[x_min, x_max], [y_min, y_max], [z_min, z_max]],
        - body_mask_shape:np.array := 1 x 3 array holding the dimension of the original mask
    Dependencies:
        DicomRTTool: https://www.sciencedirect.com/science/article/abs/pii/S1879850021000485
    """

    pth_dir_dicom = os.path.abspath(pth_dir_dicom)
    assert os.path.exists(pth_dir_dicom), "given dicom path does not exist"
    assert glob(pth_dir_dicom + "/*.dcm"), "there are no dicom files in this directory"

    # load the structure file into an rt_struct object
    dicom_reader = DicomReaderWriter(
        description="getting structure masks", arg_max=True
    )
    dicom_reader.walk_through_folders(pth_dir_dicom)
    all_rois = dicom_reader.return_rois()

    output_dict = {}
    for query_structure_name in query_structure_list:
        # # find the name of the body structure inside the rt_structure object
        dicom_structure_name = [
            name for name in all_rois if query_structure_name in name.lower()
        ]

        # # get the numpy array of the body structure:
        assert len(dicom_structure_name) >= 1, "body contour not found!"
        dicom_reader.set_contour_names_and_associations(
            contour_names=dicom_structure_name
        )

        dicom_reader.get_mask()
        mask_numpy = dicom_reader.mask

        # so we got the mask but the dimensions may not match the dimension of the dose
        # let's get the relative extent of the body mask compared to the whole grid and resample
        # the extents
        structure_index_range = np.zeros([3, 2], dtype=int)
        for i in range(3):
            structure_index_range[i, :] = np.floor(
                np.array(
                    [
                        np.argwhere(mask_numpy == 1)[:, i].min(),
                        # off set of +1 is added to acount for python stopping before range end
                        np.argwhere(mask_numpy == 1)[:, i].max() + 1,
                    ]
                )
            ).astype(int)
            # np.argwhere(mask_numpy==1)[:, i].max()+1]) / np.array(mask_numpy.shape[i]) * self.num_voxels[3-i-1]).astype(int)

        structure_index_range = np.flip(structure_index_range, axis=0)
        output_dict[query_structure_name] = {
            "structure_index_range": structure_index_range,
            "dicom_mask_shape": np.flip(np.array(mask_numpy.shape)),
        }

    return output_dict


def get_strcuture_mask_from_dicom(pth_dir_dicom: str, query_structure_list: list):
    r"""
    Purpose:
        to get the mask of the structures using dicom RT structure file.
    Inputs:
        - pth_dir_dicom := path to the directory with the dicom files of a patient.
            it should contain both images and RTSTRUCT file
        - query_structure_list := list of structure names to find the mask of.
    Outputs:

    """
    pth_dir_dicom = os.path.abspath(pth_dir_dicom)
    assert os.path.exists(pth_dir_dicom), "given dicom path does not exist"
    assert glob(pth_dir_dicom + "/*.dcm"), "there are no dicom files in this directory"

    # load the structure file into an rt_struct object
    dicom_reader = DicomReaderWriter(
        description="getting structure masks", arg_max=True
    )
    dicom_reader.walk_through_folders(pth_dir_dicom)
    all_rois = dicom_reader.return_rois()

    result_dict = {}

    for query_structure_name in query_structure_list:
        # # find the name of the body structure inside the rt_structure object
        dicom_structure_name = [
            name for name in all_rois if query_structure_name in name.lower()
        ]

        # # get the numpy array of the body structure:
        assert len(dicom_structure_name) >= 1, "body contour not found!"
        dicom_reader.set_contour_names_and_associations(
            contour_names=dicom_structure_name
        )

        dicom_reader.get_mask()
        mask_numpy = dicom_reader.mask
        result_dict[query_structure_name] = mask_numpy

    return result_dict


# if __name__ == "__main__":
# app()
