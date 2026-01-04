from glob import glob
from pathlib import Path

import numpy as np

from brachyutils.geometry.phantom_utils import BrachyPhantom
from brachyutils.geometry.applicator_utils import BrachyApplicator

def test_brachy_phantom():
    # pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_nrrd = "data_test/prostate_glen_p1_ct.nrrd"
    # pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    phantom_obj = BrachyPhantom(
        # dir_dicom=pth_dicom,
        pth_phantom_file=pth_nrrd,
        # pth_structures_file=pth_structure 
    )
    phantom_obj.info()


def test_get_structure_mask():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom, pth_structures_file=pth_structure)
    print(phantom_obj.get_structure_mask(["ctv"], mask_type=np.ndarray))


def test_write_image_to_dicom():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_nrrd = "data_test/prostate_glen_p1_ct.nrrd"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "data_test/test_export_plan/prostate/test_p1_ct"
    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        # pth_phantom_file=pth_nrrd,
        # pth_structures_file=pth_structure
    )
    phantom_obj.write_image_to_dicom(pth_out)

    new_phantom = BrachyPhantom(
        dir_dicom=pth_out,
    )
    new_phantom.is_equal(phantom_obj)

def test_write_image_to_nrrd():
    from time import time
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_out = "data_test/test_export_plan/prostate/prostate_glen_p1_ct.nrrd"
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom)
    t0=time()
    phantom_obj.write_image_to_nrrd(pth_out)
    print(f"time taken to write: {time()-t0}")
    # new_phantom = BrachyPhantom(pth_phantom_file=pth_out)
    # assert (new_phantom.is_equal(phantom_obj)), "Test failed the two phantoms are not equal."

def test_write_structures_to_nrrd():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "data_test/test_export_plan/prostate/prostate_glen_p1_structs.seg.nrrd"
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom, pth_structures_file=pth_structure)
    phantom_obj.write_structures_to_nrrd(pth_out, overlap=True)


def test_write_structures_to_dicom():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "data_test/test_export_plan/prostate/test_p1_dcm"
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom, pth_structures_file=pth_structure)
    phantom_obj.write_image_to_dicom(pth_out)
    phantom_obj.write_structures_to_dicom(pth_out)


def test_read_structures_from_nrrd():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_structures = "data_test/test_export_plan/prostate/prostate_glen_p1_structs.seg.nrrd"
    pth_out = "data_test/test_export_plan/prostate/test_p1_dcm/rs.seg.nrrd"
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom, pth_structures_file=pth_structures)
    print(phantom_obj.info())
    # phantom_obj.write_image_to_nrrd(pth_out)
    phantom_obj.write_structures_to_nrrd(pth_out, overlap=True)

def test_upsample_structure():
    # pth_dicom = "/home/ubuntu/YourLocalHome/Data/prostate/prostate-glen-2023/p8"
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "data_test/test_export_plan/prostate"
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom, pth_structures_file=pth_structure)
    phantom_obj.resample_to(spacing=np.array([1., 1., 1.]))
    phantom_obj.export_to(dir_nrrd_out=pth_out)

def test_write_to_egsphant():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    # pth_dicom = "/home/ubuntu/YourLocalHome/Data/prostate/prostate-glen-2023/p8"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "data_test/test_export_plan/prostate/ct.egsphant"
    # pth_materials = "data_test/prostate-glen-p1-dcm/CTtoDensityProstate.txt"
    # assign_material_from_ct = True
    pth_materials = "admin/constants/structure_materials_prostate.json"
    assign_material_from_ct = False

    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=pth_structure,
    )
    phantom_obj.write_to_egsphant(
        pth_output=pth_out,
        material_dict=pth_materials,
        assign_material_from_ct=assign_material_from_ct,
        resampled_spacing=[1., 1., 1.],
    )
    # from brachyutils.geometry.egsphant_utils import BrachyEgsphant
    # egsphant_obj = BrachyEgsphant(pth_egsphant_file=pth_out)
    # egsphant_obj.write_to_file(Path(pth_out).parent.joinpath("egsphant.seq.nrrd"))

    # new_egsphant = BrachyEgsphant(Path(pth_out).parent.joinpath("egsphant.seq.nrrd"))
    # new_egsphant.write_to_file(pth_out)
    # new_egsphant.is_equal(egsphant_obj)

def test_load_egsphant():
    pth_egsphant = "data_test/prostate-glen-p1-planFiles/ct.egsphant"
    pth_out = "data_test/test_export_plan/prostate/test_ct.egsphant"

    phantom_obj = BrachyPhantom(pth_egsphant_file=pth_egsphant)
    phantom_obj.write_to_egsphant(
        pth_output=pth_out,
    )

def test_crop_phantom():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_structure = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "data_test/test_export_plan"
    phantom_obj = BrachyPhantom(dir_dicom=pth_dicom, pth_structures_file=pth_structure)
    crop_coordinates = np.array([[-100, 100], [-150, 200], [-1280, -1140]])
    phantom_obj.crop_by_coordinates(crop_coordinates)
    phantom_obj.write_image_to_nrrd(pth_out + "/test_ct.nrrd")
    phantom_obj.write_structures_to_nrrd(pth_out + "/test_rs.seg.nrrd")


def test_catheter_table():
    from brachyutils.geometry.catheter_utils.catheter_table import DwellPosition, Catheter, CatheterTable
    dwell_dict_0 = {
        "index": 0,
        # "angle": 0,
        "position": np.random.rand(3), 
        "relativePos": 5,
        'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        "time": 45.3,
        # "weight": 0.003,
    }
    dwell_dict_1 = {
        "index": 1,
        # "angle": 0,
        "position": np.random.rand(3), 
        "relativePos": 5,
        'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        "time": np.random.rand(1) * 100,
        # "weight": 0.003,
    }
    dwell_dict_2 = {
        "index": 2,
        "angle": 180,
        "position": np.random.rand(3), 
        "relativePos": 5,
        'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        "time": np.random.rand(1) * 100,
        # "weight": 0.003,
    }
    dwell_obj = DwellPosition(**dwell_dict_0)
    print(dwell_obj.to_dict())
    
    catheter_dict = {
        "index": 0,
        "dwells": [
            dwell_dict_0,
            dwell_dict_1,
            dwell_dict_2,
        ],
        "points" :[],
        "afterloader_channel_number": 0,
    }
    catheter_obj = Catheter(**catheter_dict)
    print(catheter_obj.to_dict())

    # # test loadin from dicom
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_json = "data_test/test_export_plan/prostate/test_catheter_table.json"
    pth_plan = glob(pth_dicom + "/RP*.dcm")[0]
    catheter_table = CatheterTable(catheter_list=pth_plan)
    catheter_table.write_to_json(pth_json)
    
    cat_tab_json = CatheterTable(catheter_list=pth_json)
    cat_tab_json.info()

def test_catheter():
    from brachyutils.geometry.catheter_utils.catheter_table import Catheter, DwellPosition
    # # create a catheter from tip and last dwell position
    new_catheter = Catheter(
        index=0,
        tip_position=[25, 25, 25],
        last_dwell_coordinate=[0, 0, 0]
    )
    print(new_catheter)
    # create a catheter from digitization points
    coordinates_on_1_axis = np.arange(53, 1.5, -2)
    points = np.stack([
        coordinates_on_1_axis,
        coordinates_on_1_axis,
        coordinates_on_1_axis], axis=-1)
    new_catheter = Catheter(index = 0, points=points)
    print(new_catheter)

def test_catheter_to_mrk_json():
    from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
    pth_dicom = Path("data_test/prostate-glen-p1-dcm").resolve()
    pth_out = "data_test/test_export_plan/prostate/test_catheter_table.mrk.json"
    cat_table = CatheterTable(catheter_list=list(pth_dicom.glob("RP*.dcm"))[0])
    cat_table.write_to_slicer_markup(pth_mrk_json=pth_out)

def test_BrachyApplicator():
    pth_applicator_stl = "data_test/rectal-jgh-planFiles/applicator_0.stl"
    applicator_obj = BrachyApplicator(pth_applicator_stl)
    applicator_obj.info()

def test_BrachyApplicator_to_mac():
    pth_applicator_stl = "data_test/rectal-jgh-planFiles/applicator_0.stl"
    origin = np.array([0, 0, 0])
    rotation = np.array([0, 0, 0])
    material = "Tungsten"
    density = 19.3
    pth_outfile = "data_test/test_export_plan/applicator_0.mac"
    applicator_obj = BrachyApplicator(
        pth_input_file=pth_applicator_stl,
        material=material,
        density=density,
        origin=origin,
        rotation=rotation,
    )
    applicator_obj.to_mac(pth_outfile)


def test_BrachyApplicator_to_stl():
    pth_applicator_stl = "data_test/rectal-jgh-planFiles/applicator_0.stl"
    origin = np.array([0, 0, 0])
    rotation = np.array([90, 1, 0, 0])
    coordinates = np.array([0, 0, 0])
    material = "Tungsten"
    density = 19.3
    pth_outfile = "data_test/test_export_plan/applicator_0_tilted.stl"
    applicator_obj = BrachyApplicator(
        pth_input_file=pth_applicator_stl,
        material=material,
        density=density,
        origin=origin,
        rotation=rotation,
        coordinates=coordinates,
    )
    applicator_obj.to_stl(pth_outfile)


def test_BrachyApplicator_set_rotation():
    pth_applicator_stl = "data_test/rectal-jgh-planFiles/applicator_0.stl"
    origin = np.array([0, 0, 0])
    coordinates = np.array([50, 50, 50])
    rotation = np.array([90, 0, 1, 0])
    rotation_origin = np.array([50, 50, 50])
    material = "Tungsten"
    density = 19.3
    pth_outfile = "data_test/test_export_plan/applicator_0_tilted.stl"
    applicator_obj = BrachyApplicator(
        pth_input_file=pth_applicator_stl,
        material=material,
        density=density,
        origin=origin,
        coordinates=coordinates,
    )
    applicator_obj.set_rotation(rotation, rotation_origin)
    applicator_obj.to_stl(pth_outfile)

def test_load_nifti_image_and_segmentation_file():
    # CT images Abdomen
    pth_img_nifti = Path("data_test/registration/abdomin_mr_ct/tr_mr_image_0001.nii.gz")
    pth_img_out = Path("data_test/test_export_plan/abdomin_mr_ct/tr_mr_image_0001.nrrd")
    pth_label_nifti = Path("data_test/registration/abdomin_mr_ct/tr_mr_label_0001.nii.gz")
    pth_label_out = Path("data_test/test_export_plan/abdomin_mr_ct/tr_mr_label_0001.seg.nrrd")

    # mri images prostate
    # pth_img_nifti = Path("data_test/registration_prostate_mr_us/train_mr_image_case000000.nii.gz")
    # pth_img_out = Path("data_test/test_export_plan/test_mr_image_case000000.nrrd")
    # pth_label_nifti = Path("data_test/registration_prostate_mr_us/train_mr_label_case000000.nii.gz")
    # pth_label_out = Path("data_test/test_export_plan/test_mr_label_case000000.seg.nrrd")
    
    # ultrasound images
    # pth_img_nifti = Path("data_test/registration_prostate_mr_us/train_us_image_case000000.nii.gz")
    # pth_img_out = Path("data_test/test_export_plan/test_us_image_case000000.nrrd")
    # pth_label_nifti = Path("data_test/registration_prostate_mr_us/train_us_label_case000000.nii.gz")
    # pth_label_out = Path("data_test/test_export_plan/test_us_label_case000000.seg.nrrd")

    phantom_obj = BrachyPhantom(
        pth_phantom_file=pth_img_nifti,
        pth_structures_file=pth_label_nifti
        )
    phantom_obj.info()
    phantom_obj.write_image_to_nrrd(pth_img_out)
    phantom_obj.write_structures_to_nrrd(pth_label_out, overlap=True)

def test_resample_to():
    r"""
    Purpose:
        - to see if the structure sampling is working as expected.
        I noticed that some structures (body contour exported to dicom by slicer) and
        represented by opentps are not upsampled properly.
        Will attempt rttools here.
    """
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_structures = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "data_test/test_export_plan/prostate"
    
    origin = None
    spacing = np.array([1., 1., 1.])
    
    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=pth_structures
        )
    phantom_obj.resample_to(origin, spacing, True)
    phantom_obj.export_to(dir_nrrd_out=pth_out)

def test_dicom_rt_tools():

    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_structures = glob(pth_dicom + "/RS*.dcm")[0]
    pth_out = "data_test/test_export_plan/prostate"
    spacing = np.array([1., 1., 1.])
    
    phantom_obj = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=pth_structures
        )

    import DicomRTTool as rt_tools
    dcm_reader = rt_tools.DicomReaderWriter()
    dcm_reader.walk_through_folders(pth_dicom)
    all_rois = dcm_reader.return_rois()
    dcm_reader.set_contour_names_and_associations(contour_names=all_rois)
    dcm_reader.get_images_and_mask()
    image = dcm_reader.images_dictionary.popitem()[1]
    mask_dict = dcm_reader.mask_dictionary
    
    from brachyutils.geometry.phantom_utils import sitk_to_Image3D
    new_mask_dict = {}
    for mask_name in mask_dict:
        new_mask_dict[mask_name] = sitk_to_Image3D(mask_dict[mask_name])
    
    phantom_obj.set_structure_set(new_mask_dict)

    phantom_obj.export_to(dir_nrrd_out=pth_out)


def test_get_delivered_catheter_table():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_plan = glob(pth_dicom + "/RP*.dcm")[0]
    from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
    cat_table = CatheterTable(catheter_list=pth_plan)
    delivered_cat_table = cat_table.get_delivered_catheter_table()
    assert cat_table.num_catheters >= delivered_cat_table.num_catheters, "Test failed the number of catheters in the delivered table is not equal to the original table."
    assert cat_table.num_dwell_positions >= delivered_cat_table.num_dwell_positions, "Test failed the number of dwell positions in the delivered table is not equal to the original table."

def test_generate_sphere_mask():
    from brachyutils.geometry.phantom_utils import generate_sphere_mask
    center = np.array([15, 5, 5])
    radius = 4
    mask = generate_sphere_mask(
        center=center,
        radius=radius,
        gridSize=np.array([20, 20, 10]),
        spacing=np.array([1., 1., 1.]),
        origin=[0, 0, 0],
    )
    for slice in mask.imageArray.swapaxes(0,2):
        print(slice)

def test_load_pet_dicom():
    from brachyutils.geometry.phantom_utils import BrachyPhantom
    dir_pet_dicom = Path("data_test/pet-dcm")
    phantom_obj = BrachyPhantom(dir_dicom=dir_pet_dicom)
    phantom_obj.info()

if __name__ == "__main__":
    # print("testing BrachyPhantom")
    # test_brachy_phantom()
    # test_get_structure_mask()
    # test_write_image_to_dicom()
    # test_write_image_to_nrrd()
    # test_write_structures_to_nrrd()
    # test_upsample_structure()
    # test_write_structures_to_dicom()
    # test_read_structures_from_nrrd()
    test_write_to_egsphant()
    # test_load_egsphant()
    # test_crop_phantom()
    # print("testing CatheterTable")
    # test_catheter_table()
    # test_catheter()
    # print("testing BrachyApplicator")
    # test_BrachyApplicator()
    # test_BrachyApplicator_to_mac()
    # test_BrachyApplicator_to_stl()
    # test_BrachyApplicator_set_rotation()
    # test_load_nifti_image_and_segmentation_file()
    # test_resample_to()
    # test_dicom_rt_tools()
    # test_catheter_to_mrk_json()
    # test_get_delivered_catheter_table()
    # test_generate_sphere_mask()
    # test_load_pet_dicom()