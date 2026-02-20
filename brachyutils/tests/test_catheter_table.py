from glob import glob
from pathlib import Path
import numpy as np
from brachyutils.geometry.catheter_utils.catheter_table import DwellPosition, Catheter, CatheterTable

def test_dwells_catheters():
    dwell_dict_0 = {
        "index": 0,
        "position": np.random.rand(3), 
        "relativePos": 5,
        'rotation': [0.0, 0.0, 0.0],
        "time": 45.3,
        "catheter_index":0,
    }
    dwell_dict_1 = {
        "index": 1,
        # "angle": 0,
        "position": np.random.rand(3), 
        "relativePos": 5,
        'rotation': [0.0, 0.0, 0.0],
        "time": np.random.rand(1) * 100,
        "catheter_index":0,
    }
    dwell_dict_2 = {
        "index": 2,
        "angle": 180,
        "position": np.random.rand(3), 
        "relativePos": 5,
        'rotation': [0.0, 0.0, 0.0],
        "time": np.random.rand(1) * 100,
        "catheter_index":0,
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
    
def test_loading_from_dicom():
    # # test loadin from dicom
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_json_delivered = "data_test/test_export_plan/prostate/cat_table_delivered.mrk.json"
    pth_json_all = "data_test/test_export_plan/prostate/cat_table_all.mrk.json"
    pth_json_all_in_ptv = "data_test/test_export_plan/prostate/cat_table_all_in_ptv.mrk.json"

    pth_plan = glob(pth_dicom + "/RP*.dcm")[0]
    catheter_table_delivered = CatheterTable(
        catheter_list=pth_plan,
        from_delivered_dwellpositions=True)
    catheter_table_delivered.write_to_slicer_markup(
        pth_mrk_json=pth_json_delivered
    )
    catheter_table_all = CatheterTable(
        catheter_list=pth_plan,
        from_delivered_dwellpositions=False)
    catheter_table_all.write_to_slicer_markup(
        pth_mrk_json=pth_json_all
    )

    # get the mask of the ptv
    from brachyutils.geometry.phantom_utils import BrachyPhantom
    phant = BrachyPhantom(
        dir_dicom=pth_dicom,
        pth_structures_file=glob(pth_dicom + "/RS*.dcm")[0])
    ptv_mask = phant.get_structure_mask(["CTV"], strict_name_match=False).popitem()[-1]
    catheter_table_all_in_ptv = CatheterTable(
        catheter_list=pth_plan,
        from_delivered_dwellpositions=True)
    catheter_table_all_in_ptv.remove_outside_mask(
            mask=ptv_mask,
            margin_mm=10
        )
    catheter_table_all_in_ptv.write_to_slicer_markup(
        pth_mrk_json=pth_json_all_in_ptv
    )

    # cat_tab_json = CatheterTable(catheter_list=pth_json)
    # cat_tab_json.info()

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

def test_get_from_delivered_dwellpositions():
    pth_dicom = "data_test/prostate-glen-p1-dcm"
    pth_plan = glob(pth_dicom + "/RP*.dcm")[0]
    from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
    cat_table = CatheterTable(catheter_list=pth_plan)
    delivered_cat_table = cat_table.get_from_delivered_dwellpositions()
    assert cat_table.num_catheters >= delivered_cat_table.num_catheters, "Test failed the number of catheters in the delivered table is not equal to the original table."
    assert cat_table.num_dwell_positions >= delivered_cat_table.num_dwell_positions, "Test failed the number of dwell positions in the delivered table is not equal to the original table."

def test_load_dose_rates():
    dir_dicom = Path("data_test/prostate-glen-p1-dcm")
    dir_dose_rates = Path("data_test/prostate-glen-p1-dose")

    cat_tab = CatheterTable(
        catheter_list=list(dir_dicom.glob("RP*.dcm"))[0],
        from_delivered_dwellpositions=False
    )
    cat_tab.load_dose_rates(
        dir_dose_rate=dir_dose_rates
    )
if __name__ == "__main__":
    # test_dwells_catheters()
    # test_loading_from_dicom()
    # test_catheter()
    # test_catheter_to_mrk_json()
    # test_get_from_delivered_dwellpositions()
    test_load_dose_rates()
