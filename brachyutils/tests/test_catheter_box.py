from glob import glob
from pathlib import Path
import numpy as np
from trimesh import Trimesh

from brachyutils.geometry.phantom_utils import BrachyPhantom
from brachyutils.geometry.catheter_utils.config_cathgen import Config_Angled_CathGen
from brachyutils.geometry.catheter_utils.catheter_cluster_box_utils import (
    segment_lines_to_ply, decision_planes_to_ply)
from brachyutils.geometry.catheter_utils.catheter_cluster_box import CatheterClusterBox

def get_test_structure_meshes():
    dir_dicom = Path("data_test/prostate-glen-p1-dcm")
    outdir = "data_test/test_export_plan/prostate/line_connectors_from_contours"
    pth_struct = list(dir_dicom.glob("RS*.dcm"))[0]
    phant:BrachyPhantom = BrachyPhantom(
        dir_dicom=dir_dicom,
        pth_structures_file=pth_struct)
    mesh_dict = phant.get_structure_mask(
        query_structure_list=["CTV", "urethra", "rectum"],
        # query_structure_list=["CTV", "urethra"],
        mask_type=Trimesh, strict_name_match=False)
    return mesh_dict

def test_obb_planes(return_planes=False):
    from brachyutils.geometry.catheter_utils.catheter_cluster_box_utils import obb_planes
    outdir = "data_test/test_export_plan/prostate/line_connectors_from_contours"

    mesh_dict = get_test_structure_meshes()
    decision_plane_dict = obb_planes(
    meshes = mesh_dict,
    margin_mm = 5,
    rotation_angle_deg = 14,
    num_planes = 2,
    )
    decision_planes_to_ply(
    out_ply_dir = outdir, 
    decision_plane_dict = decision_plane_dict,
    )
    if return_planes:
        return decision_plane_dict

def test_get_segment_lines():
    from brachyutils.geometry.catheter_utils.catheter_cluster_box_utils import (
        get_segment_lines, grid_on_plane, segment_lines_to_ply)
    insertion_point_spacing_mm = 10.
    decision_planes = test_obb_planes(return_planes=True)
    outdir = "data_test/test_export_plan/prostate/line_connectors_from_contours"
    inferior_plane_grid = grid_on_plane(
        plane_origin = decision_planes[0]["origin"],
        obb_T = decision_planes[0]["transform"],
        extents = decision_planes[0]["extents"],
        insertion_point_spacing_mm = insertion_point_spacing_mm,
    )
    print("inf normal:      ", decision_planes[0]["normal"])
    print("sup normal:      ", decision_planes[1]["normal"])
    print("inf transform Z: ",  decision_planes[0]["transform"][:3, 2])
    print("sup extents:     ", decision_planes[1]["extents"])
    point_pairs = get_segment_lines(
        departure_plane = decision_planes[0],
        departure_plane_grid = inferior_plane_grid,
        landing_plane = decision_planes[1],
        config_angled_cathgen = Config_Angled_CathGen()
        )
    segment_lines_to_ply(
        out_ply_dir=outdir,
        point_pairs=point_pairs,
        )

def test_generate_candidate_segments():
    from brachyutils.geometry.catheter_utils.catheter_cluster_box_utils import generate_candidate_segments
    outdir = "data_test/test_export_plan/prostate/line_connectors_from_contours"
    mesh_dict = get_test_structure_meshes()
    valid_lines, plane_dict = generate_candidate_segments(
        mesh_dict=mesh_dict,
        insertion_point_spacing_mm=15,
        oar_danger_dist_mm=5,
        target_structures=["CTV"],
        config_angled_cathgen=Config_Angled_CathGen(),
        bb_margin_mm = 5,
        bb_rotation_angle_deg = 12,
        bb_num_planes = 3,
    )
    decision_planes_to_ply(
        out_ply_dir=outdir,
        decision_plane_dict=plane_dict
    )
    segment_lines_to_ply(
        out_ply_dir=outdir,
        point_pairs=valid_lines,
        )

def test_gen_catheter_table_from_contours():
    from brachyutils.geometry.catheter_utils.catheter_cluster_box_utils import gen_catheter_table_from_contours
    from brachyutils.geometry.phantom_utils import BrachyPhantom
    from brachyutils.geometry.catheter_utils.config_cathgen import Config_Angled_CathGen

    dir_dicom = Path("data_test/prostate-glen-p1-dcm")
    outdir = "data_test/test_export_plan/prostate/line_connectors_from_contours"
    pth_struct = list(dir_dicom.glob("RS*.dcm"))[0]
    phant:BrachyPhantom = BrachyPhantom(
        dir_dicom=dir_dicom,
        pth_structures_file=pth_struct)
    mesh_dict = phant.get_structure_mask(
        query_structure_list=["CTV", "urethra", "rectum"],
        mask_type=Trimesh, strict_name_match=False)
    cat_table = gen_catheter_table_from_contours(
        mesh_dict=mesh_dict,
        target_structures=["CTV"],
        grid_n=5,
        out_ply_dir=outdir,
        perpendicular=False,
        config_angled_cathgen=Config_Angled_CathGen()
    )
    
    # let's export the catheter table to json and to .ply for visualization
    cat_table[0].write_to_ply(
        dir_ply=Path(outdir)
    )
    print("debug here")

def test_catheter_cluster_box():
    outdir = "data_test/test_export_plan/prostate/line_connectors_from_contours"
    structure_dict = get_test_structure_meshes()
    insertion_point_spacing_mm = 15
    oar_collision_margin_mm = 5
    target_structure_names = ["CTV"]
    config_angle = Config_Angled_CathGen()
    num_decision_planes=3
    rotation_angle_deg=12

    CatheterClusterBox(
        structure_dict = structure_dict,
        insertion_point_spacing_mm = insertion_point_spacing_mm,
        oar_collision_margin_mm = oar_collision_margin_mm,
        target_structure_names = target_structure_names,
        config_angle = config_angle,
        num_decision_planes=num_decision_planes,
        rotation_angle_deg=rotation_angle_deg,
    )

if __name__ == "__main__":
    # test_obb_planes()
    # test_get_segment_lines()
    # test_generate_candidate_segments()
    # test_gen_catheter_table_from_contours()
    test_catheter_cluster_box()