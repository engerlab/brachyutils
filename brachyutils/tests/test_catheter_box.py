from glob import glob
from pathlib import Path
import numpy as np
from trimesh import Trimesh

from brachyutils.geometry.catheter_utils.catheter_cluster_box_utils import decision_planes_to_ply
from brachyutils.geometry.phantom_utils import BrachyPhantom

def get_test_structure_meshes():
    dir_dicom = Path("data_test/prostate-glen-p1-dcm")
    outdir = "data_test/test_export_plan/prostate/line_connectors_from_contours"
    pth_struct = list(dir_dicom.glob("RS*.dcm"))[0]
    phant:BrachyPhantom = BrachyPhantom(
        dir_dicom=dir_dicom,
        pth_structures_file=pth_struct)
    mesh_dict = phant.get_structure_mask(
        query_structure_list=["CTV", "urethra", "rectum"],
        mask_type=Trimesh, strict_name_match=False)
    return mesh_dict

def test_obb_planes():
    from brachyutils.geometry.catheter_utils.catheter_cluster_box_utils import obb_planes
    from brachyutils.geometry.catheter_utils.catheter_cluster_box_utils import decision_planes_to_ply
    outdir = "data_test/test_export_plan/prostate/line_connectors_from_contours"

    mesh_dict = get_test_structure_meshes()
    decision_plane_dict = obb_planes(
    meshes = mesh_dict,
    margin_mm = 10.0,
    rotation_angle_deg = -15,
    num_planes = 4,
    )
    decision_planes_to_ply(
    out_ply_dir = outdir, 
    decision_plane_dict = decision_plane_dict,
    )


def test_build_line_connectors():
    from brachyutils.geometry.catheter_utils.catheter_cluster_box_utils import build_line_connectors
    import trimesh
    dir_out = "data_test/test_export_plan/prostate/line_connectors"
    np.random.seed(42)
    centers = [[12, 0, 15], [-12, 0, 18], [0, 12, 20], [0, -12, 12], [0, 0, 26]]
    demo_meshes = []
    for c in centers:
        pts  = np.random.randn(50, 3) * 2 + c
        hull = trimesh.convex.convex_hull(pts)
        demo_meshes.append(hull)

    build_line_connectors(
        meshes        = demo_meshes,
        grid_n        = 5,
        danger_dist   = 5.0,
        tube_radius   = 0.5,
        perpendicular = False,
        out_dir       = dir_out,
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

if __name__ == "__main__":
    test_obb_planes()
    # test_build_line_connectors()
    # test_gen_catheter_table_from_contours()
