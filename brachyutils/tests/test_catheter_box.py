from glob import glob
from pathlib import Path
import numpy as np
from brachyutils.geometry.catheter_utils.catheter_table import DwellPosition, Catheter, CatheterTable
from trimesh import Trimesh

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
    test_build_line_connectors()
    test_gen_catheter_table_from_contours()
