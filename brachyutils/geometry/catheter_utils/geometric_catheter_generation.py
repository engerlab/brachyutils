"""
line_connectors.py
==================
Pipeline:
  1. Accept a list of polygon meshes (trimesh.Trimesh objects or loaded from STL/OBJ).
  2. Fit an Oriented Bounding Box (OBB) around all meshes → derive top + bottom planes.
  3. Sample a regular GRID_N x GRID_N grid of candidate line endpoints on each plane.
  4. Optionally enforce lines perpendicular to the planes (parallel to OBB Z-axis).
  5. Discard any line that intersects a mesh OR passes within DANGER_DIST of any mesh.
  6. Export everything as STL files ready for Blender / 3D Slicer.

Dependencies:
    pip install trimesh[easy] scipy
"""
from typing import List, Dict
import os
import numpy as np
import trimesh
import trimesh.creation
import trimesh.transformations as tf
from trimesh.ray.ray_triangle import RayMeshIntersector
from scipy.spatial import cKDTree
from pathlib import Path
from opentps.core.data._roiContour import ROIContour
from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay

# ══════════════════════════════════════════════════════
#  PARAMETERS  — tune these
# ══════════════════════════════════════════════════════
# GRID_N         = 5      # NxN candidate lines  (5 → 25 candidates)
# DANGER_DIST    = 5.0    # mm: lines closer than this to any mesh are discarded
# TUBE_RADIUS    = 0.5    # visual radius of exported line tubes
PROX_SAMPLES   = 40     # samples along each line for proximity check
# PERP_LINES     = False  # True → lines perpendicular to planes (parallel to OBB Z)
# STL_OUT_DIR    = "stl_output"


# ══════════════════════════════════════════════════════
#  STEP 1 — OBB → two bounding planes
# ══════════════════════════════════════════════════════

def obb_planes(meshes: list) -> tuple:
    """
    Fit an OBB around the union of all meshes.

    Returns
    -------
    origin_top : (3,)  point on top plane
    origin_bot : (3,)  point on bottom plane
    normal     : (3,)  unit normal shared by both planes (top → bottom direction)
    obb_T      : (4,4) OBB transform (world ← OBB local frame)
    extents    : (3,)  OBB full extents [ex, ey, ez]
    """
    combined  = trimesh.util.concatenate(meshes)
    obb       = combined.bounding_box_oriented
    T         = obb.primitive.transform
    extents   = obb.primitive.extents

    R         = T[:3, :3]
    centre    = T[:3,  3]
    z_axis    = R[:, 2]
    half_z    = extents[2] / 2.0

    origin_top = centre + z_axis * half_z
    origin_bot = centre - z_axis * half_z
    normal     = z_axis / np.linalg.norm(z_axis)

    return origin_top, origin_bot, normal, T, extents


# ══════════════════════════════════════════════════════
#  STEP 2 — Grid sampling on a plane
# ══════════════════════════════════════════════════════

def grid_on_plane(plane_origin: np.ndarray,
                  obb_T: np.ndarray,
                  extents: np.ndarray,
                  n: int) -> np.ndarray:
    """
    Sample an NxN grid of 3-D points on a plane, staying inside the OBB face.

    Parameters
    ----------
    plane_origin : (3,)  point on the plane (e.g. OBB top/bottom face centre)
    obb_T        : (4,4) OBB transform (provides X/Y in-plane axes)
    extents      : (3,)  OBB extents [ex, ey, ez]
    n            : int   number of grid points per axis

    Returns
    -------
    pts : (N*N, 3)
    """
    R    = obb_T[:3, :3]
    x_ax = R[:, 0]
    y_ax = R[:, 1]
    ex, ey = extents[0], extents[1]

    # Inset slightly from edges
    us = np.linspace(-ex/2 + ex/(2*n), ex/2 - ex/(2*n), n)
    vs = np.linspace(-ey/2 + ey/(2*n), ey/2 - ey/(2*n), n)
    UU, VV = np.meshgrid(us, vs)
    pts = (plane_origin
           + UU.ravel()[:, None] * x_ax
           + VV.ravel()[:, None] * y_ax)
    return pts  # (n*n, 3)


# ══════════════════════════════════════════════════════
#  STEP 3 — Collision + proximity filter
# ══════════════════════════════════════════════════════

def _sample_segment(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return p0 + t[:, None] * (p1 - p0)


def line_is_invalid(p0: np.ndarray,
                    p1: np.ndarray,
                    meshes: list,
                    danger_dist: float,
                    n_samples: int = PROX_SAMPLES) -> bool:
    """
    Return True if line p0→p1:
      - intersects any mesh face, OR
      - passes within `danger_dist` of any mesh vertex.

    Uses BVH-accelerated ray casting (trimesh) + KD-tree proximity (scipy).
    """
    direction = p1 - p0
    length    = np.linalg.norm(direction)
    if length < 1e-9:
        return False

    ray_dir = direction / length
    pts     = _sample_segment(p0, p1, n_samples)

    for mesh in meshes:
        # --- Ray intersection ---
        intersector = RayMeshIntersector(mesh)
        locs, _, _  = intersector.intersects_location(
            ray_origins    = p0[None],
            ray_directions = ray_dir[None],
        )
        if len(locs):
            t_hits = np.dot(locs - p0, ray_dir)
            if np.any((t_hits > 1e-6) & (t_hits < length - 1e-6)):
                return True   # intersects this mesh

        # --- Proximity to mesh vertices ---
        tree = cKDTree(mesh.vertices)
        dists, _ = tree.query(pts, k=1)
        if np.any(dists < danger_dist):
            return True   # too close

    return False


# ══════════════════════════════════════════════════════
#  STEP 4 — Line → tube geometry
# ══════════════════════════════════════════════════════

def line_to_tube(p0: np.ndarray,
                    p1: np.ndarray,
                    radius: float,
                    sections: int = 12) -> trimesh.Trimesh | None:
    """Create a cylindrical tube mesh from p0 to p1."""
    direction = p1 - p0
    length    = np.linalg.norm(direction)
    if length < 1e-9:
        return None

    cyl  = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    z    = np.array([0.0, 0.0, 1.0])
    axis_dir = direction / length
    angle    = np.arccos(np.clip(np.dot(z, axis_dir), -1.0, 1.0))
    cross    = np.cross(z, axis_dir)

    if np.linalg.norm(cross) < 1e-9:
        R = np.eye(4)
    else:
        R = tf.rotation_matrix(angle, cross)

    T = tf.translation_matrix((p0 + p1) / 2.0)
    cyl.apply_transform(T @ R)
    return cyl


# ══════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════

def build_line_connectors(
    meshes:List[trimesh.Trimesh],
    grid_n:int ,
    danger_dist:float,
    perpendicular:bool,
    out_dir:str | Path = None,
    tube_radius:float = 3.0,
    ) -> tuple[dict, list]:
    """
    ### Purpose:
    - Given a set of 3D meshes, automatically generate a set of
    straight line connectors between two bounding planes,
    while avoiding collisions and close proximity to the meshes.
    Export everything as STL files for visualization.
    Full pipeline: OBB planes → grid → filter → export STL.

    ### Inputs:
    - meshes: List[trimesh.Trimesh] := list of trimesh.Trimesh
    - grid_n: int :=  N for NxN grid of candidate lines
    - danger_dist: float := discard lines within this distance of any mesh
    - tube_radius: float := radius of exported tube geometry
    - perpendicular: bool := if True, lines run parallel to OBB Z axis (perpendicular to planes)
    - out_dir: str := directory for STL output

    ### Outputs:
    exported: Dict[str, str]:= dictionary of {label: filepath} for all exported STL files
    valid_lines: List[Tuple[np.ndarray, np.ndarray]] := list of (p0, p1) tuples
    """
    os.makedirs(out_dir, exist_ok=True) if out_dir is not None else None

    # ── 1. OBB planes ───────────────────────────────────────────────────────
    o_top, o_bot, normal, obb_T, extents = obb_planes(meshes)

    # ── 2. Grid points on each plane ────────────────────────────────────────
    top_pts = grid_on_plane(o_top, obb_T, extents, grid_n)
    bot_pts = grid_on_plane(o_bot, obb_T, extents, grid_n)

    if perpendicular:
        # Project bottom points onto the top plane along normal,
        # forcing all lines to be parallel to the OBB Z axis.
        bot_pts_proj = np.array([
            p - np.dot(p - o_top, normal) * normal for p in bot_pts
        ])
        pairs = list(zip(bot_pts_proj, bot_pts))
    else:
        # Free straight lines: connect matching grid indices
        pairs = list(zip(top_pts, bot_pts))

    # ── 3. Filter colliding / too-close lines ───────────────────────────────
    valid_lines = [
        (p0, p1) for p0, p1 in pairs
        if not line_is_invalid(p0, p1, meshes, danger_dist)
    ]
    n_total = len(pairs)
    n_valid = len(valid_lines)
    print(f"Candidates: {n_total}  |  Valid (kept): {n_valid}  |  Discarded: {n_total - n_valid}")
    return valid_lines

    # ── 4. Export STL files ──────────────────────────────────────────────────
    # exported = {}

    # # Input meshes
    # for i, m in enumerate(meshes):
    #     path = os.path.join(out_dir, f"mesh_{i:02d}.stl")
    #     m.export(path)
    #     exported[f"mesh_{i:02d}"] = path

    # # Bounding planes as thin flat boxes
    # for label, centre in [("plane_top", o_top), ("plane_bot", o_bot)]:
    #     ex, ey, _ = extents
    #     box = trimesh.creation.box(extents=[ex, ey, 0.2])
    #     Tbox          = obb_T.copy()
    #     Tbox[:3, 3]   = centre
    #     box.apply_transform(Tbox)
    #     path = os.path.join(out_dir, f"{label}.stl")
    #     box.export(path)
    #     exported[label] = path

    # # Line tubes
    # for idx, (p0, p1) in enumerate(valid_lines):
    #     tube = line_to_tube(p0, p1, tube_radius)
    #     if tube is not None:
    #         path = os.path.join(out_dir, f"line_{idx:03d}.stl")
    #         tube.export(path)
    #         exported[f"line_{idx:03d}"] = path

    # print(f"\nSTL files written to: {out_dir}/")
    # for k, v in exported.items():
    #     print(f"  {v}")

    # return valid_lines, exported

def gen_catheter_table_from_contours(
    contour_dict: Dict[str, ROIContour],
    grid_n:int,
    danger_dist_mm:float = 3.0,
    perpendicular:bool = True,
    out_stl_dir:str = None,
    catheter_radius:float = 2.0,
    ) -> CatheterTable:
    r"""
    ### Purpose
    - Given a dictionary of ROIContours, generate a CatheterTable by:
      1. Extracting the contour vertices as meshes.
      2. Running the `build_line_connectors` pipeline to get valid line segments.
      3. Converting valid line segments into Catheter and DwellPosition objects.
    
    ### Inputs
    - contour_dict: Dict[str, ROIContour] := dictionary of ROIContours (e.g. from TPS)
    - grid_n: int := number of candidate lines per plane axis (total candidates = grid_n^2)
    - danger_dist_mm: float := minimum allowed distance (mm) from any contour vertex
    - perpendicular: bool := if True, lines run parallel to OBB Z axis (perpendicular to planes)
    - out_stl_dir: str := if provided, directory to export STL files of meshes + lines
    - catheter_radius: float := visual radius of exported line tubes (mm)
    """
    
    mesh_list = []
    for name, contour in contour_dict.items():
        vertices, faces = get_vertices_faces_from_polygon_mesh(contour.polygonMesh)
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            face_colors=contour.color,
            )
        mesh_list.append(mesh)
    # valid_lines = build_line_connectors(
    #     meshes=mesh_list,
    #     grid_n=grid_n,
    #     danger_dist=danger_dist_mm,
    #     tube_radius=catheter_radius,
    #     perpendicular=perpendicular,
    #     out_dir=out_stl_dir
    # )
    
    if out_stl_dir is not None:
        # for i, line in enumerate(valid_lines):
        #     tube = line_to_tube(line[0], line[1], catheter_radius)
        #     if tube is not None:
        #         path = os.path.join(out_stl_dir, f"line_{i:03d}.stl")
        #         tube.export(path)
        for i, mesh in enumerate(mesh_list):
            path = os.path.join(out_stl_dir, f"mesh_{i:02d}.stl")
            mesh.export(path)
            

def get_vertices_faces_from_polygon_mesh(polygon_mesh:ArrayLike) -> np.ndarray:
    vertices=[]
    for polygon_2d in polygon_mesh:
        Xs = polygon_2d[0::3]
        Ys = polygon_2d[1::3]
        Zs = polygon_2d[2::3]
        vertices_2d = np.column_stack((Xs, Ys, Zs))
        vertices.append(vertices_2d)
    vertices = np.vstack(vertices)
    faces = Delaunay(vertices[:, :2]).simplices
    return vertices, faces

# ══════════════════════════════════════════════════════
#  HOW TO USE WITH YOUR OWN MESHES
# ══════════════════════════════════════════════════════
# 
# Option A – load from STL files:
#   meshes = [trimesh.load("mesh_a.stl"), trimesh.load("mesh_b.stl"), ...]
#
# Option B – build from numpy vertex arrays (e.g. your existing coords):
#   import trimesh.convex
#   meshes = [trimesh.convex.convex_hull(vertices_array) for vertices_array in your_arrays]
#
# Option C – already have faces:
#   meshes = [trimesh.Trimesh(vertices=verts, faces=faces) for verts, faces in your_data]
#
# Then call:
#   exported, valid_lines = build_line_connectors(
#       meshes        = meshes,
#       grid_n        = 5,         # 5x5 = 25 candidate lines
#       danger_dist   = 5.0,       # mm
#       tube_radius   = 0.5,
#       perpendicular = False,     # True = lines perpendicular to planes
#       out_dir       = "stl_output",
#   )
#
# ── BLENDER import ──────────────────────────────────────────────────────────
# File → Import → STL → select all files in stl_output/
# To assign colors per object: select object → Material Properties → New → Base Color
# Suggested colors:
#   mesh_*    → grey   (0.5, 0.5, 0.5)
#   plane_*   → blue   (0.2, 0.4, 0.8, alpha=0.4)
#   line_*    → orange (1.0, 0.5, 0.1)
#
# ── 3D Slicer import ────────────────────────────────────────────────────────
# File → Add Data → choose STL files → OK
# In "Models" module, assign color + opacity per model.
#
# ── Optional: convert to single VTP for 3D Slicer ───────────────────────────
# import pyvista as pv
# combined = pv.PolyData()
# for path in exported.values():
#     combined += pv.read(path)
# combined.save("all_in_one.vtp")