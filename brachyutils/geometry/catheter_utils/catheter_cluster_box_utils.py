from collections import defaultdict
from typing import List, Dict
import numpy as np
import trimesh
import trimesh.creation
import trimesh.transformations as tf
from trimesh.ray.ray_triangle import RayMeshIntersector
from scipy.spatial import cKDTree
from pathlib import Path
from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable, Catheter
from brachyutils.geometry.catheter_utils.config_cathgen import Config_Angled_CathGen

# ══════════════════════════════════════════════════════
#  PARAMETERS  — tune these
# ══════════════════════════════════════════════════════
# GRID_N         = 5      # NxN candidate lines  (5 → 25 candidates)
# DANGER_DIST    = 5.0    # mm: lines closer than this to any mesh are discarded
# TUBE_RADIUS    = 0.5    # visual radius of exported line tubes
PROX_SAMPLES   = 40     # samples along each line for proximity check
# PERP_LINES     = False  # True → lines perpendicular to planes (parallel to OBB Z)
# STL_OUT_DIR    = "stl_output"

def obb_planes(
    meshes: list,
    margin_mm: float = 10.0,
    rotation_angle_deg: float = 0,
    num_planes: int = 2,
) -> dict:
    """
    ### Purpose:
    - Build a rotated box that contains the original unrotated meshes.
    - The method ensures that the rotated box still contains the original meshes by applying
    the inverse of the rotation to the meshes before computing the bounding box, 
    and then applying the original rotation to the resulting box.

    ### Inputs:
    - meshes: list of trimesh.Trimesh objects to fit the box around.
    - margin_mm: float = 10.0 := margin around the meshes (mm).
    - rotation_angle_deg: float = 0 := rotation angle around the world X axis (degrees).
      This value should be less than 15 degrees.
    - num_planes: int = 2 := number of decision planes to define in the catheter box.
      At least there are 2 planes: inferior plane and superior plane.

    ### Outputs:
    - decision_plane_dict : dict mapping plane indices to their origin, normal, and box transform
    """
    if np.abs(rotation_angle_deg) > 15:
        raise ValueError(
            "rotation_angle_deg should be less than 15 degrees to avoid excessive distortion of the catheter box."
        )
    if margin_mm < 0:
        raise ValueError("margin_mm must be non-negative.")
    if len(meshes) == 0:
        raise ValueError("meshes must contain at least one mesh.")

    # Stack all vertices once: fastest path, no mesh copies
    vertices = np.vstack([np.asarray(mesh.vertices) for mesh in meshes.values()])
    if vertices.shape[0] == 0:
        raise ValueError("Input meshes contain no vertices.")

    # 1. Axis-aligned bounding box in regular coordinates
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)], dtype=float)
    centre = bounds.mean(axis=0)

    # 2. Rotation axis: world x-axis through AABB center
    rotation_axis = np.array([1.0, 0.0, 0.0], dtype=float)

    # 3-4. Negated rotation about that axis through the center
    angle_rad = np.deg2rad(rotation_angle_deg)
    T_neg = trimesh.transformations.rotation_matrix(
        angle=-angle_rad,
        direction=rotation_axis,
        point=centre
    )

    # 5-6. Copy only vertices and apply negated transform
    vertices_h = np.column_stack([vertices, np.ones(len(vertices))])
    vertices_neg = (vertices_h @ T_neg.T)[:, :3]

    # 7. AABB of negatively rotated vertices, then add margin
    bounds_neg = np.array(
        [vertices_neg.min(axis=0) - margin_mm,
         vertices_neg.max(axis=0) + margin_mm],
        dtype=float
    )

    # Convert bounds to extents + transform in the negated frame
    extents, obb_T_neg = trimesh.bounds.to_extents(bounds_neg)

    # 8. Apply original rotation to the box
    T_pos = trimesh.transformations.rotation_matrix(
        angle=angle_rad,
        direction=rotation_axis,
        point=centre
    )
    obb_T = T_pos @ obb_T_neg

    # Extract top/bottom planes from final rotated box
    R = obb_T[:3, :3]
    centre_rot = obb_T[:3, 3]

    superior_axis = R[:, 2]
    superior_axis = superior_axis / np.linalg.norm(superior_axis)

    decision_plane_dict = defaultdict(dict)
    if num_planes < 2:
        raise ValueError("num_planes must be at least 2.")
    superior_plane_spacing = extents[2] / (num_planes-1)
    for i in range(num_planes):
        origin_decision_plane = (
            centre_rot + superior_axis * superior_plane_spacing * (i - (num_planes-1)/2)
        )
        decision_plane_dict[i] = {
            "origin": origin_decision_plane,
            "normal": superior_axis,
            "transform": obb_T,
            "extents": extents,}

    return decision_plane_dict

def grid_on_plane(
    plane_origin: np.ndarray,
    obb_T: np.ndarray,
    extents: np.ndarray,
    n: int) -> np.ndarray:
    """
    Sample an NxN grid of 3-D points on a plane, staying inside the OBB face.

    Parameters
    ----------
    plane_origin : (3,)  point on the plane (e.g. OBB superior/inferior face centre)
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

def line_to_tube(
    p0: np.ndarray,
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

def angled_catheter_pairs(
    o_top: np.ndarray,
    o_bot: np.ndarray,
    normal: np.ndarray,
    obb_T: np.ndarray,
    extents: np.ndarray,
    grid_n: int,
    alt_max: float,
    alt_step: float,
    az_max: float,
    az_step: float,
) -> list:
    """
    ### Purpose
    - Generate angled catheter point pairs for a range of altitude and azimuthal angles.

    Each bottom grid point is paired with one or more top plane intersection points,
    produced by sweeping a ray from the bottom point across a discrete grid of
    (altitude, azimuth) angles. Pairs where the top intersection falls outside the
    OBB extents are discarded.

    Azimuth=0 is defined as the direction from the bottom point towards the
    bottom plane centre (radially inwards) and sweeps symmetrically from -az_max
    to +az_max. Altitude=0 is parallel to the normal.

    ### Inputs
    - o_top    : (3,) point on the top (superior) plane
    - o_bot    : (3,) point on the bottom (inferior) plane
    - normal   : (3,) unit normal pointing inferior → superior
    - obb_T    : (4,4) OBB transform (world ← OBB local frame)
    - extents  : (3,) OBB full extents [ex, ey, ez]
    - grid_n   : number of grid points per axis on each plane
    - alt_max  : maximum altitude angle away from normal (degrees); sweeps -alt_max to +alt_max
    - alt_step : altitude angle increment (degrees)
    - az_max   : half-width of azimuthal sweep (degrees); sweeps -az_max to +az_max
    - az_step  : azimuthal angle increment (degrees)

    ### Outputs
    - pairs : list of (top_pt, bot_pt) tuples, each a (3,) np.ndarray
    """
    # ── OBB axes and half-extents for bounds check ───────────────────────────
    obb_x  = obb_T[:3, 0]
    obb_y  = obb_T[:3, 1]
    half_x = extents[0] / 2.0
    half_y = extents[1] / 2.0

    # ── Angle grids ──────────────────────────────────────────────────────────
    alt_steps = np.arange(-alt_max, alt_max + 1e-9, alt_step)
    az_steps  = np.arange(-az_max, az_max + 1e-9, az_step)

    # ── Bottom grid points ───────────────────────────────────────────────────
    bot_pts = grid_on_plane(o_bot, obb_T, extents, grid_n)

    # ── Build pairs ──────────────────────────────────────────────────────────
    pairs = []

    for bot_pt in bot_pts:

        # ── Per-point local azimuth basis ────────────────────────────────────
        # u_axis: azimuth=0, pointing radially outward from bottom plane centre
        # v_axis: azimuth=+90°, completing the right-handed transverse frame
        radial  = bot_pt - o_bot
        radial -= np.dot(radial, normal) * normal   # project onto transverse plane
        norm_r  = np.linalg.norm(radial)

        if norm_r < 1e-9:
            # Centre point — fall back to a fixed reference direction
            arbitrary = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(arbitrary, normal)) > 0.9:
                arbitrary = np.array([0.0, 1.0, 0.0])
            u_axis = np.cross(normal, arbitrary)
            u_axis /= np.linalg.norm(u_axis)
        else:
            u_axis = -radial / norm_r # for pointing inwards towards the center of the bottom plane.

        v_axis = np.cross(normal, u_axis)
        v_axis /= np.linalg.norm(v_axis)

        for alt_deg in alt_steps:
            for az_deg in az_steps:

                # Collapse redundant azimuth samples at zero altitude
                if alt_deg == 0.0 and az_deg != 0.0:
                    continue

                alt_rad = np.radians(alt_deg)
                az_rad  = np.radians(az_deg)

                # Ray direction: normal tilted by alt toward the az direction
                az_dir  = np.cos(az_rad) * u_axis + np.sin(az_rad) * v_axis
                ray_dir = np.cos(alt_rad) * normal + np.sin(alt_rad) * az_dir
                ray_dir /= np.linalg.norm(ray_dir)

                # Intersect with top plane: dot(p - o_top, normal) = 0
                denom = np.dot(ray_dir, normal)
                if abs(denom) < 1e-12:    # ray parallel to plane
                    continue
                t = np.dot(o_top - bot_pt, normal) / denom
                if t <= 0:                # intersection behind bottom point
                    continue

                top_pt = bot_pt + t * ray_dir

                # Discard if top_pt is outside OBB extents on the top plane
                delta  = top_pt - o_top
                proj_x = abs(np.dot(delta, obb_x))
                proj_y = abs(np.dot(delta, obb_y))
                if proj_x <= half_x and proj_y <= half_y:
                    pairs.append((top_pt, bot_pt))

    return pairs

def build_line_connectors(
    mesh_dict:Dict[str, trimesh.Trimesh],
    grid_n:int ,
    danger_dist:float,
    perpendicular:bool,
    target_structures:List[str],
    config_angled_cathgen:Config_Angled_CathGen = None,
    **kwargs
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
    - perpendicular: bool := if True, lines run parallel to OBB Z axis (perpendicular to planes)

    ### Outputs:
    valid_lines: List[Tuple[np.ndarray, np.ndarray]] := list of (p0, p1) tuples
    """
    # # find the meshes that collide with or are close to the target structures. Only they are relevant
    # # for defining the bounding planes.
    meshes_4_planes = []
    target_meshes = [mesh_dict[name] for name in target_structures if name in mesh_dict]
    from trimesh.collision import CollisionManager
    collision_manager = CollisionManager()
    for name, mesh in mesh_dict.items():
        if name not in target_structures:
            collision_manager.add_object(name, mesh)    
    for target_mesh in target_meshes:
        names_colliding = collision_manager.in_collision_single(
            target_mesh,
            return_names=True)
        if names_colliding[0]:  # if there are any collisions, add the colliding meshes to the plane calculation
            meshes_4_planes += [mesh_dict[name] for name in names_colliding[1]]
    meshes_4_planes += target_meshes

    decision_plane_dict, normal, obb_T, extents = obb_planes(
        meshes_4_planes,
        margin_mm = kwargs.get("margin_mm", 10.0),
        rotation_angle_deg = kwargs.get("rotation_angle_deg", 0),
        num_planes = kwargs.get("num_planes", 2),
        )

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
        if config_angled_cathgen is None:
            raise ValueError("config_angled_cathgen must be provided when perpendicular=False")
        # Generate angled pairs by sweeping rays from each bottom point
        # across a grid of altitude and azimuth angles.
        pairs = angled_catheter_pairs(
            o_top=o_top,
            o_bot=o_bot,
            normal=normal,
            obb_T=obb_T,
            extents=extents,
            grid_n=grid_n,
            alt_max=config_angled_cathgen.alt_max,
            alt_step=config_angled_cathgen.alt_step,
            az_max=config_angled_cathgen.az_max,
            az_step=config_angled_cathgen.az_step,
        )        

    # ── 3. Filter colliding / too-close lines ───────────────────────────────
    oar_meshes = [mesh_dict[name] for name in mesh_dict if name not in target_structures]
    valid_lines = [
        (p0, p1) for p0, p1 in pairs
        if not line_is_invalid(p0, p1, oar_meshes, danger_dist)
    ]
    n_total = len(pairs)
    n_valid = len(valid_lines)
    print(f"Candidates: {n_total}  |  Valid (kept): {n_valid}  |  Discarded: {n_total - n_valid}")
    return valid_lines , o_top, o_bot, extents, obb_T

def gen_catheter_table_from_contours(
    mesh_dict: Dict[str, trimesh.Trimesh],
    target_structures: List[str],
    grid_n:int,
    danger_dist_mm:float = 3.0,
    perpendicular:bool = True,
    config_angled_cathgen:Config_Angled_CathGen = None,
    out_ply_dir:str | Path = None,
    catheter_radius:float = 1.0,
    ) -> CatheterTable:
    r"""
    ### Purpose
    - Given a dictionary of Trimesh objects, generate a CatheterTable by:
      1. Extracting the contour vertices as meshes.
      2. Running the `build_line_connectors` pipeline to get valid line segments.
      3. Converting valid line segments into Catheter and DwellPosition objects.
    
    ### Inputs
    - mesh_dict: Dict[str, trimesh.Trimesh] := dictionary of Trimesh objects (e.g. from TPS)
    - target_structures: List[str] := list of structure names to be irradiated
    - grid_n: int := number of candidate lines per plane axis (total candidates = grid_n^2)
    - danger_dist_mm: float := minimum allowed distance (mm) from any contour vertex
    - perpendicular: bool := if True, lines run parallel to OBB Z axis (perpendicular to planes)
    - out_ply_dir: str := if provided, directory to export STL files of meshes + lines
    - catheter_radius: float := visual radius of exported line tubes (mm)
    """
    valid_lines , o_top, o_bot, extents, obb_T = build_line_connectors(
        mesh_dict=mesh_dict,
        grid_n=grid_n,
        danger_dist=danger_dist_mm,
        perpendicular=perpendicular,
        target_structures=target_structures,
        config_angled_cathgen=config_angled_cathgen
    )

    # convert the valid lines into catheters in the dwell position.
    # Consider making the Candidate CatheterTable, which would have a cluster 
    # of trajectories for each insertion point (bottom point)
    valid_catheters = [
        Catheter(
            index=idx,
            digitization_points=line)
        for idx, line in enumerate(valid_lines)]
    catheter_table = CatheterTable(
        catheters_dict=valid_catheters,
    )
    return catheter_table

def decision_planes_to_ply(
    out_ply_dir: str | Path,
    decision_plane_dict: dict,
    ) -> None:
    out_ply_dir = Path(out_ply_dir)
    out_ply_dir.mkdir(parents=True, exist_ok=True)

    # Bounding planes as thin flat boxes
    for depth, data in decision_plane_dict.items():
        ex, ey, ez = data["extents"]
        box = trimesh.creation.box(extents=[ex, ey, 0.2])
        Tbox = data["transform"].copy()
        Tbox[:3, 3] = data["origin"]
        box.apply_transform(Tbox)
        path = out_ply_dir / f"plane_{depth}.ply"
        box.export(path)

    # # this code for visualization
    # if out_ply_dir is not None:
    #     out_ply_dir = Path(out_ply_dir)
    #     for i, line in enumerate(valid_lines):
    #         tube = line_to_tube(line[0], line[1], catheter_radius)
    #         if tube is not None:
    #             path = out_ply_dir / f"line_{i:03d}.ply"
    #             tube.export(path)
    #     for name, mesh in mesh_dict.items():
    #         path = out_ply_dir / f"{name}.ply"
    #         mesh.export(path)
    #     # Bounding planes as thin flat boxes
    #     for label, centre in [("plane_top", o_top), ("plane_bot", o_bot)]:
    #         ex, ey, ez = extents
    #         box = trimesh.creation.box(extents=[ex, ey, 0.2])
    #         Tbox          = obb_T.copy()
    #         Tbox[:3, 3]   = centre
    #         box.apply_transform(Tbox)
    #         path = os.path.join(out_ply_dir, f"{label}.ply")
    #         box.export(path)