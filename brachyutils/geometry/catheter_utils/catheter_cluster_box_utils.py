from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import trimesh
import trimesh.creation
import trimesh.transformations as tf
from trimesh.ray.ray_triangle import RayMeshIntersector
from scipy.spatial import cKDTree
from pathlib import Path
from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable, Catheter
from brachyutils.geometry.catheter_utils.config_cathgen import Config_Catheter_Rotation, Decision_Plane
from math import radians
from scipy.spatial import cKDTree

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
    meshes: list | dict,
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
    if isinstance(meshes, dict):
        meshes = list(meshes.values)
    vertices = np.vstack([np.asarray(mesh.vertices) for mesh in meshes])
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
        decision_plane_dict[i] = Decision_Plane(
            depth= i,
            origin= origin_decision_plane,
            normal= superior_axis,
            transform= obb_T,
            extents= extents
            )

    return decision_plane_dict

def grid_on_plane(
    plane_origin: np.ndarray,
    obb_T: np.ndarray,
    extents: np.ndarray,
    insertion_point_spacing_mm: float) -> np.ndarray:
    """
    ### Purpose:
    - Sample an NxN grid of 3-D points on a plane, staying inside the OBB face.

    ### Inputs
    - plane_origin : (3,)  point on the plane (e.g. OBB superior/inferior face centre)
    - obb_T        : (4,4) OBB transform (provides X/Y in-plane axes)
    - extents      : (3,)  OBB extents [ex, ey, ez]
    - insertion_point_spacing_mm : float := spacing between adjacent grid points (mm)

    ### Returns
    - pts : (N*N, 3)
    """
    R    = obb_T[:3, :3]
    x_ax = R[:, 0]
    y_ax = R[:, 1]
    ex, ey = extents[0], extents[1]
    n_x = max(2, int(np.floor(ex / insertion_point_spacing_mm)))
    n_y = max(2, int(np.floor(ey / insertion_point_spacing_mm)))
    # Inset slightly from edges
    us = np.linspace(-ex/2 + ex/(2*n_x), ex/2 - ex/(2*n_x), n_x)
    vs = np.linspace(-ey/2 + ey/(2*n_y), ey/2 - ey/(2*n_y), n_y)
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


def line_is_invalid(
    p0: np.ndarray,
    p1: np.ndarray,
    meshes: list,
    danger_dist: float,
    plane_depth: int,
    old_valid_lines:List[tuple],
    n_samples: int = PROX_SAMPLES,) -> bool:
    """
    Return True if line p0→p1:
      - intersects any mesh face, OR
      - passes within `danger_dist` of any mesh vertex.

    Uses BVH-accelerated ray casting (trimesh) + KD-tree proximity (scipy).
    """
    # plane of depth >=1 must have lines with a parent segment
    if plane_depth >=1 :
        parent_found = False
        for old_line in old_valid_lines:
            if np.all(p0 == old_line[1]):
                parent_found = True
                break
        if not parent_found:
            return True

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

def generate_candidate_segments(
    mesh_dict:Dict[str, trimesh.Trimesh],
    insertion_point_spacing_mm:float,
    oar_danger_dist_mm:float,
    target_structure_names:List[str],
    Config_Catheter_Rotation:Config_Catheter_Rotation = None,
    catheter_radius_mm:float=1,
    **kwargs
    ) -> Tuple[List, Dict]:
    """
    ### Purpose:
    - Given a set of 3D meshes, automatically generate a set of
    straight line connectors between two bounding planes,
    while avoiding collisions and close proximity to the meshes.
    Export everything as STL files for visualization.
    Full pipeline: OBB planes → grid → filter → export STL.

    ### Inputs:
    - meshes: List[trimesh.Trimesh] := list of trimesh.Trimesh
    - insertion_point_spacing_mm: float := spacing for the insertion grid
    - oar_danger_dist_mm: float := distance threshold for danger zones. It is from the center
    of the catheter to the surface of the OAR mesh. Therefore, the distance from surface of catheter
    to the surface of the OAR is oar_danger_dist_mm - catheter_radius_mm.
    - target_structure_names: List[str] := list of target structure names

    ### Outputs:
    valid_lines: List[Tuple[np.ndarray, np.ndarray]] := list of (p0, p1) tuples
    """
    # # find the meshes that collide with or are close to the target structures. Only they are relevant
    # # for defining the bounding planes.
    meshes_4_planes = []
    target_meshes = [mesh_dict[name] for name in target_structure_names if name in mesh_dict]
    from trimesh.collision import CollisionManager
    collision_manager = CollisionManager()
    for name, mesh in mesh_dict.items():
        if name not in target_structure_names:
            collision_manager.add_object(name, mesh)    
    for target_mesh in target_meshes:
        names_colliding = collision_manager.in_collision_single(
            target_mesh,
            return_names=True)
        if names_colliding[0]:  # if there are any collisions, add the colliding meshes to the plane calculation
            meshes_4_planes += [mesh_dict[name] for name in names_colliding[1]]
    meshes_4_planes += target_meshes

    # convert the meshes into a dictionary    
    decision_plane_dict = obb_planes(
        meshes_4_planes,
        margin_mm = kwargs.get("bb_margin_mm", 10.0),
        rotation_angle_deg = kwargs.get("bb_rotation_angle_deg", 0),
        num_planes = kwargs.get("bb_num_planes", 2),
        )

    # # between two deicion planes, define the pairs of points
    # # that form digitization points for the catheter segments. 
    inferior_plane_grid = grid_on_plane(
        plane_origin = decision_plane_dict[0].origin,
        obb_T = decision_plane_dict[0].transform,
        extents = decision_plane_dict[0].extents,
        insertion_point_spacing_mm = insertion_point_spacing_mm,
    )
    digitization_pairs = []
    n_total = 0
    oar_meshes = [mesh_dict[name] for name in mesh_dict if name not in target_structure_names]
    for i, plane in enumerate(decision_plane_dict.values()):
        if i == len(decision_plane_dict)-1:
            break
        plane_digi_points = get_segment_lines(
            departure_plane = plane,
            departure_plane_grid = inferior_plane_grid,
            landing_plane = decision_plane_dict[i+1],
            Config_Catheter_Rotation = Config_Catheter_Rotation,
        )
        # digitization_pairs += plane_digi_points
        # the superior plane points become the inferior plane points for the next iteration
        inferior_plane_grid = [pf for pi, pf in plane_digi_points]
        n_total += len(plane_digi_points)
        # 3. Filter colliding / too-close lines 
        new_valid_lines = [
            (p0, p1) for p0, p1 in plane_digi_points
            if not line_is_invalid(
                p0, p1, oar_meshes, (oar_danger_dist_mm-catheter_radius_mm),
                plane.depth, digitization_pairs
                )
        ]
        plane.segment_lines = new_valid_lines
        digitization_pairs += new_valid_lines

    n_valid = len(digitization_pairs)
    print(f"Candidates: {n_total}  |  Valid (kept): {n_valid}  |  Discarded: {n_total - n_valid}")
    return digitization_pairs, decision_plane_dict

def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero-length vector")
    return v / n

def rotate_vector(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Rotate vector v around given axis (unit or non-unit) by angle_rad (radians)
    using Rodrigues' rotation formula.
    """
    v = np.asarray(v, dtype=float)
    k = normalize(axis)  # ensure unit axis
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    term1 = v * cos_theta
    term2 = np.cross(k, v) * sin_theta
    term3 = k * np.dot(k, v) * (1.0 - cos_theta)
    return term1 + term2 + term3

def intersect_ray_with_plane(ray_origin: np.ndarray,
                             ray_dir: np.ndarray,
                             plane_origin: np.ndarray,
                             plane_normal: np.ndarray,
                             eps: float = 1e-6):
    """
    Return intersection point of ray and plane, or None if no valid intersection.
    Ray:  R(t) = ray_origin + t * ray_dir, t >= 0
    Plane: (p - plane_origin) · plane_normal = 0
    """
    ray_origin = np.asarray(ray_origin, dtype=float)
    ray_dir = normalize(ray_dir)
    plane_origin = np.asarray(plane_origin, dtype=float)
    plane_normal = normalize(plane_normal)

    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < eps:
        # Ray is parallel (or almost parallel) to plane
        return None

    t = -np.dot(ray_origin - plane_origin, plane_normal) / denom
    if t < 0:
        # Intersection behind origin along ray_dir; discard
        return None

    return ray_origin + t * ray_dir

def point_world_to_local(point_world: np.ndarray,
                         plane_origin: np.ndarray,
                         basis_x: np.ndarray,
                         basis_y: np.ndarray,
                         basis_z: np.ndarray) -> np.ndarray:
    """
    Convert world-space point to plane-local coordinates given origin and basis vectors.
    Assumes basis_x, basis_y, basis_z are orthonormal.
    """
    point_world = np.asarray(point_world, dtype=float)
    plane_origin = np.asarray(plane_origin, dtype=float)

    # Vector from origin to point
    v = point_world - plane_origin

    # Project onto basis vectors to get local coords
    local_x = np.dot(v, basis_x)
    local_y = np.dot(v, basis_y)
    local_z = np.dot(v, basis_z)

    return np.array([local_x, local_y, local_z], dtype=float)

def point_inside_extents(local_point: np.ndarray,
                         extents: np.ndarray,
                         use_xy_only: bool = True) -> bool:
    """
    Check if local_point lies inside extents.
    If use_xy_only is True, ignore z and only check x,y.
    extents is assumed to be [extent_x, extent_y, extent_z].
    """
    local_point = np.asarray(local_point, dtype=float)
    extents = np.asarray(extents, dtype=float)

    if use_xy_only:
        return (abs(local_point[0]) <= extents[0] and
                abs(local_point[1]) <= extents[1])
    else:
        return np.all(np.abs(local_point) <= extents)

def get_segment_lines(
    departure_plane: dict,
    departure_plane_grid: List[np.ndarray],
    landing_plane: dict,
    Config_Catheter_Rotation
    ) -> List[tuple[np.ndarray, np.ndarray]]:
    r"""
    ### Purpose:
    - Given an inferior plane and a superior plane, generate the digitization pairs
    connecting the two planes. The inferior plane points are given by the grid,
    and the superior plane points are generated by sweeping rays from each inferior
    point across a grid of right-left angle (x-axis) and anterior-posterior (y-axis) angle.

    ### Inputs:
    - inferior_plane: dict := dictionary containing the inferior plane information
    - inferior_plane_grid: List[np.ndarray] := list of points on the inferior plane
    - superior_plane: dict := dictionary containing the superior plane information
    - Config_Catheter_Rotation: Config_Catheter_Rotation := configuration for the angled catheter generation

    ### Outputs:
    - digitization_pairs: List[tuple[np.ndarray, np.ndarray]] := list of
      (superior_point, inferior_point) tuples
    """
    n = normalize(departure_plane.normal)
    origin0 = np.asarray(departure_plane.origin, dtype=float)
    origin1 = np.asarray(landing_plane.origin, dtype=float)

    transform0 = np.asarray(departure_plane.transform, dtype=float)
    transform1 = np.asarray(landing_plane.transform, dtype=float)

    # Basis vectors from transform (assuming same for both planes here)
    basis_x = normalize(transform0[0, :3])  # plane x-axis in world
    basis_y = normalize(transform0[1, :3])  # plane y-axis in world
    basis_z = normalize(transform0[2, :3])  # plane z-axis; could be normal-ish

    # For ray directions, use the provided normal n as "central" direction
    central_dir = n

    extent_landing = np.asarray(landing_plane.extents, dtype=float)

    landing_points = []  # list of (theta_x, theta_y, landing_point_world)
    # Build angle ranges
    if Config_Catheter_Rotation.x_angle_step == 0:
        y_angles = np.array([Config_Catheter_Rotation.x_angle_max])
    else:
        y_angles = np.arange(-Config_Catheter_Rotation.x_angle_max,
                            Config_Catheter_Rotation.x_angle_max + 1e-6,
                            Config_Catheter_Rotation.x_angle_step)
    if Config_Catheter_Rotation.y_angle_step == 0:
        x_angles = np.array([Config_Catheter_Rotation.y_angle_max])
    else:
        x_angles = np.arange(-Config_Catheter_Rotation.y_angle_max,
                         Config_Catheter_Rotation.y_angle_max + 1e-6,
                         Config_Catheter_Rotation.y_angle_step)
    for departure_point in departure_plane_grid:
        for ax_deg in x_angles:
            for ay_deg in y_angles:
                ax_rad = radians(ax_deg)
                ay_rad = radians(ay_deg)

                # 1) rotate central_dir around plane x-axis by ax_rad
                dir_after_x = rotate_vector(central_dir, basis_x, ax_rad)

                # 2) rotate result around plane y-axis by ay_rad
                final_dir = rotate_vector(dir_after_x, basis_y, ay_rad)

                # Intersect this ray with landing plane
                intersection = intersect_ray_with_plane(
                    ray_origin=departure_point,
                    ray_dir=final_dir,
                    plane_origin=origin1,
                    plane_normal=n
                )

                if intersection is None:
                    continue  # no valid intersection
                
                # # If intersection falls out of the landing plane also continue
                in_landing_plane = np.array([False, False, False])
                for ax, pos in enumerate(intersection):
                    if (origin1[ax] - extent_landing[ax]/2
                        <= pos
                        <= origin1[ax] + extent_landing[ax]/2):
                        in_landing_plane[ax] = True
                        
                if not in_landing_plane.all():
                    continue
                # At this point, intersection is valid. Convert to Python tuples.
                landing_points.append((departure_point, intersection))

    return landing_points

def gen_catheter_table_from_contours(
    mesh_dict: Dict[str, trimesh.Trimesh],
    target_structures: List[str],
    oar_danger_dist_mm_mm:float = 3.0,
    insertion_point_spacing_mm:float = 5.0,
    Config_Catheter_Rotation:Config_Catheter_Rotation = None,
    ) -> CatheterTable:
    r"""
    ### Purpose
    - Given a dictionary of Trimesh objects, generate a CatheterTable by:
      1. Extracting the contour vertices as meshes.
      2. Running the `generate_candidate_segments` pipeline to get valid line segments.
      3. Converting valid line segments into Catheter and DwellPosition objects.
    
    ### Inputs
    - mesh_dict: Dict[str, trimesh.Trimesh] := dictionary of Trimesh objects (e.g. from TPS)
    - target_structures: List[str] := list of structure names to be irradiated
    - insertion_point_spacing_mm: float := spacing for the insertion grid
    - oar_danger_dist_mm_mm: float := minimum allowed distance (mm) from any OAR vertex
    - Config_Catheter_Rotation: Config_Catheter_Rotation: The config for angled catheter insertion.
    """
    valid_lines , o_top, o_bot, extents, obb_T = generate_candidate_segments(
        mesh_dict=mesh_dict,
        insertion_point_spacing_mm=insertion_point_spacing_mm,
        oar_danger_dist_mm_mm=oar_danger_dist_mm_mm,
        target_structures=target_structures,
        Config_Catheter_Rotation=Config_Catheter_Rotation
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
    for depth, plane in decision_plane_dict.items():
        ex, ey, ez = plane.extents
        box = trimesh.creation.box(extents=[ex, ey, 0.2])
        Tbox = plane.transform.copy()
        Tbox[:3, 3] = plane.origin
        box.apply_transform(Tbox)
        path = out_ply_dir / f"plane_{plane.depth}.ply"
        box.export(path)

def segment_lines_to_ply(
    out_ply_dir:str | Path,
    point_pairs: List[tuple],
    catheter_radius:float=1,
    ):
    out_ply_dir = Path(out_ply_dir)
    out_ply_dir.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(point_pairs):
        tube = line_to_tube(np.array(line[0]), np.array(line[1]), catheter_radius)
        if tube is not None:
            path = out_ply_dir / f"line_{i:03d}.ply"
            tube.export(path)

class TupleKeyDict(dict):
    def __init__(self, data, _dim=0):
        # data is a dict with tuple(int,int,int) keys
        self.data = data
        self._dim = _dim  # which dimension the next [...] applies to

    def __getitem__(self, item):
        # Filter keys where key[self._dim] matches item (slice or int)
        all_keys = sorted(self.data.keys())  # sort for consistent slicing
        dim_values = sorted(set(k[self._dim] for k in all_keys))

        # Apply the slice/index to the unique values of this dimension
        if isinstance(item, slice):
            matching_values = set(dim_values[item])
            filtered = {k: v for k, v in self.data.items()
                        if k[self._dim] in matching_values}
            return TupleKeyDict(filtered, self._dim + 1)

        elif isinstance(item, int):
            target = dim_values[item]  # support negative indexing too
            filtered = {k: v for k, v in self.data.items()
                        if k[self._dim] == target}
            # Last dimension — return values directly if only one result
            if self._dim == 2:
                return list(filtered.values())[0] if len(filtered) == 1 else filtered
            return TupleKeyDict(filtered, self._dim + 1)

    def __repr__(self):
        return f"TupleKeyDict({self.data})"
    
    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()


def point_seg_distance(p, a, b):
    """Distance from point(s) p to segment(s) a-b. All (M,3) or (3,)."""
    p, a, b = map(np.atleast_2d, (p, a, b))
    ab = b - a
    t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.einsum('ij,ij->i', ab, ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)

def seg_seg_distance_vec(a0, a1, b0, b1):
    """Vectorized segment-segment distance. Inputs (N,3). Returns dist, ptA, ptB."""
    a0, a1, b0, b1 = map(np.atleast_2d, (a0, a1, b0, b1))
    N = max(map(len, (a0, a1, b0, b1)))
    a0, a1, b0, b1 = [np.broadcast_to(x, (N, 3)) for x in (a0, a1, b0, b1)]

    A, B = a1 - a0, b1 - b0
    magA, magB = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    _A, _B = A / magA[:, None], B / magB[:, None]
    cross = np.cross(_A, _B)
    denom = np.einsum('ij,ij->i', cross, cross)

    parallel = denom < 1e-12
    dist = np.zeros(N)
    ptA = np.zeros((N, 3))
    ptB = np.zeros((N, 3))

    # Non-parallel
    mask = ~parallel
    if mask.any():
        t = b0[mask] - a0[mask]
        detA = np.linalg.det(np.stack([t, _B[mask], cross[mask]], axis=1))
        detB = np.linalg.det(np.stack([t, _A[mask], cross[mask]], axis=1))
        t0 = np.clip(detA / denom[mask], 0, magA[mask])
        t1 = np.clip(detB / denom[mask], 0, magB[mask])
        ptA[mask] = a0[mask] + _A[mask] * t0[:, None]
        ptB[mask] = b0[mask] + _B[mask] * t1[:, None]
        dist[mask] = np.linalg.norm(ptA[mask] - ptB[mask], axis=1)

    # Parallel: min of 4 endpoint-segment distances
    if parallel.any():
        idx = np.where(parallel)[0]
        d = np.column_stack([
            point_seg_distance(a0[idx], b0[idx], b1[idx]),
            point_seg_distance(a1[idx], b0[idx], b1[idx]),
            point_seg_distance(b0[idx], a0[idx], a1[idx]),
            point_seg_distance(b1[idx], a0[idx], a1[idx]),
        ])
        mi = np.argmin(d, axis=1)
        dist[idx] = d[np.arange(len(idx)), mi]
        for k, i in enumerate(idx):
            if mi[k] == 0: ptA[i], ptB[i] = _closest(a0[i], b0[i], b1[i]), b0[i]
            elif mi[k] == 1: ptA[i], ptB[i] = _closest(a1[i], b0[i], b1[i]), b0[i]
            elif mi[k] == 2: ptB[i], ptA[i] = _closest(b0[i], a0[i], a1[i]), a0[i]
            else: ptB[i], ptA[i] = _closest(b1[i], a0[i], a1[i]), a0[i]

    return dist, ptA, ptB

def _closest(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return a + t * ab

def find_colliding_pairs(starts, ends, danger_mm, chunk_size=5000):
    """
    Find segment pairs closer than danger_mm, excluding:
    - pairs with same start point (starts[i] == starts[j])
    - pairs where start of one == end of other (starts[i] == ends[j] or starts[j] == ends[i])
    
    Returns: pairs (K,2), dists (K,), pts_a (K,3), pts_b (K,3)
    """
    N = len(starts)
    mids = (starts + ends) * 0.5
    half_lens = np.linalg.norm(ends - starts, axis=1) * 0.5

    tree = cKDTree(mids)
    radius = danger_mm + half_lens.max()
    cand = tree.query_pairs(r=radius, output_type='ndarray')
    if len(cand) == 0:
        return np.empty((0,2), int), np.empty(0), np.empty((0,3)), np.empty((0,3))

    i, j = cand[:, 0], cand[:, 1]

    # ---- Exclusion filters (vectorized, exact float match) ----
    same_start       = np.all(starts[i] == starts[j], axis=1)
    start_i_eq_end_j = np.all(starts[i] == ends[j], axis=1)
    start_j_eq_end_i = np.all(starts[j] == ends[i], axis=1)

    exclude = same_start | start_i_eq_end_j | start_j_eq_end_i
    i, j = i[~exclude], j[~exclude]
    
    if len(i) == 0:
        return np.empty((0,2), int), np.empty(0), np.empty((0,3)), np.empty((0,3))

    # AABB filter
    mi, ma = np.minimum(starts[i], ends[i]), np.maximum(starts[i], ends[i])
    mj, mj2 = np.minimum(starts[j], ends[j]), np.maximum(starts[j], ends[j])
    keep = (mi <= mj2 + danger_mm).all(1) & (mj <= ma + danger_mm).all(1)
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return np.empty((0,2), int), np.empty(0), np.empty((0,3)), np.empty((0,3))

    # Chunked exact evaluation
    out_i, out_j, out_d, out_pa, out_pb = [], [], [], [], []
    for s in range(0, len(i), chunk_size):
        sl = slice(s, s + chunk_size)
        d, pa, pb = seg_seg_distance_vec(starts[i[sl]], ends[i[sl]], starts[j[sl]], ends[j[sl]])
        m = d < danger_mm
        if m.any():
            out_i.append(i[sl][m]); out_j.append(j[sl][m])
            out_d.append(d[m]); out_pa.append(pa[m]); out_pb.append(pb[m])
    
    if not out_i:
        return np.empty((0,2), int), np.empty(0), np.empty((0,3)), np.empty((0,3))
    
    return (np.column_stack([np.concatenate(out_i), np.concatenate(out_j)]),
            np.concatenate(out_d), np.concatenate(out_pa), np.concatenate(out_pb))

def get_angles(
    line: np.typing.ArrayLike,
    plane_normal: np.typing.ArrayLike
):
    r"""
    ### Purpose:
    - To get the rotation angles of a line based on the plane normal.

    ### Inputs:
    - `line` := 2 points each with a coordinate in 3D
    - `plane_normal` := a 3D vector.

    ### Outputs:
    - `angles` := A pair of angles in degrees, each in range [-180, 180].
      The first is the rotation around the y axis (x_angle),
      and the second is the rotation around the x axis (y_angle).
    """
    line = np.asarray(line, dtype=float)
    n = np.asarray(plane_normal, dtype=float)
    n = n / np.linalg.norm(n)

    d = line[1] - line[0]
    d = d / np.linalg.norm(d)

    # pick a reference vector not parallel to n, to build a stable in-plane frame
    ref = np.array([0.0, 0.0, 1.0])
    if np.allclose(np.abs(np.dot(n, ref)), 1.0):
        ref = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(ref, n)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(n, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    dx = np.dot(d, x_axis)
    dy = np.dot(d, y_axis)
    dz = np.dot(d, n)

    x_angle = np.degrees(np.arctan2(dx, dz))  # rotation around local y axis, range [-180, 180]
    y_angle = np.degrees(np.arctan2(dy, dz))  # rotation around local x axis, range [-180, 180]

    return np.array([x_angle, y_angle])