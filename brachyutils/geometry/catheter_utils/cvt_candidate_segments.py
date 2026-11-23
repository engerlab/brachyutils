"""
cvt_candidate_segments.py

CVT-based generation of candidate catheter segments for the ClusterBox.
One catheter per insertion point, no branching: each catheter has
(num_decision_planes - 1) segments.

Plane 0 : ordinary CVT inside the valid region -> n insertion points.
Plane k+1: each catheter's landing point is a CVT generator confined to the
           rectangle reachable within the max x/y angles from its departure point.

Libraries: numpy, trimesh, scipy (cKDTree), shapely (>=2.0, already a trimesh dependency).
Reused from the existing code base: obb_planes, line_is_invalid, normalize,
Decision_Plane / Config_Catheter_Rotation.
"""
from typing import Dict, List, Tuple

import numpy as np
import shapely
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union
from trimesh.collision import CollisionManager

from brachyutils.geometry.catheter_utils.catheter_cluster_box_utils import (
    obb_planes,
    line_is_invalid,
    normalize,
)
from brachyutils.geometry.catheter_utils.config_cathgen import (
    Config_Catheter_Rotation as _ConfigRotation,
    Decision_Plane,
)


# ----------------------------------------------------------------------------
# Plane geometry helpers
# ----------------------------------------------------------------------------
def _plane_basis(plane: Decision_Plane):
    """Return (origin, x_axis, y_axis, to_2D) for a decision plane.
    to_2D maps world coordinates to the plane's local frame (z ~ 0 on the plane)."""
    T = np.asarray(plane.transform, dtype=float)
    R = T[:3, :3]
    origin = np.asarray(plane.origin, dtype=float)
    x_ax, y_ax = R[:, 0], R[:, 1]
    to_2D = np.eye(4)
    to_2D[:3, :3] = R.T
    to_2D[:3, 3] = -R.T @ origin
    return origin, x_ax, y_ax, to_2D


def _target_region_2d(target_meshes, plane, origin, x_ax, y_ax, to_2D):
    """Cross-section of the targets on the plane (2D shapely geometry).
    Falls back to the 2D projection (silhouette) if no target crosses the plane."""
    n = normalize(plane.normal)
    polys = []
    for mesh in target_meshes:
        sec = mesh.section(plane_normal=n, plane_origin=origin)
        if sec is None:
            continue
        planar, _ = sec.to_planar(to_2D=to_2D, check=False)
        polys += list(planar.polygons_full)
    if polys:
        return unary_union(polys).buffer(0), "section"

    tris = []
    for mesh in target_meshes:
        rel = np.asarray(mesh.vertices) - origin
        xy = np.column_stack([rel @ x_ax, rel @ y_ax])[mesh.faces]  # (F,3,2)
        for t in xy:
            poly = Polygon(t)
            if poly.area > 1e-9:
                tris.append(poly)
    return unary_union(tris).buffer(0), "projection"


def _oar_section_polys(oar_meshes, plane, origin, to_2D, margin):
    """Union of OAR cross-sections on the plane, grown by the safety margin."""
    n = normalize(plane.normal)
    polys = []
    for mesh in oar_meshes:
        sec = mesh.section(plane_normal=n, plane_origin=origin)
        if sec is None:
            continue
        planar, _ = sec.to_planar(to_2D=to_2D, check=False)
        polys += list(planar.polygons_full)
    return unary_union(polys).buffer(margin) if polys else None


def _valid_mask(xy, region, oar_poly, oar_tree, origin, x_ax, y_ax, margin):
    """True where a 2D point is inside the target region, outside the (buffered)
    OAR cross-sections and farther than `margin` from every OAR vertex (3D)."""
    ok = shapely.contains_xy(region, xy[:, 0], xy[:, 1])
    if oar_poly is not None:
        ok &= ~shapely.contains_xy(oar_poly, xy[:, 0], xy[:, 1])
    if oar_tree is not None and ok.any():
        p3 = origin + xy[ok, 0:1] * x_ax + xy[ok, 1:2] * y_ax
        d, _ = oar_tree.query(p3, k=1)
        idx = np.where(ok)[0]
        ok[idx[d < margin]] = False
    return ok


def _plane_samples(plane, target_meshes, oar_meshes, oar_tree, margin, n_ref):
    """Dense uniform samples (2D, plane-local) of the valid, OAR-free region."""
    origin, x_ax, y_ax, to_2D = _plane_basis(plane)
    region, kind = _target_region_2d(target_meshes, plane, origin, x_ax, y_ax, to_2D)
    if region.is_empty:
        raise ValueError(f"Plane {plane.depth}: empty target region.")
    oar_poly = _oar_section_polys(oar_meshes, plane, origin, to_2D, margin)

    minx, miny, maxx, maxy = region.bounds
    h = max(np.sqrt(region.area / max(20 * n_ref, 5000)), 0.05)
    gx, gy = np.meshgrid(np.arange(minx, maxx, h), np.arange(miny, maxy, h))
    cand = np.column_stack([gx.ravel(), gy.ravel()])
    ok = _valid_mask(cand, region, oar_poly, oar_tree, origin, x_ax, y_ax, margin)
    if not ok.any():
        raise ValueError(f"Plane {plane.depth}: no OAR-free area (region from {kind}).")
    return dict(origin=origin, x_ax=x_ax, y_ax=y_ax,
                samples=cand[ok], h=h, kind=kind)


# ----------------------------------------------------------------------------
# CVT (Lloyd) routines
# ----------------------------------------------------------------------------
def _cvt_lloyd(samples, n, init=None, iters=60, tol=1e-3, rng=None):
    """Unconstrained Lloyd's algorithm on dense uniform samples of the region."""
    rng = rng or np.random.default_rng(0)
    n = min(n, len(samples))
    gens = np.empty((0, 2)) if init is None else np.asarray(init)[:n]
    if len(gens) < n:
        extra = samples[rng.choice(len(samples), n - len(gens), replace=False)]
        gens = np.vstack([gens, extra])
    for _ in range(iters):
        _, lab = cKDTree(gens).query(samples)
        cnt = np.bincount(lab, minlength=n)
        new = np.column_stack([np.bincount(lab, samples[:, k], minlength=n)
                               for k in range(2)])
        empty = cnt == 0
        new[~empty] /= cnt[~empty, None]
        if empty.any():
            new[empty] = samples[rng.choice(len(samples), empty.sum())]
        shift = np.linalg.norm(new - gens, axis=1).max()
        gens = new
        if shift < tol:
            break
    return gens, samples


def _constrained_lloyd(samples, start, lo, hi, iters=60, tol=1e-3):
    """Lloyd's algorithm where generator i is confined to the box [lo[i], hi[i]]."""
    gens = np.clip(start, lo, hi)
    n = len(gens)
    for _ in range(iters):
        _, lab = cKDTree(gens).query(samples)
        cnt = np.bincount(lab, minlength=n)
        cen = np.column_stack([np.bincount(lab, samples[:, k], minlength=n)
                               for k in range(2)])
        has = cnt > 0
        cen[has] /= cnt[has, None]
        cen[~has] = gens[~has]
        new = np.clip(cen, lo, hi)
        shift = np.linalg.norm(new - gens, axis=1).max()
        gens = new
        if shift < tol:
            break
    return gens


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def generate_candidate_segments_cvt(
    mesh_dict: Dict[str, trimesh.Trimesh],
    insertion_point_spacing_mm: float,
    oar_danger_dist_mm: float,
    target_structure_names: List[str],
    num_physical_catheters: int,
    Config_Catheter_Rotation: _ConfigRotation = None,
    catheter_radius_mm: float = 1,
    cvt_iters: int = 60,
    max_snap_candidates: int = 50,
    drop_incomplete_chains: bool = False,
    seed: int = 0,
    **kwargs,
) -> Tuple[List, Dict]:
    """
    ### Purpose:
    - CVT version of generate_candidate_segments. One catheter per insertion
      point, no branching: n catheters, each with (num_planes - 1) segments.

    ### Inputs:
    - Same as generate_candidate_segments, plus:
    - insertion_point_spacing_mm: also used as the minimum spacing between points on a plane.
    - num_physical_catheters: requested number of catheters (reduced if the region is too small).
    - cvt_iters: Lloyd iterations.
    - max_snap_candidates: alternative landing points tried when the CVT point is invalid.
    - drop_incomplete_chains: drop catheters that cannot reach the last plane.
    - kwargs: bb_margin_mm, bb_rotation_angle_deg, bb_num_planes.

    ### Outputs:
    - (digitization_pairs, decision_plane_dict); each plane's segment_lines is filled.
    """
    cfg = _ConfigRotation() if Config_Catheter_Rotation is None else Config_Catheter_Rotation
    if isinstance(cfg, dict):
        raise NotImplementedError("Per-insertion-point angle configs are not supported in CVT mode.")
    rng = np.random.default_rng(seed)

    # --- mesh selection and planes (same as the grid version) ---
    target_meshes = [mesh_dict[k] for k in target_structure_names if k in mesh_dict]
    cm = CollisionManager()
    for name, mesh in mesh_dict.items():
        if name not in target_structure_names:
            cm.add_object(name, mesh)
    meshes_4_planes = list(target_meshes)
    for tm in target_meshes:
        hit, names = cm.in_collision_single(tm, return_names=True)
        if hit:
            meshes_4_planes += [mesh_dict[n] for n in names]
    plane_dict = obb_planes(
        meshes_4_planes,
        margin_mm=kwargs.get("bb_margin_mm", 10.0),
        rotation_angle_deg=kwargs.get("bb_rotation_angle_deg", 0),
        num_planes=kwargs.get("bb_num_planes", 2),
    )
    planes = list(plane_dict.values())

    oar_meshes = [m for k, m in mesh_dict.items() if k not in target_structure_names]
    margin = max(oar_danger_dist_mm - catheter_radius_mm, 0.0)
    oar_tree = (cKDTree(np.vstack([m.vertices for m in oar_meshes]))
                if oar_meshes else None)

    # --- plane 0: ordinary CVT gives the insertion points ---
    ctx0 = _plane_samples(planes[0], target_meshes, oar_meshes, oar_tree,
                          margin, num_physical_catheters)
    area0 = len(ctx0["samples"]) * ctx0["h"] ** 2
    n_max = max(1, int(area0 / (0.866 * insertion_point_spacing_mm ** 2)))  # hex packing
    n = min(num_physical_catheters, n_max, len(ctx0["samples"]))
    g0, _ = _cvt_lloyd(ctx0["samples"], n, iters=cvt_iters, rng=rng)
    _, nn = cKDTree(ctx0["samples"]).query(g0)       # snap onto valid samples
    g0 = ctx0["samples"][nn]
    pts0 = ctx0["origin"] + g0[:, 0:1] * ctx0["x_ax"] + g0[:, 1:2] * ctx0["y_ax"]
    print(f"Plane 0: {n} CVT insertion points (region from {ctx0['kind']})")
    chains = [[p] for p in pts0]
    alive = np.ones(n, dtype=bool)

    # --- deeper planes: one constrained-CVT landing point per catheter ---
    for k in range(len(planes) - 1):
        pl, nxt = planes[k], planes[k + 1]
        ctx = _plane_samples(nxt, target_meshes, oar_meshes, oar_tree,
                             margin, num_physical_catheters)
        o, xa, ya, S = ctx["origin"], ctx["x_ax"], ctx["y_ax"], ctx["samples"]

        dz = abs(np.dot(np.asarray(nxt.origin) - np.asarray(pl.origin),
                        normalize(pl.normal)))
        hw = np.array([dz * np.tan(np.radians(cfg.x_angle_max)),   # along plane x
                       dz * np.tan(np.radians(cfg.y_angle_max))])  # along plane y

        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        dep = np.array([chains[i][-1] for i in idx])
        c2d = np.column_stack([(dep - o) @ xa, (dep - o) @ ya])
        lo, hi = c2d - hw, c2d + hw
        gens = _constrained_lloyd(S, c2d.copy(), lo, hi, iters=cvt_iters)

        taken = []
        for r, i in enumerate(idx):
            in_box = np.all((S >= lo[r] - ctx["h"]) & (S <= hi[r] + ctx["h"]), axis=1)
            cand = S[in_box]
            if len(cand) == 0:
                alive[i] = False
                continue
            order = np.argsort(np.linalg.norm(cand - gens[r], axis=1))
            chosen = None
            for c in cand[order[:max_snap_candidates]]:
                p1 = o + c[0] * xa + c[1] * ya
                if taken and np.min(np.linalg.norm(np.array(taken) - p1, axis=1)) \
                        < insertion_point_spacing_mm:
                    continue
                # depth=0 skips the parent check; chaining holds by construction
                if not line_is_invalid(chains[i][-1], p1, oar_meshes, margin, 0, []):
                    chosen = p1
                    break
            if chosen is None:
                alive[i] = False
            else:
                chains[i].append(chosen)
                taken.append(chosen)
        print(f"Plane {k + 1}: {alive.sum()}/{n} catheters still alive")

    # --- assemble per-plane segment_lines in chain order ---
    n_planes = len(planes)
    keep = ([c for c in chains if len(c) == n_planes] if drop_incomplete_chains
            else [c for c in chains if len(c) > 1])
    digitization_pairs = []
    for k, pl in enumerate(planes[:-1]):
        lines = [(c[k], c[k + 1]) for c in keep if len(c) > k + 1]
        pl.segment_lines = lines
        digitization_pairs += lines
    print(f"Catheters: requested {n} | kept {len(keep)} | segments {len(digitization_pairs)}")
    return digitization_pairs, plane_dict
