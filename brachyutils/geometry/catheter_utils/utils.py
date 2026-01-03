import os 
from typing import List, Dict
from itertools import combinations
import json
import numpy as np
import SimpleITK as sitk
from scipy.optimize import curve_fit, minimize_scalar, linear_sum_assignment, root_scalar
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix, ConvexHull
from scipy.interpolate import splprep, splev
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, HDBSCAN

def is_headless():
    return os.environ.get('DISPLAY') is None

########################## POINTS HELPER FUNCTIONS ##########################


def distance(x1, x2):
    return np.linalg.norm(np.array(x1) - np.array(x2))


def min_cost_two_list(list_points1, list_points2, return_points:bool=False):
    """
    Compute the minimum cost to go from every dwell positions to every dwell positions
    This is a linear sum assignment problem.
    """
    if len(list_points1) == 0 or len(list_points2) == 0:
        return np.inf if return_points else (np.inf, [])
    # Compute the distance matrix between the two lists of points
    distances = cdist(list_points1, list_points2, metric="euclidean")
    # Solve the linear sum assignment problem
    row_ind, col_ind = linear_sum_assignment(distances)
    min_cost = distances[row_ind, col_ind].sum()/max(len(list_points1), len(list_points2))
    if return_points:
        pairs = np.column_stack((row_ind, col_ind))
        pairs = filter_pairs(pairs)
        return min_cost, [[list_points1[pair[0]], list_points2[pair[1]]] for pair in pairs]
    return min_cost

def min_dist_two_list(list_points1, list_points2, return_points=False, fast_with_arrays=True):
    """
    Compute the minimum distance between two lists of points. 
    This is the minimum distance between any two points from the two lists.
    Args:
        list_points1 (List[List[float]]): First list of points.
        list_points2 (List[List[float]]): Second list of points.
        return_points (bool): If True, return the points that are at the minimum distance.
        fast_with_arrays (bool): If True, use numpy arrays for faster computation.
    Returns:
        float: The minimum distance between the two lists of points.
    """
    if fast_with_arrays:
        # Compute the distance matrix between the two lists of points
        distances = cdist(list_points1,list_points2, metric="euclidean")
        min_distance = np.min(distances)
        if return_points:
            args = np.where(distances == min_distance)
            pairs = np.column_stack(args)
            pairs = filter_pairs(pairs)
            return min_distance, [[list_points1[pair[0]], list_points2[pair[1]]] for pair in pairs]
        return min_distance 
    else:
        # Compute distance point by point
        min_dist = np.inf
        for point1 in list_points1:
            for point2 in list_points2:
                dist = distance(point1, point2)
                if dist < min_dist:
                    min_dist = dist
                    points = [point1, point2]
        if return_points:
            return min_dist, points
        return min_dist

def max_dist_two_list(list_points1, list_points2):
    """
    Compute the maximum distance between any two points from the two lists of points.

    NOT USED and DEPRECATED cdist() is much faster than this loop method. 
    min_cost is the best way probably.
    """
    max_dist = 0.0
    for point1 in list_points1:
        for point2 in list_points2:
            dist = distance(point1, point2)
            if dist > max_dist:
                max_dist = dist
    return max_dist

def avg_dist_closest_pts_two_lists(list_points1, list_points2, return_points=False):
    """
    Compute the average distance of the closest points between two lists of points.
    This function computes the distance between each point in list_points1 and the closest point in list_points2,
    and then averages these distances.
    DEPRECATED, cdist would do a much better job.
    """
    min_dist = np.inf
    min_dists = []
    for point1 in list_points1:
        for point2 in list_points2:
            dist = distance(point1, point2)
            if dist < min_dist:
                min_dist = dist
                points = [point1, point2]
        min_dists.append(min_dist)
    return np.mean(min_dists)

def list_to_x_y_z(list_of_points):
    x, y, z = [], [], []
    for point in list_of_points:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])
    return x, y, z


def x_y_z_to_list(x, y, z):
    list_of_points = []
    for i in range(len(x)):
        list_of_points.append([x[i], y[i], z[i]])
    return list_of_points


def find_extremal_points(points: List):
    max_distance = 0
    extremal_points = None

    # Generate all pairs of points
    for p1, p2 in combinations(points, 2):
        dist = distance(p1, p2)
        if dist > max_distance:
            max_distance = dist
            extremal_points = (p1, p2)

    return extremal_points, max_distance


def find_extremal_points_a(points: np.array):

    distances = cdist(points, points, metric="euclidean")
    pairs = np.column_stack(np.where(distances == np.max(distances)))
    # Pairs of points indexes are duplicated, so we need to filter them
    pairs = filter_pairs(pairs)
    max_distance = np.max(distances)
    return [points[pair] for pair in pairs], max_distance

def filter_pairs(pairs):
    seen = set()
    filtered_pairs = []

    for pair in pairs:
        # Convert pair to tuple for set operations
        pair_tuple = tuple(pair)
        reverse_pair_tuple = tuple(pair[::-1])

        if pair_tuple not in seen and reverse_pair_tuple not in seen:
            seen.add(pair_tuple)
            filtered_pairs.append(pair)

    return np.array(filtered_pairs)

def get_physical_coord_for_needle(sitk_needle, needle_idx:int=None, return_indexes:bool=False):
    """
    Get the physical coordinates of the points in the needle.

    Parameters
    ----------
    needle_array : np.ndarray
        Numpy array containing the needle.
    sitk_needles : SimpleITK.Image
        SimpleITK image containing the needles.
    Returns
    -------
    list
        List of physical coordinates.
    """

    pysical_coord_points = []
    needle_array = sitk.GetArrayFromImage(sitk_needle)
    # Numpy array indexes are not the same as physical coordinates
    if needle_idx is None:
        points_indexes = np.where(needle_array!=0)
    else:
        points_indexes = np.where(needle_array==needle_idx)

    array_indexes = []
    for point_index in zip(*points_indexes):
        # Inverting the indexes because SimpleITK uspiecewise_lineares (z, y, x) and numpy uses (x, y, z)
        int_pos = [int(point_index[i]) for i in range(3)][::-1]
        array_indexes.append(int_pos)
        physical_pos = sitk_needle.TransformIndexToPhysicalPoint(int_pos)
        pysical_coord_points.append(physical_pos)
        
    if return_indexes:
        return pysical_coord_points, array_indexes
    else:
        return pysical_coord_points

def create_group_from_labels(points:List[List[float]], labels:List[int]):
    """
    Create groups based on labels.
    """
    # Attach cluster labels to points
    clustered_points = list(zip(points, labels))

    groups = []
    for lab in np.unique(labels):
        group = []
        for point, label in clustered_points:
            if label == lab:
                group.append(point)
        # Sort the group by x coordinate
        group = sorted(group, key=lambda x: x[0])
        groups.append(group)
    return groups


def is_one_row(points:List[List[float]], thr_dist:float= 5.0, fit:str="spline",
                save_files:bool=False, save_dir:str="", print_details:bool=False):
    """
    Check if all points are in a single row.
    We first fit a line to the points and then we project the points 
    onto the line to see if their distance to the line is within a threshold. 
    """

    distances = []
    if save_files:
        projected_points = []

    if fit== "spline":
        points_2D = np.array(points)[:, :2]
        tck, u = fit_spline2D(points_2D, s=len(points_2D), k=3, nest=0)
        # Project the points onto the spline
        for pt in points:
            projected_point, t, d = project_point_to_spline(pt[:2], tck)
            if save_files:
                projected_points.append(projected_point.tolist())
            distances.append(d)
    else:
        assert fit == "line", "Only 'spline' and 'line' fits are supported."
        mean, direction = fit_line(points)

        # Project the points onto the line
        for pt in points:
            projected_point = project_point_to_line(pt, mean, direction, return3D=True).tolist()
            if save_files:
                projected_points.append(projected_point)
            distances.append(distance(projected_point, pt))

    if save_files:
        xs = np.array([pt[0] for pt in points])
        arg_max_x = np.argmax(xs, axis=0)
        arg_min_x = np.argmin(xs, axis=0)

        create_slicer_markup_segments(
            os.path.join(save_dir, f"is_one_row__direction_segments.mrk.json"), 
            [projected_points[arg_max_x], projected_points[arg_min_x]], 
            color=[1.0,0.8,0.7] # red for created data
            )
        create_slicer_markup_points(
            os.path.join(save_dir, f"is_one_row__pts.mrk.json"), 
            points, 
            color=[1.0,0.8,0.7] # red for created data
            )
        create_slicer_markup_points(
            os.path.join(save_dir, f"is_one_row__projected_pts.mrk.json"), 
            projected_points, 
            color=[1.0,0.8,0.7] # red for created data
            )
        
    if print_details:
        print("Distances to the spline fitted from all insertion points: ", describe_array(distances))
    # Check if all projected points are close to the line
    return np.max(distances) < thr_dist
    
def get_potential_mean_directions_oriented_rows(
        points:List[List[float]], save_files:bool=False, save_dir:str=""):
    """
    Get the mean points and direction of a group of points.
    Vector should be oriented in the direction of the rows.
    """
      
    mean = np.mean(points, axis=0)

    # Creating the insertion grid polygon will serve to align the identified
    # row direction to the sides of the polygon.
    points_2D = points[:, :2]  # Take only the x and y coordinates
    corners_hull = find_four_corners_convex_hull(points_2D, order_pca=False).tolist()
    corners_hull_3D = []
    for pt in corners_hull:
        corners_hull_3D.append(np.array([pt[0], pt[1], points[0][2]]).tolist())

    polygon_side_sizes = []
    polygon_side_pairs = []
    direction_convex_hull_sides = []
    for i in range(len(corners_hull)):
        polygon_side_sizes.append(
            distance(corners_hull[i], corners_hull[(i+1)%4]))
        polygon_side_pairs.append(
            [np.array(corners_hull[i]), np.array(corners_hull[(i+1)%4])])
        direction_convex_hull_sides.append(
            np.array(corners_hull[i]) - np.array(corners_hull[(i+1)%4]))
    # We select the direction that is perpendicular to the smallest side
    # of the polygon
    min_side_idx = np.argmin(polygon_side_sizes)
    if save_files:
        create_slicer_markup_points(
            os.path.join(save_dir, "smallest_side.mrk.json"), 
            [corners_hull_3D[min_side_idx], corners_hull_3D[(min_side_idx+1)%4]],

        )
        for i, d in enumerate(direction_convex_hull_sides):
            create_slicer_markup_points(
                os.path.join(save_dir, f"polygon_convexhull_side_{i}.mrk.json"), 
                [corners_hull_3D[i], corners_hull_3D[(i+1)%4]]
            )

    ## Not getting a direction strictly perpendicular to the smallest side
    # but getting the mean direction of the two polygon sides that surround
    # the smallest side.
    # Get the two sides that surround the smallest side
    potential_directions = []
    for i in range(2):
        vector_to_be_parallel_to = np.mean(
            # First side surrounding the smallest side
            [np.array(direction_convex_hull_sides[(min_side_idx+i+1)%4]) ,
            # Second side surrounding the smallest side
            np.array(-direction_convex_hull_sides[(min_side_idx+i+3)%4])], 
            axis=0
        )

        # Make 3D
        vector_to_be_parallel_to = np.array(
            [
                vector_to_be_parallel_to[0], vector_to_be_parallel_to[1], 0
            ]
        )
        
        normal_projected_pts = project_point_to_line(
            np.array(points), mean, vector_to_be_parallel_to, return3D=True)
        if save_files:
            save_segments_from_points(
                corners_hull_3D, mean, vector_to_be_parallel_to, save_dir,
                file_name=f"vector_to_be_parallel_to_{i}.mrk.json"
            )   
            create_slicer_markup_points(
                os.path.join(save_dir, f"projected_normal_pts_cetortobeparallel_{i}.mrk.json"), 
                normal_projected_pts.tolist()
                )
        normal_distances = []
        for pt, n_pt in zip(points, normal_projected_pts.tolist()):
            normal_distances.append(distance(pt, n_pt))
        
        # Force the direction to be towards x axis. Should avoid very large angles.
        if vector_to_be_parallel_to[0] < 0:
            vector_to_be_parallel_to = -vector_to_be_parallel_to
        potential_directions.append(vector_to_be_parallel_to)
       
    return mean, potential_directions

def divide_points_into_rows_via_binning(points: List[List[float]], n_clusters: int) -> List[List[List[float]]]:
    """
    Divide points into rows using a binning approach.
    """
    ys = np.array(points)[:, 1]
    y_min, y_max = np.min(ys), np.max(ys)
    bin_edges = np.linspace(y_min, y_max, n_clusters + 1)

    # Step 2: Assign each point to a row/bin based on its y
    labels_1st_assign = np.digitize(ys, bin_edges) - 1
    # (digitize bins are 1-based, so subtract 1)

    # Fix edge case: y == y_max should go in last bin
    labels_1st_assign[labels_1st_assign == n_clusters] = n_clusters - 1
    print("labels_1st_assign", labels_1st_assign)

    # Step 3: Group points by their assigned labels
    rows = [[] for _ in range(n_clusters)]
    for pt, label in zip(points, labels_1st_assign):
        rows[label].append(pt)
    return rows

def get_binning_metric_for_n_clusters(
        points: List[List[float]], n_clusters: int=10) -> float:
    """
    Get the metric for the specified number of clusters.
    """
    best_n_cluster = 0 
    best_metric = 99999
    best_grps = None
    for n_cluster in range(2, n_clusters + 1):
        groups = divide_points_into_rows_via_binning(points, n_cluster)
        metric = get_parallel_to_x_axis_metric(groups)
        print(f"n_clusters: {n_cluster}, metric: {metric}")
        if metric < best_metric:
            best_metric = metric
            best_n_cluster = n_cluster
            best_grps = groups
    print(f"Best n_clusters: {best_n_cluster}, Best metric: {best_metric}")
    return best_metric, best_grps, best_n_cluster

def get_nb_rows_from_dbscan(points: List[List[float]], hdbscan:bool=True) -> int:
    """
    Get the number of rows from the points using DBSCAN clustering.
    The number of rows is the number of clusters found by DBSCAN.
    We are testing different eps values.
    """
    if not hdbscan:
        best_nb_rows = None
        metric = 99999
        best_labels = None
        # eps is the max distance between two points of the same cluster.
        # This is the most important parameter of DBSCAN. An within a row
        # points should in theory not be not be to far appart, the insertion 
        # grid contains squares of 10mm so 20 mm should be a good starting point.
        for eps in [10, 15, 20]:
            clustering_model = DBSCAN(eps=eps, min_samples=2, metric='euclidean')
            clustering_model.fit(points)
            if len(set(clustering_model.labels_)) > 1:
                temp_metric = get_parallel_to_x_axis_metric(
                    create_group_from_labels(points, clustering_model.labels_)
                )
                if temp_metric < metric:
                    metric = temp_metric
                    best_nb_rows = len(set(clustering_model.labels_))
                    best_labels = clustering_model.labels_
    else:
        # HDBSCAN automatically selects the best eps value.
        hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=2, metric='euclidean')
        hdbscan_model.fit(points)
        metric = get_parallel_to_x_axis_metric(
            create_group_from_labels(points, hdbscan_model.labels_)
        )
        best_nb_rows = len(set(hdbscan_model.labels_))
        best_labels = hdbscan_model.labels_
   
    return metric, best_nb_rows, best_labels


def get_closeness_to_line_metric(groups:List[List[List[float]]]) -> float:
    """
    Fit a line on each group of points and compute the closeness to the line.
    """
    closeness_values = []
    for group in groups:
        if len(group) < 2:
            continue
        # Fit a line to the group
        mean, direction = fit_line(group)
        # Compute the closeness to the line
        projected_points = project_point_to_line(
            np.array(group), mean, direction, return3D=True)
        # Compute the closeness as the mean distance to the line
        distances = []
        for pt, projected_pt in zip(group, projected_points.tolist()):
            distances.append(distance(pt, projected_pt))
        closeness = np.mean(distances) if distances else 0.0
        closeness_values.append(closeness)
    return np.mean(closeness_values) if closeness_values else 0.0

def get_compactness_metric(groups:List[List[List[float]]]) -> float:
    """
    Compute the compactness metric for the groups of points.
    Compactness is defined as the area of the bounding box
    of a group.
    """
    compactness_values = []
    for group in groups:
        if len(group) < 2:
            continue
        # Get the bounding box of the group
        xs = [pt[0] for pt in group]
        ys = [pt[1] for pt in group]
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        # Compute the area of the bounding box
        area = (x_max - x_min) * (y_max - y_min)
        compactness_values.append(area)
    return np.mean(compactness_values) if compactness_values else 0.0


def compute_angle_to_x_axis(group:List[List[float]]) -> float:
    """
    Compute the angle to x axis of a group of points.
    """
    mean, direction = fit_line(group)
    if direction[0] < 0:
        direction = -direction
    x_axis = np.array([1, 0, 0])
    angle = get_angle_between_two_vectors(direction, x_axis)
    return angle

def get_parallel_to_x_axis_metric(groups:List[List[List[float]]], criterion:str="max") -> float:
    """
    Compute the angle to x axis metric for the groups of points.
    """
    angles = []
    for group in groups:
        if len(group) < 2:
            continue
        # Compute the angle of the group
        angle = compute_angle_to_x_axis(group)
        # Should be positive anyway but we ensure it is positive
        angles.append(np.abs(angle))
    if criterion == "mean":
        return np.mean(angles) if angles else 0.0
    elif criterion == "max":
        return np.max(angles) if angles else 0.0
    else:
        raise ValueError("Criterion must be 'mean' or 'max'.")

######################################## 3DSlicer utils ########################################


def save_segments_from_points(
        points:List[List[float]], mean:List[float], 
        direction:List[float], save_dir:str, 
        file_name:str="segment.mrk.json", color:List[float]=[1.0,0.8,0.7], 
        save_pts:bool=False):
    """
    Save the segments from the points to a file to open them in 3D Slicer.
    Points should be 3D.
    """
    assert file_name.endswith(".mrk.json"), "File name must end with .mrk.json to be used in 3D Slicer."
    assert len(points[0]) == 3, "Points should be 3D."
    all_xs = [pt[0] for pt in points]
    arg_max_x = np.argmax(all_xs)
    arg_min_x = np.argmin(all_xs)
    pt1_segment = project_point_to_line(
        np.array(points[arg_max_x]), mean, direction, return3D=True
        ) 
    pt2_segment = project_point_to_line(
        np.array(points[arg_min_x]), mean, direction, return3D=True
        )
    create_slicer_markup_segments(
        os.path.join(save_dir, file_name), 
        [pt1_segment.tolist(), pt2_segment.tolist()], 
        color=color
        )
    if save_pts:
        create_slicer_markup_points(
            os.path.join(save_dir, f"{file_name.split('.')[0]}_pts.mrk.json"), 
            points, 
            color=color
            )
        
def save_grps_3Dslicer(grps:List[List[List[float]]], save_dir:str, key:str=""):
    """
    Save the groups to a json file.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for grp_idx, grp in enumerate(grps):
        mean, direction = fit_line(grp)

        row_color = np.random.uniform(low=0, high=1, size=(3)).tolist()
        save_segments_from_points(
            grp, mean, direction, save_dir, 
            file_name=f"{key}row_{grp_idx}_direction_segments.mrk.json", 
            color=row_color, save_pts=True
            )  

########################## SITK FUNCTION HELPERS ##########################


def transform_image(ref_img, img_to_resample, transform):
    img_to_resample_arr_og = sitk.GetArrayViewFromImage(img_to_resample)
    values, counts = np.unique(img_to_resample_arr_og, return_counts=True)
    ind = np.argmax(counts)
    bg = values[ind]
    transformed_img = sitk.Resample(
        image1=img_to_resample,
        referenceImage=ref_img,
        transform=transform,
        interpolator=sitk.sitkBSpline,
        defaultPixelValue=int(bg),
        outputPixelType=img_to_resample.GetPixelID(),
    )
    return transformed_img


def rotate_volume(volume, rotation_transform, interpolator=sitk.sitkLinear):

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume)
    resampler.SetTransform(rotation_transform)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)  # Set to a suitable background value
    resampler.SetOutputPixelType(volume.GetPixelID())

    return resampler.Execute(volume)


########################## FITTING FUNCTION HELPERS ##########################


# Define a linear function to fit each segment
def linear_func(t, a, b):
    return a * t + b


def fit_segment(segment):
    t = np.arange(len(segment))
    params_x, _ = curve_fit(linear_func, t, segment[:, 0])
    params_y, _ = curve_fit(linear_func, t, segment[:, 1])
    params_z, _ = curve_fit(linear_func, t, segment[:, 2])
    return params_x, params_y, params_z


def fit_line(points, return_all_pcs:bool=False):
    # Step 1: Center the data
    mean = np.mean(points, axis=0)
    centered_points = points - mean

    # Step 2: Apply PCA (compute the first principal component)
    U, S, Vt = np.linalg.svd(centered_points)
    if return_all_pcs:
        return mean, Vt
    direction = Vt[0]  # The first principal component
    return mean, direction


def project_point_to_line(
        points:np.ndarray|List[float]|List[List[float]], 
        mean:np.ndarray|List[float], direction:np.ndarray|List[float], 
        return3D:bool=False):
    """
    Project a point onto the line defined by a mean point and direction vector.

    Parameters:
    - point: The point to project (numpy array).
    - mean: The mean point on the line (numpy array).
    - direction: The direction vector of the line (numpy array).

    Returns:
    - t: The scalar value such that mean + t * direction is the projection of the point on the line.
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean)
    if not isinstance(direction, np.ndarray):
        direction = np.array(direction)
    assert mean.ndim==direction.ndim, (
        f"The mean and direction must have the same number of dimensions but has {mean.ndim} and {direction.ndim}."
    )
    assert np.any(points.shape[i] == 3 for i in range(len(points.shape))), (
        f"The points must be 3D but has shape {points.shape}."
    )

    # Calculate the projection scalar
    t = np.dot(points - mean, direction) / np.dot(direction, direction)
    if return3D:
        # Calculate the projected points in 3D space
        if points.ndim==2 and mean.ndim==1:
            t = np.expand_dims(t, axis=1)
            direction = np.expand_dims(direction, axis=0)
            mean = np.expand_dims(mean, axis=0)
        projected_points = mean + t * direction
        return projected_points
    else:
        return t


def fit_spline(points:np.ndarray, s:int, k:int, nest:int):

    if not isinstance(points, np.ndarray):
        points = np.array(points)
    assert points.ndim == 2, "Points must be a 2D array."
    assert points.shape[1] == 3, "Points must have 3 coordinates (x, y, z)."
    # Transpose to get (x, y, z) arrays
    x, y, z = points.T

    # Fit a B-spline to the noisy data
    # nest is important. If not set the interpolation will have a lot of knots
    # due to the interslice gaps.
    tck, u = splprep([x, y, z], s=s, k=k, nest=nest)  # s is the smoothing factor

    return tck, u

def fit_spline2D(points:np.ndarray, s:int, k:int, nest:int):

    # Transpose to get (x, y) arrays
    x, y = points.T

    # Fit a B-spline to the noisy data
    # nest is important. If not set the interpolation will have a lot of knots
    # due to the interslice gaps.
    tck, u = splprep([x, y], s=s, k=k, nest=nest)  # s is the smoothing factor

    return tck, u

def project_point_to_spline(point:List[float], tck:float):
    """
    Given a point and a step, we want to find the point on the spline function
    that is at a distance step from the point.
    """
    def step_func(t: float) -> float:
        point_coords = splev(t, tck)
        return abs(distance(point, point_coords))
    opt = minimize_scalar(
        step_func, bounds=(-0.5, 1.5), method="bounded"
    )
    return np.array(splev(opt.x, tck)), opt.x, opt.fun


########################## REORDERING POINTS ##########################


def reorder_points_nneighbor(points):

    extremal_points, _ = find_extremal_points(points)
    ordered_points = [extremal_points[0]]
    index_to_remove = np.where(points == extremal_points[0])
    points = np.delete(points, index_to_remove, axis=0)

    # Here we order the points based on their distance to each other
    while len(points) > 0:
        last_point = ordered_points[-1]
        distances = cdist(np.array([last_point]), np.array(points))
        nearest_index = np.argmin(distances)
        ordered_points.append(points[nearest_index])
        points = np.delete(points, nearest_index, axis=0)

    return np.array(ordered_points)


def solve_tsp(dist_matrix):
    tsp_size = len(dist_matrix)
    num_routes = 1  # The number of routes, which is 1 in the TSP.
    depot = 0  # The depot is the starting point of the route.

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(tsp_size, num_routes, depot)

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(
            dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
        )

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    assignment = routing.SolveWithParameters(search_parameters)

    index = routing.Start(0)
    plan_output = []
    while not routing.IsEnd(index):
        plan_output.append(manager.IndexToNode(index))
        index = assignment.Value(routing.NextVar(index))
    plan_output.append(
        manager.IndexToNode(index)
    )  # add the final point back to the depot
    return plan_output


def reorder_points_tsp(points):
    # Create distance matrix for the points
    dist_matrix = distance_matrix(points, points)

    # Solve TSP to get the optimal order
    optimal_order = solve_tsp(dist_matrix)

    # Reorder points according to TSP solution
    ordered_points_tsp = points[optimal_order]
    return ordered_points_tsp

########################## GEOMETRY HELPER ##########################

def find_corners_PCA(points:List[List[float]]):
    """
    Find the corners of a set of points using PCA.
    """
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(points)

    min_x = transformed[np.argmin(transformed[:, 0])]
    max_x = transformed[np.argmax(transformed[:, 0])]
    min_y = transformed[np.argmin(transformed[:, 1])]
    max_y = transformed[np.argmax(transformed[:, 1])]

    # Back-transform to original space
    corners = np.vstack([min_x, max_x, min_y, max_y])
    return pca.inverse_transform(corners)

def find_four_corners_convex_hull(points:List[List[float]], order_pca:bool=False):
    """
    Find the corners of a set of points using the convex hull.
    Find the extreme points of the convex hull, then select the 
    four corners that form the largest polygon area.
    """
    # Compute the convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    assert hull_points.shape[0] >= 4, "Convex hull must have at least 4 points"

    # Try all combinations of 4 points on the hull
    max_area = 0
    best_quad = None
    for quad in combinations(hull_points, 4):
        quad = np.array(quad)
        # Shoelace formula to compute the area of a polygon
        area = 0.5 * np.abs(np.dot(quad[:, 0], np.roll(quad[:, 1], 1)) - 
                            np.dot(quad[:, 1], np.roll(quad[:, 0], 1)))
        if area > max_area:
            max_area = area
            best_quad = quad

    if order_pca:
        # Order the points using PCA. This is important to have a consistent
        # ordering of the points if we want to compute directions in downstream 
        # applications.
        pca = PCA(n_components=2)
        transformed = pca.fit(points).transform(best_quad)
        min_x = transformed[np.argmin(transformed[:, 0])]
        max_x = transformed[np.argmax(transformed[:, 0])]
        min_y = transformed[np.argmin(transformed[:, 1])]
        max_y = transformed[np.argmax(transformed[:, 1])]
        best_quad = pca.inverse_transform(np.vstack([min_x, max_x, min_y, max_y]))
        raise DeprecationWarning(
            "The PCA ordering is not working properly, with 4 points fit transformed it " \
            "can sometimes give same point as min or max of different axes."
        )
    else:
        # Order the points using the original order given by Convex Hull, 
        # which is counter clock wise.
        order = []
        for i in range(4):
            order.append(np.where(np.all(hull_points == best_quad[0], axis=1))[0][0])
        assert np.all(np.sort(order) == np.array(order)), (
            "Selecting best 4 corners changed their counterclockwise order"
        )
    return best_quad

def project_point_onto_plane(point, plane_point, plane_normal):
    # Vector from plane point to the point to project
    v = point - plane_point
    # Distance from point to plane along the normal
    distance = np.dot(v, plane_normal)
    # Projection = original point minus the component along the normal
    projected = point - distance * plane_normal
    return projected, abs(distance)

def check_corners_convex_hull(corners:List[List[float]], threshold_mm:float=10.):
    """
    Projecting each of the points onto the plane given by the corners
    and check if there distance to the plane is small.
    If the distance is not small, then the corners are not a good representation
    of the points.
    """
    assert len(corners) == 4, "There must be 4 corners"

    all_points_close_to_same_plane = True
    for i in range(4):
        # Create a plane from 3 points
        # Project the 4th point onto the plane
        # Compute the distance to projected point
        p1 = np.array(corners[i])
        p2 = np.array(corners[(i + 1) % 4])
        p3 = np.array(corners[(i + 2) % 4])
        p4 = np.array(corners[(i + 3) % 4])
        # Step 1: Plane from first 3 points
        normal = compute_surface_normal(p1, p2, p3)

        # Step 2: Project the 4th point
        projected_p4, d = project_point_onto_plane(p4, p1, normal)

        if d > threshold_mm:
            all_points_close_to_same_plane = False

    problematic_pt = None
    if not all_points_close_to_same_plane:
        # We find the problematic point, which we define as the point 
        # the farthest to its two neighbors.
        max_metric = 0
        for i in range(1, 5):
            # Compute the distance to the previous and next points
            prev_pt = corners[i - 1]
            next_pt = corners[(i + 1) % 4]
            current_pt = corners[i % 4]
            metric = distance(current_pt, prev_pt) + distance(current_pt, next_pt)
            if metric > max_metric:
                max_metric = metric
                problematic_pt = current_pt
    return all_points_close_to_same_plane, problematic_pt

def get_catheter_directions_from_dwell_positions(
        dwell_positions: Dict[str, List[List[float]]], button_first: bool = True) -> List[List[float]]:
    # Get a direction vector for each of the catheters
    catheter_directions = []
    for catheter_random_key in dwell_positions.keys():
        d_pos = dwell_positions[catheter_random_key]
        mean, direction = fit_line(d_pos)
        if button_first:
            # We want to have the catheter pointing towards the button/tip.
            # Tip is assumed to be the first dwell position
            if np.dot(direction, np.array(d_pos[0]) - np.array(d_pos[-1])) < 0:
                direction = -direction
        else:
            # We want to have the catheter direction towards last dwell position.
            if np.dot(direction, np.array(d_pos[-1]) - np.array(d_pos[0])) < 0:
                direction = -direction
        catheter_directions.append(direction)
    return catheter_directions

def get_angle_between_two_vectors(v1:List[float]|np.ndarray, v2:List[float]|np.ndarray):
    assert len(v1) == len(v2), "Vectors must be of the same length"
    # Compute the dot product and the magnitudes
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Compute the angle in radians
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))

    # Optional: convert to degrees
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def compute_surface_normal_4_points(p1, p2, p3, p4):
    """
    Compute the surface normal of a quadrilateral defined by four points.
    The points should be in 3D space.
    """
    assert len(p1) == len(p2) == len(p3) == len(p4) == 3, "Points must be in 3D space"
    # Two triangles: (p1, p2, p3) and (p1, p3, p4)
    n1 = compute_surface_normal(p1, p2, p3)
    n2 = compute_surface_normal(p1, p3, p4)
    # Average the two normals
    n = (n1 + n2) / 2
    n /= np.linalg.norm(n)
    return n

def compute_surface_normal(p1, p2, p3):
    """
    Compute the surface normal of a triangle defined by three points.
    The points should be in 3D space.
    """
    assert len(p1) == len(p2) == len(p3) == 3, "Points must be in 3D space"
    # One triangle: (p1, p2, p3)
    n = np.cross(p2 - p1, p3 - p1)
    n /= np.linalg.norm(n)
    return n


def project_on_z_coord(points:List[List[float]], z_coord:float, fit_type:str="line") -> List[List[float]]:
    """
    Project the points on a given z coordinate.
    If fit_type is "line", we fit a line to the points and project them on the line.
    If fit_type is "spline", we fit a spline to the points and project them on the spline.
    """
    assert fit_type in ["line", "spline"], "fit_type must be 'line' or 'spline'"
    
    if fit_type == "spline":

        tck, u = fit_spline(points, s=1000, k=3, nest=0)
        def func(t_):
            x, y, z = splev(t_, tck)
            return abs(z - z_coord)
    else:
        mean, direction = fit_line(points)
        def func(t_):
            x = mean[0] + t_ * direction[0]
            y = mean[1] + t_ * direction[1]
            z = mean[2] + t_ * direction[2]
            return abs(z - z_coord)

    if fit_type == "spline":
        opt = minimize_scalar(
            func, bounds=(-0.5, 1.5), method="bounded"
        )
        return np.array(splev(opt.x, tck)).tolist(), opt.x, opt.fun
    else:
        opt = minimize_scalar(
            # Catheter is on average 10cm long, so we can use a large range
            # Here from middle point of the catheter we take 20 cm in both 
            # directions as the range for optimization.
            func, bounds=(-2000., 2000.), method="bounded"
        )
        return np.array([
            mean[0] + opt.x * direction[0],
            mean[1] + opt.x * direction[1],
            mean[2] + opt.x * direction[2]
        ]).tolist(), opt.x, opt.fun

    
########################## MISCELLANEOUS ##########################


def describe_array(array, percentile:bool=False):
    """
    Describe the array.
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if array.size == 0:
        return "No values"
    mean = np.mean(array)
    std = np.std(array)
    min_val = np.min(array)
    max_val = np.max(array)
    if percentile:
        percentile_90 = np.percentile(array, 90)
        percentile_95 = np.percentile(array, 95)
        percentile_99 = np.percentile(array, 99)
        median = np.median(array)
        return "{mean:.2f} +/- {std:.2f} [{min_val:.2f}, {max_val:.2f}] median {med:.2f}, percentiles: 90\\% {percentile_90:.2f} 95\\% {percentile_95:.2f} 99\\% {percentile_99:.2f}".format(
            mean=mean, std=std, min_val=min_val, max_val=max_val, med=median, percentile_90=percentile_90, percentile_95=percentile_95, percentile_99=percentile_99
        )
    else:
        return "{mean:.2f} +/- {std:.2f} [{min_val:.2f}, {max_val:.2f}]".format(
            mean=mean, std=std, min_val=min_val, max_val=max_val
        )


def determine_breast_side(ct_volume:sitk.Image, dwellpositions:List[List[float]], center_of_mass:bool=True) -> str:
    """
    Determine the breast side based on the dwell positions and the CT volume.
    This is a heuristic based on the position of the dwell positions relative to the CT volume.
    We first identify a pseudo-body contour, then we get the mid x coordinate of the body contour.
    We finally compare that to the x coordinate of the dwell positions.

    Using the center of mass of the body to determine mid_x coord is more robust than using the min/max coordinates.
    """
    assert isinstance(dwellpositions, list) and len(dwellpositions[0]) == 3, (
        "Dwell positions must be a list of [x, y, z] coordinates"
    )
    ct_npy = sitk.GetArrayFromImage(ct_volume)
    body_pseudo_mask = np.where((ct_npy > -500) & (ct_npy < 500), 1, 0)
    nz = np.where(body_pseudo_mask == 1)

    if center_of_mass:
        xs = np.array(nz[2])
        mean_xs = np.mean(xs)
        mid_x_coord = ct_volume.TransformIndexToPhysicalPoint((int(np.round(mean_xs)), 0, 0))[0]
    else:
        # Get the min and max coordinates of the body contour
        zmin, ymin, xmin = [int(np.min(a)) for a in nz]
        zmax, ymax, xmax = [int(np.max(a)) + 1 for a in nz]
        first_nz_pt = ct_volume.TransformIndexToPhysicalPoint((xmin, ymin, zmin))
        last_nz_pt = ct_volume.TransformIndexToPhysicalPoint((xmax, ymax, zmax))
        mid_x_coord = (first_nz_pt[0] + last_nz_pt[0]) / 2
    
    # Compare mid_x_coord with dwell positions
    dwell_x_coords = np.array([pos[0] for pos in dwellpositions])
    dp_before_mid = np.sum(dwell_x_coords < mid_x_coord)
    dp_after_mid = np.sum(dwell_x_coords > mid_x_coord)
    
    if dp_before_mid < dp_after_mid:
        # We see the dwell positions on the right side of the mid x coordinate on an axial slice
        # which is the left side of the patient breast but the the right side in the Dr's view.
        assert dp_after_mid > 2 * dp_before_mid, (
            "The number of dwell positions after the mid x coordinate is not significantly larger than before. "
            "This could indicate an error in the dwell positions or the CT volume."
        )
        return "right"
    else:
        # We see the dwell positions on the left side of the mid x coordinate on an axial slice
        # but it is actually the right side of the breast.
        assert dp_before_mid > 2 * dp_after_mid, (
            "The number of dwell positions before the mid x coordinate is not significantly larger than after. "
            "This could indicate an error in the dwell positions or the CT volume."
        )
        return "left"

######################################## 3DSlicer catheter utils ########################################

def get_slicer_marker_pt_dict():
    """
    This function returns a dictionary that can be used to save a list of points in 3D Slicer.
    """
    slicer_dict = {}
    slicer_dict["@schema"] = (
        "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#"
    )
    markup_template = {
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "coordinateUnits": "mm",
            "locked": False,
            "fixedNumberOfControlPoints": False,
            "labelFormat": "%N-%d",
            "lastUsedControlPointNumber": 1,
            "controlPoints": [],
            "measurements": [],
            "display": {
                "visibility": True,
                "opacity": 1.0,
                "color": [0.4, 1.0, 1.0],
                "selectedColor": [1.0, 0.5000076295109484, 0.5000076295109484],
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": False,
                "pointLabelsVisibility": True,
                "textScale": 2.3000000000000004,
                "glyphType": "Sphere3D",
                "glyphScale": 3.0,
                "glyphSize": 5.0,
                "useGlyphScale": True,
                "sliceProjection": False,
                "sliceProjectionUseFiducialColor": True,
                "sliceProjectionOutlinedBehindSlicePlane": False,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": False,
                "translationHandleVisibility": True,
                "rotationHandleVisibility": True,
                "scaleHandleVisibility": True,
                "interactionHandleScale": 3.0,
                "snapMode": "toVisibleSurface",
            },
        }
    slicer_dict["markups"] = []
    ctrl_pt_dict_template = {
        # id, label and position are to be updated
                "id": None,
                "label": None,
                "description": "",
                "associatedNodeID": "vtkMRMLScalarVolumeNode32",
                "position": None,
                "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": False,
                "visibility": True,
                "positionStatus": "defined",
            }
    return slicer_dict, markup_template, ctrl_pt_dict_template

def create_slicer_markup_segments(output_path, point_list, color=None, remove_text=True, previous_dict:dict=None):
    """
    This function creates a json file that can be loaded in 3D Slicer to visualize a list of points.
    If there exists a previous_dict, we will add another sequence of points to the existing file.
    """
    assert len(point_list) == 2, "You should be porviding a list of 2 points if you want to plot segments."
    slicer_dict = create_slicer_markup_points(
        output_path, point_list, color=color, remove_text=remove_text, previous_dict=previous_dict
        )
    # Changing type of markup
    for i in range(len(slicer_dict["markups"])):
        slicer_dict["markups"][i]["type"] = "Line"
    # Overwriting the points markup with segment markupo\
    with open(output_path,"w") as f:
        json.dump(slicer_dict, f, indent=4)
    return slicer_dict 

def create_slicer_markup_points(output_path, point_list, color=None, remove_text=True, previous_dict:dict=None):
    """
    This function creates a json file that can be loaded in 3D Slicer to visualize a list of points.
    If there exists a previous_dict, we will add another sequence of points to the existing file.
    """
    if previous_dict is not None:
        slicer_dict = previous_dict
        _ , markup_dict, ctrl_pt_dict = get_slicer_marker_pt_dict()
    else:
        slicer_dict, markup_dict, ctrl_pt_dict = get_slicer_marker_pt_dict()
    slicer_dict["markups"].append(markup_dict)
    if color is not None:
        assert isinstance(color, list) and len(color) == 3, "Color should be a list of 3 elements."
        assert isinstance(color[0], float) and color[0] <= 1.0 and color[0] >= 0.0, "Color should be a float between 0 and 1."
        if previous_dict is not None:
            slicer_dict["markups"][-1]["display"]["color"] = color
        slicer_dict["markups"][-1]["display"]["selectedColor"] = color
    assert output_path[-9:] == ".mrk.json", "you need this extension for your file name."
    file_name_template = os.path.basename(output_path)[:-9]
    if remove_text:
        slicer_dict["markups"][-1]["display"]["textScale"] = 0.0
    for pt_idx, pt in enumerate(point_list):
        temp_ctrl_pt_dict = ctrl_pt_dict.copy()
        temp_ctrl_pt_dict["id"] = str(pt_idx + 1)
        temp_ctrl_pt_dict["position"] = pt
        temp_ctrl_pt_dict["label"] = f"{file_name_template}-{pt_idx+1}"
        slicer_dict["markups"][-1]["controlPoints"].append(temp_ctrl_pt_dict)
    if not os.path.isdir(os.path.dirname(output_path)):
        print(f"Creating directory {os.path.dirname(output_path)}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path,"w") as f:
        json.dump(slicer_dict, f, indent=4)

    return slicer_dict

def create_marker_pts_from_catheter_table(output_path, catheter_table, one_markup_per_catheter=False, color=None):
    assert output_path.endswith(".mrk.json"), "You need to provide a file name with the extension .mrk.json"
    out_dir = os.path.dirname(output_path)
    out_name = os.path.basename(output_path)[:-9]
    os.makedirs(out_dir, exist_ok=True)
    for catheter_idx, catheter in enumerate(catheter_table):
        point_list = [dp["position"].tolist() for dp in catheter["dwells"]]
        if one_markup_per_catheter:
            outpath = os.path.join(out_dir,f"{out_name}_{catheter_idx}.mrk.json")
        else:
            outpath = os.path.join(out_dir,f"{out_name}.mrk.json")
        if catheter_idx==0 or one_markup_per_catheter:
            slicer_dict = create_slicer_markup_points(
                outpath, point_list, color=color)
        else:
            slicer_dict = create_slicer_markup_points(
                outpath, point_list, previous_dict=slicer_dict, color=color)


def create_marker_pts_from_catheter_dict(
        output_path, catheter_dict, one_markup_per_catheter=False, color=None):
    assert output_path.endswith(".mrk.json"), "You need to provide a file name with the extension .mrk.json"
    out_dir = os.path.dirname(output_path)
    out_name = os.path.basename(output_path)[:-9]
    cathter_idx = 0
    for catheter_key, catheter_pts in catheter_dict.items():
        # assert len(point_list) != 0, "No points found for catheter."
        if one_markup_per_catheter:
            outpath = os.path.join(out_dir,f"{out_name}_{catheter_key}.mrk.json")
        else:
            outpath = os.path.join(out_dir,f"{out_name}.mrk.json")
        if cathter_idx==0 or one_markup_per_catheter:
            slicer_dict = create_slicer_markup_points(
                outpath, catheter_pts, color=color)
        else:
            slicer_dict = create_slicer_markup_points(
                outpath, catheter_pts, previous_dict=slicer_dict, color=color)
        cathter_idx += 1

########################## GENERAL USE ##########################


def sitk_crop(image, bounding_box):
    """
    Crop the image to the bounding box
    """
    return sitk.RegionOfInterest(
        image,
        # bounding_box[0:3] is the x_min, y_min, z_min
        bounding_box[int(len(bounding_box) / 2) :],
        # bounding_box[3:6] is the x_size, y_size, z_size
        bounding_box[0 : int(len(bounding_box) / 2)],
    )

########################## AIR + KNOWLEDGE BASED CROPPING ##########################

def get_air_bounds(volume, inside_value=0, outside_value=1):
    bin_image = sitk.OtsuThreshold(volume, inside_value, outside_value)

    # Get the bounding box of the anatomy
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(bin_image)
    bounding_box_air = label_shape_filter.GetBoundingBox(outside_value)
    return bounding_box_air

def crop_volumes_around_mask(ct:sitk.Image, mask:sitk.Image, margin_mm:float=0):
    """
    Crops the CT and the mask around the mask with a margin of margin_mm
    Also crops the remaining air around the body.
    """
    cropped_mask, bounding_box = crop_around_mask(mask, margin_mm=margin_mm)
    # cropping based on the mask
    ct_cropped_around_mask = sitk_crop(ct, bounding_box)

    # cropping remaining air
    air_bb = get_air_bounds(ct_cropped_around_mask)

    return sitk_crop(ct_cropped_around_mask, air_bb), sitk_crop(cropped_mask, air_bb)

def crop_around_mask(volume:sitk.Image, margin_mm:float=0., use_sitk:bool=True):
    """
    Gets cropping boundaries to crop any volume around the specified mask.
    """
    spacings = volume.GetSpacing()
    if use_sitk:
        non_zero_mask = sitk.BinaryThreshold(volume, lowerThreshold=1) 

        # Get bounding box from contour
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(non_zero_mask)
        bounding_box = list(label_shape_filter.GetBoundingBox(1))

        # Add margin
        for i in range(0, 3):
            bounding_box[i] = max(0, bounding_box[i] - int(margin_mm / spacings[i]))
        for i in range(3, 6):
            bounding_box[i] = min(
                # min between bounding box with margin size and remaining volume shape from bounding box
                volume.GetSize()[i-3] - bounding_box[i-3], 
                bounding_box[i] + int(margin_mm / spacings[i-3])*2
                )
    else:
        # Doing everything with numpy, slower.
        mask_array = sitk.GetArrayFromImage(volume)
        # Get nonzero mask indices efficiently
        nz = np.where(mask_array > 0) # Faster than np.nonzero(mask_array)
        if len(nz[0]) == 0:
            raise ValueError("No nonzero voxels found in catheter_contours.")
        # Compute min/max for each axis directly
        zmin, ymin, xmin = [int(np.min(a)) for a in nz]
        zmax, ymax, xmax = [int(np.max(a)) + 1 for a in nz]
        # Add margin and clamp to array shape
        shape = mask_array.shape
        zmin = max(0, zmin - int(margin_mm / spacings[2]))
        ymin = max(0, ymin - int(margin_mm / spacings[1]))
        xmin = max(0, xmin - int(margin_mm / spacings[0]))
        zmax = min(shape[0], zmax + int(margin_mm / spacings[2]))
        ymax = min(shape[1], ymax + int(margin_mm / spacings[1]))
        xmax = min(shape[2], xmax + int(margin_mm / spacings[0]))
        # bounding_box: [x, y, z, size_x, size_y, size_z]
        bounding_box = [
            xmin, ymin, zmin,
            xmax - xmin, ymax - ymin, zmax - zmin
        ]

    return sitk_crop(volume, bounding_box), bounding_box
