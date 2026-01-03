import os
import glob
import tqdm

from typing import List
import SimpleITK as sitk
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splev
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from scipy import ndimage

from brachyutils.geometry.catheter_utils.utils import create_slicer_markup_points
from ai_assisted_brachy.preprocessing.cropping import crop_around_mask
from brachyutils.geometry.catheter_utils.catheter_setup import CatheterSetUp
from brachyutils.geometry.catheter_utils.digitization.rotation import (
    calculate_rotation_matrix,
    create_rotation_transform,
    rotate_volume
)
from brachyutils.geometry.catheter_utils.utils import (
    distance,
    find_extremal_points_a,
    fit_line,
    fit_spline,
    get_physical_coord_for_needle,
    list_to_x_y_z
)


class NeedleSplineCreator:

    def __init__(self, needle_contour:sitk.Image, multiclass:bool=False, crop:bool=True, dilate:bool=True) -> None:
        """Takes a list of points belonging to one catheter, rotates the catheter so that we roughly have
        one axis that is orthogonal to the catheter. Then it is easy to just take the central points contoured
        on each slice (center of the catheter) and interpolate a spline between those points.
        """

        self.multiclass = multiclass
        self.points = None
        self.needle_contour = None
        self.rotated_needle = None
        self.rotation_matrix = None
        self.transform = None
        self.inverse_rotation = None
        self.original_catheter_pts_rotated = None
        self.original_central_points = None
        self.rotated_center_points = None
        self.central_points = None
        self.tck = None
        self.u = None
        self.bounding_box = None
        self.vox_end_coord = None
        self.vox_tip_coord = None
        self.input_catheter_contour = needle_contour

        # Considering all classes as one class
        if self.multiclass:
            # We only have 3 classes normally but upper threshold can be any above.
            # Voxel whose value is within [lowerThreshold, upperThreshold] are insideValue (inclusive bounds), 
            # otherwise they are outsideValue.
            needle_contour = sitk.BinaryThreshold(
                needle_contour, 
                lowerThreshold=1.0, upperThreshold=255.0, 
                insideValue=1, outsideValue=0)
        

        self.bounding_box = None
        if crop:
            # Croping the needle volume so that computation is faster
            self.needle_contour, self.bounding_box = crop_around_mask(needle_contour, margin_mm=5)
        else:
            self.needle_contour = needle_contour

        self.central_points = None

        
        # Dilation can help some cases were contour is only one voxel and is lost in the rotation
        if dilate:
            # Saving endpoints coord to avoid changing them during dilation
            endpts_coord, _, _ = self.get_endpoint_line(sitk.GetArrayFromImage(self.needle_contour))
            self.vox_end_coord, self.vox_tip_coord = endpts_coord

            needle_array = sitk.GetArrayFromImage(self.needle_contour)
            needle_array = ndimage.binary_dilation(needle_array).astype(needle_array.dtype)

            # correct_endpoint_dilation does not necessarily help in this context since we just 
            # need to fit a spline here but I just keep it here in case one could use it.
            correct_for_dilation = False
            if correct_for_dilation:
                # Making sure dilating did not change endpoints
                needle_array = self.correct_endpoint_dilation(needle_array)

            dilated_needle_contour = sitk.GetImageFromArray(needle_array)
            dilated_needle_contour.CopyInformation(self.needle_contour)
            self.needle_contour = dilated_needle_contour

        self.rotate()
    
    def vox_from_coord(self, point, raw_input:bool=False):
        if raw_input:
            return list(self.input_catheter_contour.TransformPhysicalPointToIndex(point))
        return list(self.needle_contour.TransformPhysicalPointToIndex(point))
    
    def coord_from_vox(self, voxel_idx, raw_input:bool=False):
        if raw_input:
            return self.input_catheter_contour.TransformIndexToPhysicalPoint(voxel_idx)
        return self.needle_contour.TransformIndexToPhysicalPoint(voxel_idx)
    
    def get_endpoint_line(self, volume:np.ndarray=None, from_raw_volume:bool=False):
        points = np.argwhere(volume != 0)
        positions = np.array(
            # TransformIndexToPhysicalPoint only takes int
            [self.coord_from_vox([int(idx) for idx in pt], raw_input=from_raw_volume) for pt in points]
        )
        endpoints, max_dist = find_extremal_points_a(positions)
        endpoints = endpoints[0]
        endpoints_coord = [np.array(self.vox_from_coord(pt, raw_input=from_raw_volume)) for pt in endpoints]
        return endpoints_coord, points, positions

    def correct_endpoint_dilation(
            self, volume:np.ndarray, remove_similar_distance_dilated_pt:bool=True, 
            endpt1_vox_idx:List[float]=None, endpt2_vox_idx:List[float]=None, from_raw_volume:bool=False):

        if endpt1_vox_idx is None:
            assert endpt2_vox_idx is None, (
                """If endpt1 is None, that means you use the original endpoints of the curve: 
                self.vox_tip_coord and self.vox_end_coord. endpt2_vox_idx should be None"""
            )
            endpt1_vox_idx = self.vox_tip_coord
            endpt2_vox_idx = self.vox_end_coord
        
        # Getting the new endpoints from dilated line
        endpoints_coord, points, positions = self.get_endpoint_line(volume, from_raw_volume=from_raw_volume)
        while not (
            np.any(np.all(np.array(endpt1_vox_idx) == endpoints_coord, axis=1))
            and np.any(np.all(np.array(endpt2_vox_idx) == endpoints_coord, axis=1))
        ):
            for endpt in endpoints_coord:
                not_begining = np.any(endpt != endpt1_vox_idx)
                not_end = np.any(endpt != endpt2_vox_idx)
                if not_begining and not_end:
                    assert np.any(endpt != endpt1_vox_idx) and np.any(
                        endpt != endpt2_vox_idx
                    ), "This point should not be an endpoint"
                    volume[endpt[0], endpt[1], endpt[2]] = 0
                    for pt_idx, pt in enumerate(points):
                        if np.all(pt == endpt):
                            break

                    points = np.delete(points, pt_idx, axis=0)
                    positions = np.delete(positions, pt_idx, axis=0)

            endpoints, max_dist = find_extremal_points_a(positions)
            # If two points are at the same distance from an enpoint
            # find_etremal pioint function can return many pairs.
            # We take the first one since all are going to be processed
            # until we find the endpoints coord.
            endpoints = endpoints[0]
            assert len(endpoints) == 2, f"There should be two endpoints but you only have {endpoints}"
            endpoints_coord = [np.array(self.vox_from_coord(pt, raw_input=from_raw_volume)) for pt in endpoints]

        if remove_similar_distance_dilated_pt:
            endpoints_copy = endpoints_coord.copy()
            # if the dilated points are changing the endpoints (including tip),
            # we need to remove the points that are at the same distance from the tip.
            if len(endpoints_coord) > 2:
                while len(endpoints_coord) > 2:
                    for pt_idx, pt in enumerate(endpoints_copy):
                        is_tip = np.all(np.array(endpt1_vox_idx) == pt)
                        is_end = np.all(np.array(endpt2_vox_idx) == pt)
                        if not (is_tip or is_end):
                            assert np.any(pt != endpt1_vox_idx) and np.any(
                                pt != endpt2_vox_idx
                            ), "This point should not be an endpoint"
                            volume[pt[0], pt[1], pt[2]] = 0
                            break
                    endpoints_coord = np.delete(endpoints_coord, pt_idx, axis=0)

        return volume
    
    def rotate(self):
        """
        Rotate the catheter so that the direction is aligned with the z-axis.
        """
        # Fit line to the points
        self.points = get_physical_coord_for_needle(self.needle_contour)
        mean, direction = fit_line(self.points)

        # Rotate the catheter so that the direction is aligned with the z-axis
        self.rotation_matrix = calculate_rotation_matrix(direction)
        self.transform = create_rotation_transform(
            self.needle_contour, self.rotation_matrix
        )
        self.inverse_rotation = self.transform.GetInverse()
        self.original_catheter_pts_rotated = [self.transform.TransformPoint(pt) for pt in self.points]
        self.rotated_needle = rotate_volume(
            self.needle_contour,
            self.transform,
            interpolator=sitk.sitkNearestNeighbor,
        )

    def interpolate_spline(self):
        """
        Interpolate a spline through the central points of the catheter.
        """
        # Get the central points of the catheter
        self.central_points = np.array(self.get_central_points())
        self.tck, self.u = fit_spline(self.central_points, s=1000, k=3, nest=0)

    def get_central_points(self):
        """
        Get the central points of the catheter.
        """
        central_points = []
        self.original_central_points = []
        self.rotated_center_points = []
        # Numpy is depth, height, width and SimpleITK is width, height, depth
        rotated_needle_array = np.swapaxes(sitk.GetArrayFromImage(self.rotated_needle), 0, 2)
        
        for z in range(self.rotated_needle.GetSize()[2]):
            slice = rotated_needle_array[:, :, z]
            # Get the central points of the needle in the slice
            central_points_slice = self.get_central_points_from_slice(slice, z)

            if central_points_slice is not None:
                self.rotated_center_points.append(central_points_slice.tolist())
                central_points.append(central_points_slice)
                self.original_central_points.append(
                    self.inverse_rotation.TransformPoint(central_points_slice)
                )

        return central_points

    def get_central_points_from_slice(self, slice, slice_idx):
        """
        Get the central points of the catheter from a slice.
        """
        needle_coords = []
        # Get the needle points from the slice
        needle_pts_x, needle_pts_y = np.where(slice != 0)

        if len(needle_pts_x) == 0:
            return None
        else:
            for pt in zip(needle_pts_x, needle_pts_y):

                cordd_pt_3d = [
                    int(i)
                    for i in [
                        pt[0],
                        pt[1],
                        slice_idx,
                    ]
                ]
                physical_pos = self.rotated_needle.TransformIndexToPhysicalPoint(
                    cordd_pt_3d
                )
                needle_coords.append(physical_pos)

            # Get the central point of the slice
            central_point = np.mean(needle_coords, axis=0)
            return central_point

    def get_point_from_spline(self, t):
        """
        Get the points from the spline. Since the spline has been created in the rotated space, we need to
        transform the points back to the original space.
        """
        rotated_points = [float(ax_val) for ax_val in splev(t, self.tck)]
        pt = self.inverse_rotation.TransformPoint(rotated_points)
        return pt

    def get_spline_points(self):
        """
        Get the points from the spline.
        """
        points = []
        u_fine = np.linspace(0, 1, 100)
        for u in u_fine:
            pt = self.get_point_from_spline(u)
            points.append(pt)
        return points

    @staticmethod
    def derivative( t:float, tck:float):
        # Derivative of the spline
        dx, dy, dz = splev(t, tck, der=1)
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def distance_on_spline(self, point1:List[float], point2:List[float]):
        """ 
        Calculate the distance between two points on the spline.
        Takes into account the curvature of the spline.
        """
        _, t1, _ = self.project_on_spline(point1)
        _, t2, _ = self.project_on_spline(point2)
        arc_length, _ = quad(self.derivative, t1, t2, args=(self.tck,))
        return abs(arc_length)
    
    def step_in_spline(
        self, point, step, bound_min=0.0, bound_max=1.0, arc:bool=False
    ):
        """
        Given a point and a step, we want to find the point on the spline function
        that is at a distance step from the point.
        """
        def step_func(t: float) -> float:
            point_coords = self.get_point_from_spline(t)
            # Here we approximate the path on the spline to be a straight line
            # Error from arc length is not significant for small steps.
            if arc:
                return abs(self.distance_on_spline(point, point_coords) - step)
            else:
                return abs(distance(point, point_coords) - step)
        opt = minimize_scalar(
            step_func, bounds=(bound_min, bound_max), method="bounded"
        )
        return list(self.get_point_from_spline(opt.x)), opt.x
    
    def project_on_spline(
        self, point:List[float]
    ):
        """
        Given a point and a step, we want to find the point on the spline function
        that is at a distance step from the point.
        """
        def step_func(t: float) -> float:
            point_coords = self.get_point_from_spline(t)
            return abs(distance(point, point_coords))
        opt = minimize_scalar(
            step_func, bounds=(-0.5, 1.5), method="bounded"
        )
        return list(self.get_point_from_spline(opt.x)), opt.x, opt.fun
    

    def _get_points_in_segment(self, point1:List[float], point2:List[float], precision_sampling:float=0.1):
        """
        Creating points every precision_sampling mm along a spline between two points.
        """
        
        ### Create points along the spline
        # Getting ts boundaries for the steps
        _, point1_t, _ = self.project_on_spline(point1)
        _, point2_t, _ = self.project_on_spline(point2)

        # Identifying first t
        if point1_t > point2_t:
            start_t = point2_t
            start_pt = point2
            end_t = point1_t
            end_pt = point1
        else:
            start_t = point1_t
            start_pt = point1
            end_t = point2_t
            end_pt = point2

        points_on_spline = []
        ts_used = []
        points_on_spline.append(start_pt)
        ts_used.append(start_t)
        previous_pt = start_pt
        previous_t = start_t

        while previous_t < end_t:
            new_pt, new_t = self.step_in_spline(previous_pt, precision_sampling, bound_min=previous_t, bound_max=end_t, arc=True)
            points_on_spline.append(new_pt)
            ts_used.append(new_t)
            if self.distance_on_spline(new_pt, end_pt) <= precision_sampling:
                break
            previous_pt = new_pt
            previous_t = new_t

        points_on_spline.append(end_pt)
        ts_used.append(end_t)

        return points_on_spline, ts_used

    def create_spline_from_voxel_coordinates(
            self, point1:List[float], point2:List[float], precision_sampling:float=0.1, 
            diameter:float=2, correct_for_dilation:bool=True):
        """
        Creating the line in the array volume. Creating a line between two digitization points.
        For each point in the line, see if the distance of the line to any voxel coordinate falls
        in the radius of the needle. If yes, set the voxel to 1."""

        # Get all points in the segment
        points_coord_in_line, _ = self._get_points_in_segment(point1, point2, precision_sampling)

        voxel_spacing =  np.array(self.input_catheter_contour.GetSpacing()).astype(np.float32) 
        volume_array = np.swapaxes(sitk.GetArrayFromImage(self.input_catheter_contour), 2, 0)

        # Get all points coordinates of points close to the line
        # (no need to evaluate the full volume, expensive in memory)
        # We want a margin that makes sure we will find all points potnetially
        # in the radius of the needle.
        radius = diameter / 2
        # Side note: for this operation, you better have spacing as float32
        margin_x = int(np.ceil(radius / voxel_spacing[0]))
        margin_y = int(np.ceil(radius / voxel_spacing[1]))
        margin_z = int(np.ceil(radius / voxel_spacing[2]))

        # Create list of points to evaluate distance to.
        # We do it in a separate step since there can be duplicates,
        # and we don't want to do unnecessary distance computations.
        points_to_evaluate_indexes = np.empty((0, 3), dtype=int)
        for pt_in_l in points_coord_in_line:
            pt_vox = self.vox_from_coord(pt_in_l, raw_input=True)
            xs_to_evaluate = np.arange(
                max(pt_vox[0] - margin_x, 0),
                min(pt_vox[0] + margin_x, volume_array.shape[0] - 1) + 1,
            )
            assert len(xs_to_evaluate) > 0, "xs_to_evaluate should not be empty"
            ys_to_evaluate = np.arange(
                max(pt_vox[1] - margin_y, 0),
                min(pt_vox[1] + margin_y, volume_array.shape[1] - 1) + 1,
            )
            zs_to_evaluate = np.arange(
                max(pt_vox[2] - margin_z, 0),
                min(pt_vox[2] + margin_z, volume_array.shape[2] - 1) + 1,
            )
            new_pt_indexes = np.array(
                np.meshgrid(xs_to_evaluate, ys_to_evaluate, zs_to_evaluate)
            ).T.reshape(-1, 3)
            # Check if new_point is already in points_to_evaluate_indexes
            for newpt in new_pt_indexes:
                if not np.any(np.all(points_to_evaluate_indexes == newpt, axis=1)):
                    # If new_point is not in the array, append it
                    points_to_evaluate_indexes = np.vstack(
                        (points_to_evaluate_indexes, newpt)
                    )

        points_to_evaluate_coords = np.array(
            [
                self.coord_from_vox([int(idx) for idx in pt], raw_input=True)
                for pt in points_to_evaluate_indexes
            ]
        )
        # Compute distance between points to evaluate and points in the line
        distances = cdist(points_to_evaluate_coords, points_coord_in_line)


        added_spline_volume = np.zeros_like(volume_array)
        # Determine points belonging to catheter
        pts_belonging_to_catheter = np.unique(np.where(distances < radius)[0])
        for pt_idx in pts_belonging_to_catheter:
            pt_vox = points_to_evaluate_indexes[pt_idx]
            added_spline_volume[pt_vox[0], pt_vox[1], pt_vox[2]] = 1

        # The fact that we fill the catheter volume around a radius should affect
        # only the shaft and not the endpoints. We correct for this by removing
        # the points that would modify the endpoints of the catheter.
        if correct_for_dilation:
            added_spline_volume = self.correct_endpoint_dilation(volume=added_spline_volume, 
                endpt1_vox_idx=self.vox_from_coord(point1, raw_input=True), 
                endpt2_vox_idx=self.vox_from_coord(point2, raw_input=True), 
                from_raw_volume=True)
        
        return added_spline_volume

    def dilate(self, volume:np.ndarray, point1:List[float], 
               point2:List[float], dilation_nb_times:int=1, 
               from_raw_volume:bool=False):
        volume = ndimage.binary_dilation(volume, iterations=dilation_nb_times).astype(volume.dtype)
        volume = self.correct_endpoint_dilation(volume=volume, 
                endpt1_vox_idx=self.vox_from_coord(point1, raw_input=True), 
                endpt2_vox_idx=self.vox_from_coord(point2, raw_input=True), 
                from_raw_volume=from_raw_volume)
        return volume
    

def mean_nb_points_in_catheter(label_path:str, dicom_data_path:str):
    """
    This function is intended to have a mean number of points in the catheter
    from our training set in order to set a smoothing parameter for the spline 
    creation.
    """

    nb_points_in_catheter = 0
    total_nb_catheters = 0
    mask_paths = glob.glob(os.path.join(label_path, "*.nrrd"))
    for patient_mask in tqdm.tqdm(mask_paths, total=len(mask_paths), desc="Calculating mean number of points in catheter"):
        case_nb = os.path.basename(patient_mask).split(".")[0].split("_")[1]
        case_setup = CatheterSetUp(os.path.join(dicom_data_path, case_nb), setup=False)
        nb_catheters = case_setup.get_nb_catheters()
        total_nb_catheters += nb_catheters
        voxels_belonging_to_cathters = np.sum(sitk.GetArrayFromImage(sitk.ReadImage(patient_mask)) != 0)
        nb_points_in_catheter += voxels_belonging_to_cathters
    print("In your dataset you have {} catheters and {} points in the catheters".format(total_nb_catheters, nb_points_in_catheter))
    return nb_points_in_catheter / total_nb_catheters



if __name__ == "__main__":

    m = mean_nb_points_in_catheter(
        label_path="/home/sebquet/EngerLab/AI_Assisted_Brachytherapy/nnUNet_raw/Dataset007_catheters_and_tip_makers_consistent_diameter_2.0_dilation_1/labelsTr",
        dicom_data_path="/home/sebquet/EngerLab/Data/export_seb/patients")
    print("M for smoothing parameters s in fit splprep is: ", m)
    exit()
    import matplotlib
    matplotlib.use('tkAgg')
    import matplotlib.pyplot as plt

    from catheter.contour.creator import CatheterContourCreator
    from catheter.digitization.contour_digitizer import DwellPositionCreator

    from catheter.catheter_setup import CatheterSetUp

    patient_path = "/home/sebquet/EngerLab/Data/Hamed_breastCancer_patient/"
    patient_id = "1167439"
    patient_path = f"/home/sebquet/EngerLab/Data/patient_seb/{patient_id}/"

    output_folder = "/home/sebquet/EngerLab/tests/spline_interpolation/"
    os.makedirs(output_folder, exist_ok=True)
    sitk_needles_contour_path = os.path.join("/home/sebquet/EngerLab/AI_Assisted_Brachytherapy/ai_pipeline_results/Dataset004_catheters_and_tip_markers/1167439/ai_generated_catheters.seg.nrrd")
    patient_volume_path = os.path.join(patient_path, "processed", "CT.nrrd")
    if not os.path.exists(sitk_needles_contour_path):
        creator = CatheterContourCreator(
            patient_path, dilation=1
        )
        catheter_contour = creator.create_catheter_contour(write=True)
    sitk.WriteImage(
        sitk.ReadImage(patient_volume_path),
        os.path.join(output_folder, "CT.nrrd"),
        useCompression=True,
    )
    sitk.WriteImage(
        sitk.ReadImage(sitk_needles_contour_path),
        os.path.join(output_folder, "catheters.seg.nrrd"),
        useCompression=True,
    )
    patient_dwells = CatheterSetUp(patient_path)
    dwellpos_dict = patient_dwells.get_dwell_positions_list()
    
    patient_plan = CatheterSetUp(patient_path)
    dwell_pos_creator = DwellPositionCreator(
        sitk_needles_contour_path,
        fit_function="spline",
        # CatheterEvaluator takes consistent tip at the most distal part
        # of tip marker now => tip_distal always True.
        tip_distal=True
    )

    first_needle = dwell_pos_creator.separate_catheters(sitk.ReadImage(sitk_needles_contour_path))[
        9
    ]
    needle_spline_creator = NeedleSplineCreator(first_needle, dilate=False)
    rotated_ct = rotate_volume(
        sitk.ReadImage(patient_volume_path),
        needle_spline_creator.transform
    )
    sitk.WriteImage(
        needle_spline_creator.rotated_needle,
        os.path.join(output_folder, "rotated_needle_copilot.seg.nrrd"),
        useCompression=True,
    )
    sitk.WriteImage(
        rotated_ct,
        os.path.join(output_folder,"rotated_ct_copilot.nrrd"),
        useCompression=True,
    )

    needle_spline_creator.interpolate_spline()
    spline_points = needle_spline_creator.get_spline_points()

    create_slicer_markup_points(
        os.path.join(output_folder, "center_points.mrk.json"),
        needle_spline_creator.original_central_points,
        color = [0.5000076295109484, 0.5000076295109484, 0.5000076295109484]
    )
    create_slicer_markup_points(
        os.path.join(output_folder, "rotated_center_points.mrk.json"),
        needle_spline_creator.rotated_center_points,
    )

    x_spline_pts, y_spline_pts, z_spline_pts = list_to_x_y_z(spline_points)
    x_contour_pts, y_contour_pts, z_contour_pts = list_to_x_y_z(
        needle_spline_creator.points
    )
    x_center_pts, y_center_pts, z_center_pts = list_to_x_y_z(
        needle_spline_creator.original_central_points
    )

    print("spline_points", spline_points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        x_spline_pts,
        y_spline_pts,
        z_spline_pts,
        "b-",
        label="Fitted Curve",
    )
    ax.plot(
        x_contour_pts,
        y_contour_pts,
        z_contour_pts,
        "go",
        label="Points from contoured needles",
    )
    ax.plot(
        x_center_pts,
        y_center_pts,
        z_center_pts,
        "yo",
        label="Center points",
    )

    ax.legend()
    plt.show()
