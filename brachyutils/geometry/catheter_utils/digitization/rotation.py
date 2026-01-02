import itertools

import SimpleITK as sitk
import numpy as np


def calculate_rotation_matrix(direction, target=np.array([0, 0, 1]), prt:bool=False):
    """
    Calculate the rotation matrix to align the direction vector with the target vector.
    """
    direction = direction / np.linalg.norm(direction)
    target = target / np.linalg.norm(target)
    axis = np.cross(direction, target)
    angle = np.arccos(np.dot(direction, target))
    if prt:
        print(f"Direction: {direction}, Target: {target}, Axis: {axis}, Angle: {angle}, in degrees {np.rad2deg(angle)}")
    if np.linalg.norm(axis) == 0:  # direction and target are parallel
        return np.eye(3)

    axis = axis / np.linalg.norm(axis)
    axis_skewed = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation_matrix = (
        np.eye(3)
        + np.sin(angle) * axis_skewed
        + (1 - np.cos(angle)) * np.dot(axis_skewed, axis_skewed)
    )
    return rotation_matrix

def create_rotation_transform(volume, rotation_matrix:np.ndarray):
    """
    Rotate the volume using the given rotation matrix.
    """
    if not isinstance(rotation_matrix, np.ndarray):
        try:
            rotation_matrix = np.array(rotation_matrix)
        except Exception as e:
            raise ValueError(f"Invalid rotation matrix: {rotation_matrix}") from e
        
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be a 3x3 matrix, got shape {rotation_matrix.shape}")
    
    # Convert the rotation matrix to a SimpleITK Transform
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(rotation_matrix.ravel())

    # Calculate the center of the volume
    size = volume.GetSize()
    center = volume.TransformIndexToPhysicalPoint(
        [size[0] // 2, size[1] // 2, size[2] // 2]
    )
    transform.SetCenter(center)

    return transform

def rotate_volume(volume, transform:sitk.Transform, interpolator=sitk.sitkLinear):
        
    """
    Rotate the volume using the rotation matrix.
    This deprecated version of the code was not working properly. The extent of the rotated volume 
    was the same as the input one even if it should have been bigger. 
    '''
    # Resample the volume with the transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume)
    resampler.SetTransform(self.transform)
    resampler.SetInterpolator(interpolator)
    rotated_volume = resampler.Execute(volume)
    return rotated_volume
    '''
    """

    # compute the resampling grid for the transformed image
    max_indexes = [sz-1 for sz in volume.GetSize()]
    extreme_indexes = list(itertools.product(*(list(zip([0]*volume.GetDimension(),max_indexes)))))
    extreme_points_transformed = [transform.TransformPoint(volume.TransformContinuousIndexToPhysicalPoint(p)) for p in extreme_indexes]

    output_min_coordinates = np.min(extreme_points_transformed, axis=0)
    output_max_coordinates = np.max(extreme_points_transformed, axis=0)
    
    # isotropic ouput spacing
    output_spacing = [min(volume.GetSpacing())]*volume.GetDimension()  
                    
    output_origin = output_min_coordinates
    output_size = [int(((omx-omn)/ospc)+0.5)  for ospc, omn, omx in zip(output_spacing, output_min_coordinates, output_max_coordinates)]
    
    output_direction = [1,0,0,0,1,0,0,0,1]
    output_pixeltype = volume.GetPixelIDValue()

    return sitk.Resample(volume, 
                        output_size, 
                        transform.GetInverse(), 
                        interpolator, 
                        output_origin,
                        output_spacing,
                        output_direction,
                        0.0,
                        output_pixeltype)