"""
CoordinateSystem module.

Copyright Marc-Andre Renaud, 2017
"""
import numpy


class CoordinateSystem(object):
    """
    pyRad wrapper for a coordinate system.

    Attributes:
    img_pos (list): x, y, z coordinates of the middle of the first pixel in image.
    spacing (list): x, y, z spacing between voxels.
    num_voxels (list): number of voxels in x, y, z directions
    orient (list): x, y, z orientation vector
    slice_coordinates (list): sorted z values of each slice in case of variable slice thickness
    slice_flipped (bool): DICOM doses that are feet first have Z coordinates flipped around...

    Methods:
    get_voxel_positions: Returns the (x,y) voxel position of every voxel in a slice.

    get_phantom_voxel_positions: Returns the (x,y) position of every voxel after
        reverting to head-first supine position

    slice_from_z: Returns the slice number closest to the "z" coordinate provided.

    """

    def __init__(self, attrs):
        self.img_pos = numpy.array(attrs["img_pos"])
        self.spacing = numpy.array(attrs["spacing"])

        try:
            self.orient = numpy.array(attrs["orient"], dtype=int)
        except KeyError:
            self.orient = numpy.array(attrs["orientation"], dtype=int)

        if len(self.orient) == 6:
            self.orient = numpy.array([self.orient[0], self.orient[4], 1])

        self.num_voxels = numpy.array(attrs["num_voxels"], dtype=int)

        self.slice_flipped = attrs.get("slice_flipped", False)
        self.slice_coordinates = attrs.get("slice_coordinates", None)
        self.slice_boundaries = attrs.get("slice_boundaries", None)

        if self.slice_coordinates is None:
            if not self.slice_flipped:
                self.slice_coordinates = numpy.arange(self.num_voxels[2]) * self.spacing[2] + self.img_pos[2]
            else:
                self.slice_coordinates = numpy.arange(self.num_voxels[2]) * self.spacing[2] * (self.orient[0] * self.orient[1]) + self.img_pos[2]
                self.slice_coordinates = self.slice_coordinates[::-1]

        slice_direction = int(self.orient[0] * self.orient[1]) if self.slice_flipped else 1
        slice_start = self.img_pos[2] - 0.5 * slice_direction * self.spacing[2]
        if self.slice_boundaries is None:
            self.slice_boundaries = numpy.arange(self.num_voxels[2] + 1) * self.spacing[2] * slice_direction + slice_start
            self.slice_boundaries = self.slice_boundaries[::slice_direction]

    def as_dict(self):
        return {
            "img_pos": list(self.img_pos),
            "spacing": list(self.spacing),
            "orient": list(self.orient),
            "num_voxels": list(self.num_voxels),
            "slice_flipped": self.slice_flipped,
            "slice_coordinates": list(self.slice_coordinates),
            "slice_boundaries": list(self.slice_boundaries)
        }

    def get_voxel_positions(self):
        x_positions = numpy.arange(self.num_voxels[0]) * self.spacing[0] * self.orient[0] + self.img_pos[0]
        y_positions = numpy.arange(self.num_voxels[1]) * self.spacing[1] * self.orient[1] + self.img_pos[1]

        position_grid = numpy.meshgrid(x_positions, y_positions)
        position_flat = numpy.array(zip(position_grid[0].flatten(), position_grid[1].flatten()))

        return position_flat

    def get_voxel_position_list(self):
        x_positions = numpy.arange(self.num_voxels[0]) * self.spacing[0] * self.orient[0] + self.img_pos[0]
        y_positions = numpy.arange(self.num_voxels[1]) * self.spacing[1] * self.orient[1] + self.img_pos[1]
        z_positions = numpy.arange(self.num_voxels[2]) * self.spacing[2] * self.orient[2] + self.img_pos[2]

        return (x_positions, y_positions, z_positions)

    def get_phantom_voxel_positions(self):
        x_positions = numpy.arange(self.num_voxels[0]) * self.spacing[0] * self.orient[0] + self.img_pos[0]
        y_positions = numpy.arange(self.num_voxels[1]) * self.spacing[1] * self.orient[1] + self.img_pos[1]

        position_grid = numpy.meshgrid(x_positions, y_positions)

        position_flat = numpy.array(zip(position_grid[0].flatten()[::self.orient[0]], position_grid[1].flatten()[::self.orient[1]]))
        return position_flat

    def get_voxel_bounds(self):
        x_bins = self.img_pos[0] - 0.5 * self.spacing[0] * self.orient[0] + numpy.arange(self.num_voxels[0]+1) * self.spacing[0] * self.orient[0]
        y_bins = self.img_pos[1] - 0.5 * self.spacing[1] * self.orient[1] + numpy.arange(self.num_voxels[1]+1) * self.spacing[1] * self.orient[1]
        z_bins = self.img_pos[2] - 0.5 * self.spacing[2] * self.orient[2] + numpy.arange(self.num_voxels[2]+1) * self.spacing[2] * self.orient[2]

        return (x_bins, y_bins, z_bins)

    def vox_from_point(self, point):
        x_voxel = int((point[0] - self.img_pos[0]) / self.spacing[0])
        y_voxel = int((point[1] - self.img_pos[1]) / self.spacing[1])
        z_voxel = int((point[2] - self.img_pos[2]) / self.spacing[2])

        return [abs(x_voxel), abs(y_voxel), abs(z_voxel)]

    def slice_from_z(self, z):
        if z < self.slice_boundaries[0] or z > self.slice_boundaries[-1]:
            return None

        slice_num = numpy.searchsorted(self.slice_coordinates, z)

        if slice_num >= len(self.slice_coordinates) - 1:
            slice_num = len(self.slice_coordinates) - 1
        elif slice_num == 0:
            if abs(z - self.slice_coordinates[slice_num]) > abs(self.slice_coordinates[slice_num + 1] - z):
                slice_num += 1
        else:
            possibilities = numpy.array([abs(z - self.slice_coordinates[slice_num-1]), abs(z - self.slice_coordinates[slice_num]), abs(z - self.slice_coordinates[slice_num+1])])
            slice_num = slice_num + (numpy.argmin(possibilities) - 1)

        if self.slice_flipped:
            slice_num = len(self.slice_coordinates) - 1 - slice_num

        return slice_num
