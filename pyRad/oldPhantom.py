"""
Phantom module.

Copyright Marc-Andre Renaud, 2017
"""
import numpy

from scipy.interpolate import RectBivariateSpline

from pyRad.CoordinateSystem import CoordinateSystem


VLINAC_MATS = {
    "G4_AIR": 0,
    "G4_WATER": 1,
    "G4_LUNG_ICRP": 2,
    "G4_B-100_BONE": 3,
    "G4_BONE_COMPACT_ICRU": 4,
    "G4_BONE_CORTICAL_ICRP": 5,
    "G4_BRAIN_ICRP": 6,
    "G4_EYE_LENS_ICRP": 7,
    "G4_MUSCLE_SKELETAL_ICRP": 8,
    "G4_MUSCLE_STRIATED_ICRU": 9,
    "G4_SKIN_ICRP": 10,
    "G4_TISSUE_SOFT_ICRP": 11,
    "G4_Pb": 12,
    "G4_BLOOD_ICRP": 13,
    "G4_EYE_LENS_ICRP": 14,
    "G4_TESTIS_ICRP": 15,
    "G4_TISSUE_SOFT_ICRU-4": 16,
    "G4_ADIPOSE_TISSUE_ICRP": 17,
    "G4_A-150_TISSUE": 18,
    "G4_MS20_TISSUE": 19,
    "G4_MUSCLE_WITH_SUCROSE": 20,
    "G4_MUSCLE_WITHOUT_SUCROSE": 21,
    "G4_PLEXIGLASS": 22,
    "G4_Ti": 23
}


class Phantom(object):
    """
    Create a phantom from an image dataset.

    Attributes:
    ct (CT Object): referenced CT for phantom creation.
    structure_set (StructureSet object): referenced structure set for phantom creation.
    params (dict): parameters for phantom creation.

    methods:
    generate_phantom: creates phantom material and density matrices from ruleset
    as_dict: returns phantom metadata as a dictionary

    """

    def __init__(self, attrs=None):
        if attrs is not None:
            for k, v in attrs.items():
                setattr(self, k, v)

        if hasattr(self.ct, "orientation"):
            x_orient = int(self.ct.orientation[0])
            y_orient = int(self.ct.orientation[4])
            z_orient = 1
            self.orient = numpy.array([x_orient, y_orient, z_orient])
        else:
            self.orient = numpy.array([1.0, 1.0, 1.0])

        self.default_density = 1.0
        self.coords = self._init_coords()
        self.pos = self._init_pos()

        self.unique_mats = self._get_unique_mats(self.params["rules"])
        self.ruleset = self._preprocess_rules(self.params["rules"])

        self._init_matrices()

    def _init_coords(self):
        coords = {"ct": self.ct.coords, "phantom": {}}

        ph_spacing = numpy.array(self.params["spacing"])

        if self.params["use_dose_coords"]:
            # Can create a phantom based on the same coordinate system as
            # another phantom or dose distribution.
            ph_origin = numpy.array(self.params["ref_dose"]["img_pos"])
            ph_spacing = numpy.array(self.params["ref_dose"]["spacing"])
            ph_voxels = numpy.array(self.params["ref_dose"]["num_voxels"])
        elif self.params["use_bounding_box"]:
            bbox_struct = self.structure_set.get_roi_object(self.params["bounding_struct"])

            if "crop_to_slices" in self.params and self.params["crop_to_slices"] is True:
                first_slice = self.params["ct_crop"][0]
                last_slice = self.params["ct_crop"][1]

                z_first = coords["ct"].slice_coordinates[first_slice]
                z_last = coords["ct"].slice_coordinates[last_slice]

                bbox = bbox_struct.get_bounding_box(z_first, z_last)
            else:
                bbox = bbox_struct.get_bounding_box()

            ph_origin = numpy.array(bbox["min"])
            ph_end = numpy.array(bbox["max"])

            for i in range(3):
                if self.orient[i] < 0:
                    real_origin = ph_end[i]
                    real_end = ph_origin[i]
                    ph_origin[i] = real_origin
                    ph_end[i] = real_end

            # Calculate number of voxels by taking the extent of the phantom in each direction
            # and dividing by the voxel size.
            ph_voxels = numpy.array(numpy.abs(ph_origin - (ph_end + ph_spacing * self.orient)) / ph_spacing, dtype=numpy.int32)
        else:
            ph_origin = numpy.array(coords["ct"].img_pos)
            ct_voxels = numpy.array(coords["ct"].num_voxels)
            ct_spacing = numpy.array(coords["ct"].spacing)
            ph_end = (ph_origin + (ct_voxels - 1) * ct_spacing * self.orient)
            ph_end[2] = self.ct.slice_coordinates[-1]

            if "crop_to_slices" in self.params and self.params["crop_to_slices"] is True:
                first_slice = self.params["ct_crop"][0]
                last_slice = self.params["ct_crop"][1]

                z_first = coords["ct"].slice_coordinates[first_slice]
                z_last = coords["ct"].slice_coordinates[last_slice]
                ph_origin[2] = max(z_first, ph_origin[2])
                ph_end[2] = min(z_last, ph_end[2])


            ph_voxels = numpy.array(numpy.abs(ph_origin - (ph_end + ph_spacing * self.orient)) / ph_spacing, dtype=numpy.int32)

        ph_coords = CoordinateSystem({
            "img_pos": ph_origin,
            "spacing": ph_spacing,
            "orient": self.orient,
            "num_voxels": ph_voxels
        })

        coords["phantom"] = ph_coords

        return coords

    def _init_pos(self):
        pos = {"ct": {}, "phantom": {}}

        for img_type in ["ct", "phantom"]:
            topleft = self.coords[img_type].img_pos
            num_voxels = self.coords[img_type].num_voxels
            spacing = self.coords[img_type].spacing

            pos[img_type]["x_pixels"] = (topleft[0] +
                                         (numpy.arange(num_voxels[0]) * spacing[0] * self.orient[0]))
            pos[img_type]["y_pixels"] = (topleft[1] +
                                         (numpy.arange(num_voxels[1]) * spacing[1] * self.orient[1]))
            pos[img_type]["z_pixels"] = (topleft[2] +
                                         (numpy.arange(num_voxels[2]) * spacing[2] * self.orient[2]))

            x_grid, y_grid = numpy.meshgrid(pos[img_type]["x_pixels"],
                                            pos[img_type]["y_pixels"])
            position_flat = numpy.array(zip(x_grid.flatten(), y_grid.flatten()))
            pos[img_type]["flat"] = position_flat

        # Allow variable slice thickness for CT. Not supported for Phantom.
        pos["ct"]["z_pixels"] = self.ct.slice_coordinates

        return pos

    def _init_matrices(self):
        ph_voxels = self.coords["phantom"].num_voxels
        self.ct_numbers = numpy.zeros((ph_voxels[2], ph_voxels[1], ph_voxels[0]), dtype=numpy.float)

        self.mat_matrix = numpy.ones((ph_voxels[2], ph_voxels[1], ph_voxels[0]), dtype=numpy.int8)
        self.density_matrix = numpy.zeros((ph_voxels[2], ph_voxels[1], ph_voxels[0]), dtype=numpy.float)

        # Flattened matrices
        #self.mat_matrix = numpy.ones((ph_voxels[2] * ph_voxels[1] * ph_voxels[0]), dtype=numpy.int8)
        #self.density_matrix = numpy.zeros((ph_voxels[2] * ph_voxels[1] * ph_voxels[0]), dtype=numpy.float)

    def _set_hu_densities(self):
        hu_values = [x["hu"] for x in self.params["ct_calibration"]["segments"]]
        density_values = [y["density"] for y in self.params["ct_calibration"]["segments"]]
        return numpy.interp(self.ct_numbers, hu_values, density_values)

    def _get_voxel_boundaries(self):
        topleft = self.coords["phantom"].img_pos
        num_voxels = self.coords["phantom"].num_voxels
        spacing = self.coords["phantom"].spacing

        x_voxels = numpy.arange(num_voxels[0] + 1) * spacing[0] * self.orient[0] + (topleft[0] - 0.5 * spacing[0] * self.orient[0])
        y_voxels = numpy.arange(num_voxels[1] + 1) * spacing[1] * self.orient[1] + (topleft[1] - 0.5 * spacing[1] * self.orient[1])
        z_voxels = numpy.arange(num_voxels[2] + 1) * spacing[2] * self.orient[2] + (topleft[2] - 0.5 * spacing[2] * self.orient[2])

        return {
            "x": x_voxels,
            "y": y_voxels,
            "z": z_voxels
        }

    def generate_phantom(self):
        self._make_ct_number_grid()

        use_calibration = self.params.get("use_calibration", False)
        if use_calibration:
            default_densities = self._set_hu_densities()
        else:
            default_densities = numpy.full(self.density_matrix.shape, self.default_density)

        for rule in self.ruleset:
            if rule["roi"] != "Default":
                roi_mask = rule["mask"]
                self.mat_matrix[roi_mask] = rule["material"]
                if not use_calibration or rule["override_density"]:
                    self.density_matrix[roi_mask] = rule["density"]
                else:
                    self.density_matrix[roi_mask] = default_densities[roi_mask]

                if len(rule["HURules"]) > 0:
                    bool_mask = numpy.zeros(self.mat_matrix.shape, dtype=numpy.bool)
                    bool_mask[roi_mask] = 1
                    for HURule in rule["HURules"]:
                        lower_ct = self.ct_numbers >= HURule["HULow"]
                        upper_ct = self.ct_numbers < HURule["HUHigh"]
                        full_mask = bool_mask & (lower_ct & upper_ct)
                        self.mat_matrix[full_mask] = HURule["material"]
                        if not use_calibration or HURule["override_density"]:
                            self.density_matrix[full_mask] = HURule["density"]
            else:
                self.mat_matrix.fill(rule["material"])
                if not use_calibration or rule["override_density"]:
                    self.density_matrix.fill(rule["density"])
                else:
                    self.density_matrix = default_densities

                for HURule in rule["HURules"]:
                    lower_ct = self.ct_numbers >= HURule["HULow"]
                    upper_ct = self.ct_numbers < HURule["HUHigh"]
                    mask = lower_ct & upper_ct
                    self.mat_matrix[mask] = HURule["material"]
                    if not use_calibration or HURule["override_density"]:
                        self.density_matrix[mask] = HURule["density"]

        if numpy.any(self.density_matrix < 0):
            raise Exception("Negative densities in density matrix")

        if self.params.get("zero_air_density", False):
            self._zero_air_density(defaultDensity=False)

        num_voxels = self.coords["phantom"].num_voxels
        self.density_matrix = self.density_matrix.reshape((num_voxels[2], num_voxels[1], num_voxels[0]))
        self.mat_matrix = self.mat_matrix.reshape((num_voxels[2], num_voxels[1], num_voxels[0]))

    def _zero_air_density(self, defaultDensity=False):
        air_mats = [index for index, mat in enumerate(self.unique_mats) if "air" in mat["name"].lower()]
        air_index = -1
        if len(air_mats):
            air_index = air_mats[0]
        else:
            return

        density = 0.0
        if defaultDensity:
            density = self.unique_mats[air_index]["density"]

        if self.params["sim_program"] == "VirtuaLinac":
            mat_index = VLINAC_MATS[self.unique_mats[air_index]["name"]]
        else:
            mat_index = air_index + 1

        self.density_matrix[self.mat_matrix == mat_index] = density

    def _get_unique_mats(self, ruleset):
        # In BEAMnrc, the first material in the phantom is set as the
        # material for the region outside the phantom. We want air.
        unique_mats = []

        if self.params["sim_program"] == "BeamNRC":

            materials = [rule["material"]["name"].lower() for rule in ruleset]
            air_index = -1
            for ind, mat in enumerate(materials):
                if "air" in mat:
                    air_index = ind
                    break

            if air_index > -1:
                unique_mats = [ruleset[air_index]["material"]]
            else:
                unique_mats = [{"name": "AIR700ICRU", "density": 0.0012048}]

        for rule in ruleset:
            if rule["material"] not in unique_mats:
                unique_mats.append(rule["material"])
            for HURule in rule["HURules"]:
                if HURule["material"] not in unique_mats:
                    unique_mats.append(HURule["material"])

        return unique_mats

    def _get_mat_index(self, rule):
        if self.params["sim_program"] == "VirtuaLinac":
            if rule["material"]["name"] in VLINAC_MATS:
                return VLINAC_MATS[rule["material"]["name"]]
            else:
                return -1
        else:
            return self.unique_mats.index(rule["material"]) + 1

    def _preprocess_rules(self, pre_ruleset):
        pre_ruleset.sort(key=lambda rule: rule["order"])
        post_ruleset = []

        for rule in pre_ruleset:
            processed_rule = {
                "roi": rule["roi"]["name"],
                "density": rule["material"]["density"],
                "override_density": rule.get("override_density", False),
                "HURules": []
            }

            processed_rule["material"] = self._get_mat_index(rule)

            if processed_rule["roi"] == "Default":
                self.default_density = processed_rule["density"]

            for HU_rule in rule["HURules"]:
                density = HU_rule["material"]["density"]
                processed_HU_rule = {
                    "material": self._get_mat_index(HU_rule),
                    "density": density,
                    "override_density": HU_rule.get("override_density", False),
                    "HULow": int(HU_rule["HULow"]),
                    "HUHigh": int(HU_rule["HUHigh"])
                }
                processed_rule["HURules"].append(processed_HU_rule)

            if rule["roi"]["name"] != "Default":
                struct_obj = self.structure_set.get_roi_object(rule["roi"]["name"])
                processed_rule["mask"] = struct_obj.get_mask(self.coords["phantom"])

            post_ruleset.append(processed_rule)

        return post_ruleset

    def _make_ct_number_grid(self):
        ct_pos = self.pos["ct"]
        ph_pos = self.pos["phantom"]
        ph_coords = self.coords["phantom"]
        ct_grid = self.ct.get_whole_grid()
        for i in range(self.coords["phantom"].num_voxels[2]):
            phantom_z = i * ph_coords.spacing[2] + ph_coords.img_pos[2]
            ct_slice = self.ct.slice_from_z(phantom_z)
            ct_slice = ct_grid[ct_slice]

            interp = RectBivariateSpline(ct_pos["y_pixels"][::self.orient[1]], ct_pos["x_pixels"][::self.orient[0]], ct_slice[:, ::self.orient[0]], kx=1, ky=1)
            self.ct_numbers[i] = interp(ph_pos["y_pixels"][::self.orient[1]], ph_pos["x_pixels"][::self.orient[0]])[:, ::self.orient[0]]
            #self.ct_numbers[i] = self.ct_numbers[i].reshape((ph_coords.num_voxels[1], ph_coords.num_voxels[0]))

        #self.ct_numbers = self.ct_numbers.flatten()

    def convert_to_webgl_format(self):
        webgl_phantom = {}
        webgl_phantom["img_pos"] = self.coords["phantom"].img_pos.tolist()
        webgl_phantom["num_voxels"] = self.coords["phantom"].num_voxels.tolist()
        webgl_phantom["spacing"] = self.coords["phantom"].spacing.tolist()

        webgl_phantom["max_norm"] = self.density_matrix.max()
        webgl_phantom["min_norm"] = self.density_matrix.min()
        webgl_phantom["num_mats"] = len(self.unique_mats)

        webgl_phantom["mat_list"] = [mat["name"] for mat in self.unique_mats]
        webgl_phantom["orient"] = self.orient.tolist()

        webgl_phantom["mat_matrix"] = {}
        webgl_phantom["density_matrix"] = {}

        for i in range(self.mat_matrix.shape[0]):
            webgl_phantom["mat_matrix"][i] = self.mat_matrix[i].flatten().tolist()
            webgl_phantom["density_matrix"][i] = self.density_matrix[i].flatten().tolist()

        return webgl_phantom

    def write_phantom(self, filename):
        if self.params["sim_program"] == "VirtuaLinac":
            return self._write_vlinac(filename)
        else:
            return self._write_egsphant(filename)

    def _write_egsphant(self, filename):
        num_materials = len(self.unique_mats)

        with open(filename, "w") as phantom_file:
            phantom_file.write("%i\n" % num_materials)
            for mat in self.unique_mats:
                phantom_file.write(mat["name"] + "\n")
            phantom_file.write(" ".join([str(0)] * num_materials) + "\n")

            messed_up_phantom_codes = ["brachydose", "cyberknife", "BeamNRC"]

            if self.params["sim_program"] in messed_up_phantom_codes:
                # EGS parses text files weirdly. It wasn't running unless
                # I put two spaces in front of the number of voxels.
                phantom_file.write("  %i  %i  %i\n" % tuple(self.coords["phantom"].num_voxels))
            else:
                phantom_file.write("%i %i %i\n" % tuple(self.coords["phantom"].num_voxels))

            boundaries = self._get_voxel_boundaries()
            phantom_file.write(" ".join([str(round(x, 3)) for x in boundaries["x"][::self.orient[0]] / 10.0]) + "\n")
            phantom_file.write(" ".join([str(round(y, 3)) for y in boundaries["y"][::self.orient[1]] / 10.0]) + "\n")
            phantom_file.write(" ".join([str(round(z, 3)) for z in boundaries["z"][::self.orient[2]] / 10.0]) + "\n")

            for grid_slice in self.mat_matrix.astype(numpy.dtype("a2")):
                for row in grid_slice[::self.orient[1], ::self.orient[0]]:
                    phantom_file.write("".join(row) + "\n")
                phantom_file.write("\n")

            for grid_slice in self.density_matrix.astype(numpy.dtype("a6")):
                for row in grid_slice[::self.orient[1], ::self.orient[0]]:
                    phantom_file.write(" ".join(row) + "\n")
                phantom_file.write("\n")

    def _write_vlinac(self, filename):
        voxels = [self.coords["phantom"].num_voxels[0],
                  self.coords["phantom"].num_voxels[2],
                  self.coords["phantom"].num_voxels[1]]

        vox_shape = [self.coords["phantom"].spacing[0],
                    self.coords["phantom"].spacing[2],
                    self.coords["phantom"].spacing[1]]

        mats = self.mat_matrix[:, ::self.orient[1], ::self.orient[0]]
        densities = self.density_matrix[:, ::self.orient[1], ::self.orient[0]]

        mats = numpy.rollaxis(mats, 0, 1)
        mats = numpy.rollaxis(mats, 2, 0)
        densities = numpy.rollaxis(densities, 0, 1)
        densities = numpy.rollaxis(densities, 2, 0)

        with open(filename, "w") as phantom_file:
            phantom_file.write("# VirtuaLinac phantom definition file\n")
            phantom_file.write("# The next 2 lines are required by VirtuaLinac:\n")
            phantom_file.write("# Number of voxels: %i %i %i\n" % tuple(voxels))
            phantom_file.write("# Voxel size: %.2f %.2f %.2f mm\n" % tuple(vox_size))

            for i in range(voxels[0]):
                for j in range(voxels[1]):
                    for k in range(voxels[2]):
                        phantom_file.write("%i %f\n" % (mats[i][j][k], densities[i][j][k]))

    @staticmethod
    def read_egsphant_header(filename):
        """Populate phantom metadata dict from header of egsphant file."""
        phantom = {}

        with open(filename, "r") as phantom_file:
            num_materials = int(phantom_file.readline().strip())
            mat_list = []
            for i in range(num_materials):
                mat_list.append(phantom_file.readline().strip())

            # dummy line
            phantom_file.readline()

            num_voxels = phantom_file.readline().strip().split()
            x_voxels = [float(x) * 10 for x in phantom_file.readline().strip().split()]
            y_voxels = [float(y) * 10 for y in phantom_file.readline().strip().split()]
            z_voxels = [float(z) * 10 for z in phantom_file.readline().strip().split()]

            x_spacing = abs(x_voxels[1] - x_voxels[0])
            y_spacing = abs(y_voxels[1] - y_voxels[0])
            z_spacing = abs(z_voxels[1] - z_voxels[0])

            phantom["mat_list"] = mat_list
            phantom["num_voxels"] = num_voxels
            phantom["topleft"] = [x_voxels[0], y_voxels[0], z_voxels[0]]
            phantom["spacing"] = [x_spacing, y_spacing, z_spacing]

        return phantom
