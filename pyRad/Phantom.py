import numpy as np

from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
from pyRad.CoordinateSystem import CoordinateSystem

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

    def __init__(self, attrs):
        self.ct = attrs["ct"]
        self.structure_set = attrs["structure_set"]
        self.params = attrs["params"]
        self.orient = np.array(self.ct.coords.orient, dtype=int)

        self.mat_names, self.mat_list = self._get_unique_mats(self.params["rules"])

        self.zoomed_grid = None
        self.coords = None
        self.phantom_mats = None
        self.phantom_densities = None
        self.default_density = 0.0

    def _get_coords(self):
        """
        Determine the coordinate system of the phantom. The coordinate
        system includes the voxel spacing, the origin of the phantom and the
        number of voxels in each direction.

        There are many different ways to create a phantom. One can either
        copy a referenced dose grid coordinate system, limit the phantom
        to the bounding box of a structure, or simply copy CT coordinates.
        """
        if self.params.get("use_dose_coords", False):
            return self._from_dose_coords(self.params["ref_dose"])

        ct_start = self.ct.coords.img_pos - 0.5 * self.ct.coords.spacing * self.ct.coords.orient

        if self.params.get("use_bounding_box", False):
            bounding_roi = self.structure_set.get_roi_object(self.params["bounding_struct"])
            bbox = bounding_roi.get_bounding_box()
            bound1 = bbox["min"]
            bound2 = bbox["max"]
        else:
            bound1 = ct_start
            bound2 = ct_start + self.ct.coords.spacing * (self.ct.coords.num_voxels + 1) * self.ct.coords.orient

        bbox = {
            "min": np.minimum(bound1, bound2),
            "max": np.maximum(bound1, bound2)
        }

        if self.params.get("crop_to_slices", False):
            z_first = self.ct.coords.slice_boundaries[self.params["ct_crop"][0]]
            z_last = self.ct.coords.slice_boundaries[self.params["ct_crop"][1]+1]

            bbox["min"][2] = max(bbox["min"][2], z_first)
            bbox["max"][2] = min(bbox["max"][2], z_last)

        spacing = np.array(self.params["spacing"])
        zoom = self.ct.coords.spacing / spacing
        ct_grid = self.ct.get_whole_grid()
        zoomed_grid = ndimage.zoom(ct_grid, (zoom[2], zoom[1], zoom[0]), order=1)

        phant_imgpos = ct_start + 0.5 * spacing * self.orient
        num_voxels = zoomed_grid.shape[::-1]

        x_pos = phant_imgpos[0] + np.arange(num_voxels[0]) * spacing[0] * self.orient[0]
        y_pos = phant_imgpos[1] + np.arange(num_voxels[1]) * spacing[1] * self.orient[1]
        z_pos = phant_imgpos[2] + np.arange(num_voxels[2]) * spacing[2] * self.orient[2]

        x_mask = (x_pos >= bbox["min"][0]) & (x_pos <= bbox["max"][0])
        y_mask = (y_pos >= bbox["min"][1]) & (y_pos <= bbox["max"][1])
        z_mask = (z_pos >= bbox["min"][2]) & (z_pos <= bbox["max"][2])

        x_indices = np.nonzero(x_mask)[0]
        y_indices = np.nonzero(y_mask)[0]
        z_indices = np.nonzero(z_mask)[0]
        x_min = x_indices[0]
        x_max = x_indices[-1] + 1

        y_min = y_indices[0]
        y_max = y_indices[-1] + 1

        z_min = z_indices[0]
        z_max = z_indices[-1] + 1

        zoomed_grid = zoomed_grid[z_min:z_max, y_min:y_max, x_min: x_max]
        img_pos = np.array([x_pos[x_mask][0], y_pos[y_mask][0], z_pos[z_mask][0]])

        coords = CoordinateSystem({"spacing": spacing,
                                   "img_pos": img_pos,
                                   "num_voxels": zoomed_grid.shape[::-1],
                                   "orient": self.orient})

        return (zoomed_grid, coords)


    def _from_dose_coords(self, ref_dose):
        dose_coords = CoordinateSystem(ref_dose)
        spacing = dose_coords.spacing
        zoom = self.ct.coords.spacing / spacing

        ct_grid = self.ct.get_whole_grid()
        ct_start = self.ct.coords.img_pos - 0.5 * self.ct.coords.spacing * self.ct.coords.orient

        zoomed_grid = ndimage.zoom(ct_grid, (zoom[2], zoom[1], zoom[0]), order=1)
        pad_zoomed_grid = np.pad(zoomed_grid, 1, mode="edge")

        phant_imgpos = ct_start + 0.5 * spacing * self.orient
        pad_imgpos = phant_imgpos - 0.5 * spacing * self.orient

        num_voxels = pad_zoomed_grid.shape[::-1]

        x_pos = pad_imgpos[0] + np.arange(num_voxels[0]) * spacing[0] * self.orient[0]
        y_pos = pad_imgpos[1] + np.arange(num_voxels[1]) * spacing[1] * self.orient[1]
        z_pos = pad_imgpos[2] + np.arange(num_voxels[2]) * spacing[2] * self.orient[2]

        d_voxels = dose_coords.num_voxels
        dose_x, dose_y, dose_z = dose_coords.get_voxel_position_list()

        # rip readability
        dose_z, dose_y, dose_x = np.broadcast_arrays(dose_z.reshape(-1, 1, 1), dose_y.reshape(1, -1, 1), dose_x)
        interp_coords = np.vstack((dose_z.flatten(), dose_y.flatten(), dose_x.flatten()))

        interp = RegularGridInterpolator((z_pos, y_pos, x_pos), pad_zoomed_grid, bounds_error=False)

        dose_ct_values = interp(interp_coords.T).reshape(d_voxels[2], d_voxels[1], d_voxels[0])
        #dose_ct_values = np.clip(dose_ct_values, -1024, 10000)

        return (dose_ct_values, dose_coords)

    def _get_unique_mats(self, ruleset):
        """
        Return a list of materials present in the phantom.
        """

        unique_mat_names = []
        unique_mats = []

        if self.params["sim_program"] == "BeamNRC":
            # In BeamNRC, the first material defined in the phantom file
            # also defines the material for the region outside the phantom.
            # We want that to be air.

            materials = [rule["material"]["name"].lower() for rule in ruleset]
            air_index = None
            for index, mat in enumerate(materials):
                if "air" in mat:
                    air_index = index
                    break

            if air_index is not None:
                unique_mats = [ruleset[air_index]["material"]]
            else:
                unique_mats = [{"name": "AIR700ICRU", "density": 0.0012048}]

            unique_mat_names.append(unique_mats[0]["name"])

        for rule in ruleset:
            if rule["material"]["name"] not in unique_mat_names:
                unique_mats.append(rule["material"])
                unique_mat_names.append(rule["material"]["name"])

            for hu_rule in rule["HURules"]:
                if hu_rule["material"]["name"] not in unique_mat_names:
                    unique_mats.append(hu_rule["material"])
                    unique_mat_names.append(hu_rule["material"]["name"])

        return (unique_mat_names, unique_mats)

    def _get_mat_index(self, rule):
        return self.mat_list.index(rule["material"]) + 1

    def _preprocess_rules(self, pre_ruleset, coords):
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

            for hu_rule in rule["HURules"]:
                density = hu_rule["material"]["density"]
                processed_hu_rule = {
                    "material": self._get_mat_index(hu_rule),
                    "density": density,
                    "override_density": hu_rule.get("override_density", False),
                    "HULow": int(hu_rule["HULow"]),
                    "HUHigh": int(hu_rule["HUHigh"])
                }
                processed_rule["HURules"].append(processed_hu_rule)

            if rule["roi"]["name"] != "Default":
                struct_obj = self.structure_set.get_roi_object(rule["roi"]["name"])
                processed_rule["mask"] = struct_obj.get_mask(coords)

            post_ruleset.append(processed_rule)

        return post_ruleset

    def _get_hu_densities(self, ct_grid):
        ct_densities = np.array(ct_grid, dtype=np.float)
        hu_values = [x["hu"] for x in self.params["ct_calibration"]["segments"]]
        density_values = [y["density"] for y in self.params["ct_calibration"]["segments"]]
        return np.interp(ct_densities, hu_values, density_values)

    def _zero_air_density(self, densities, mats, defaultDensity=False):
        air_mats = [idx for idx, mat in enumerate(self.mat_list) if "air" in mat["name"].lower()]
        try:
            air_index = air_mats[0]
        except IndexError:
            return

        density = 0.0
        if defaultDensity:
            density = self.mat_list[air_index]["density"]

        mat_index = air_index + 1

        densities[mats == mat_index] = density

    def _make_grids(self):
        """
        Make material and density grids based on the CT coordinate system and
        the rules provided by the user.
        """
        mat_matrix = np.ones(self.zoomed_grid.shape, dtype=np.int8)
        density_matrix = np.zeros(self.zoomed_grid.shape, dtype=np.float)

        rules = self._preprocess_rules(self.params["rules"], self.coords)

        use_calibration = self.params.get("use_calibration", False)
        if use_calibration:
            default_densities = self._get_hu_densities(self.zoomed_grid)
        else:
            default_densities = np.full(self.zoomed_grid.shape, self.default_density)

        for rule in rules:
            if rule["roi"] != "Default":
                roi_mask = rule["mask"]
                mat_matrix[roi_mask] = rule["material"]
                if not use_calibration or rule["override_density"]:
                    density_matrix[roi_mask] = rule["density"]
                else:
                    density_matrix[roi_mask] = default_densities[roi_mask]

                if rule["HURules"] > 0:
                    bool_mask = np.zeros(mat_matrix.shape, dtype=np.bool)
                    bool_mask[roi_mask] = True
                    for hu_rule in rule["HURules"]:
                        lower_ct = self.zoomed_grid >= hu_rule["HULow"]
                        upper_ct = self.zoomed_grid < hu_rule["HUHigh"]
                        full_mask = bool_mask & (lower_ct & upper_ct)
                        mat_matrix[full_mask] = hu_rule["material"]
                        if not use_calibration or hu_rule["override_density"]:
                            density_matrix[full_mask] = hu_rule["density"]
            else:
                mat_matrix.fill(rule["material"])
                if not use_calibration or rule["override_density"]:
                    density_matrix.fill(rule["density"])
                else:
                    density_matrix = default_densities

                for hu_rule in rule["HURules"]:
                    lower_ct = self.zoomed_grid >= hu_rule["HULow"]
                    upper_ct = self.zoomed_grid < hu_rule["HUHigh"]
                    mask = lower_ct & upper_ct
                    mat_matrix[mask] = hu_rule["material"]
                    if not use_calibration or hu_rule["override_density"]:
                        density_matrix[mask] = hu_rule["density"]

        if np.any(density_matrix < 0):
            raise Exception("Negative densities in density matrix")

        if self.params.get("zero_air_density", False):
            self._zero_air_density(density_matrix, mat_matrix, defaultDensity=False)

        return (mat_matrix, density_matrix)

    def _write_egsphant(self, filename):
        num_materials = len(self.mat_names)

        density_matrix = self.phantom_densities[:, ::self.orient[1], ::self.orient[0]]
        mat_matrix = self.phantom_mats[:, ::self.orient[1], ::self.orient[0]]

        with open(filename, "w") as phantom_file:
            phantom_file.write("%i\n" % num_materials)
            for mat in self.mat_names:
                phantom_file.write(mat + "\n")
            phantom_file.write(" ".join([str(0)] * num_materials) + "\n")

            messed_up_phantom_codes = ["BeamNRC"]

            if self.params["sim_program"] in messed_up_phantom_codes:
                phantom_file.write("  %i  %i  %i\n" % tuple(self.coords.num_voxels))
            else:
                phantom_file.write("%i %i %i\n" % tuple(self.coords.num_voxels))

            (x_bounds, y_bounds, z_bounds) = self.coords.get_voxel_bounds()
            phantom_file.write(" ".join([str(round(x, 5)) for x in x_bounds[::self.orient[0]] / 10.0]) + "\n")
            phantom_file.write(" ".join([str(round(y, 5)) for y in y_bounds[::self.orient[1]] / 10.0]) + "\n")
            phantom_file.write(" ".join([str(round(z, 5)) for z in z_bounds[::self.orient[2]] / 10.0]) + "\n")

            for grid_slice in mat_matrix.astype(np.dtype("a2")):
                for row in grid_slice:
                    phantom_file.write("".join(row) + "\n")
                phantom_file.write("\n")

            for grid_slice in density_matrix.astype(np.dtype("a6")):
                for row in grid_slice:
                    #phantom_file.write(" ".join([str(round(x, 5)) for x in row]) + "\n")
                    phantom_file.write(" ".join(row) + "\n")
                phantom_file.write("\n")

    def generate_phantom(self):
        self.zoomed_grid, self.coords = self._get_coords()
        self.phantom_mats, self.phantom_densities = self._make_grids()

    def write_phantom(self, filename):
        return self._write_egsphant(filename)

    def convert_to_webgl_format(self):
        webgl_phantom = {}
        webgl_phantom["img_pos"] = self.coords.img_pos.tolist()
        webgl_phantom["num_voxels"] = self.coords.num_voxels.tolist()
        webgl_phantom["spacing"] = self.coords.spacing.tolist()

        webgl_phantom["max_norm"] = self.phantom_densities.max()
        webgl_phantom["min_norm"] = self.phantom_densities.min()
        webgl_phantom["num_mats"] = len(self.mat_names)

        webgl_phantom["mat_list"] = [mat for mat in self.mat_names]
        webgl_phantom["orient"] = self.orient.tolist()

        webgl_phantom["mat_matrix"] = {}
        webgl_phantom["density_matrix"] = {}

        for i in range(self.phantom_mats.shape[0]):
            webgl_phantom["mat_matrix"][i] = self.phantom_mats[i].flatten().tolist()
            webgl_phantom["density_matrix"][i] = self.phantom_densities[i].flatten().tolist()

        return webgl_phantom

    @staticmethod
    def read_egsphant_header(filename):
        """Populate phantom metadata dict from header of egsphant file."""
        phantom = {}

        with open(filename, "r") as phantom_file:
            num_materials = int(phantom_file.readline().strip())
            mat_list = []
            for _ in range(num_materials):
                mat_list.append(phantom_file.readline().strip())

            # dummy line
            phantom_file.readline()

            num_voxels = phantom_file.readline().strip().split()
            x_voxels = [float(x) * 10 for x in phantom_file.readline().strip().split()]
            y_voxels = [float(y) * 10 for y in phantom_file.readline().strip().split()]
            z_voxels = [float(z) * 10 for z in phantom_file.readline().strip().split()]

            spacing = [
                abs(x_voxels[1] - x_voxels[0]),
                abs(y_voxels[1] - y_voxels[0]),
                abs(z_voxels[1] - z_voxels[0])
            ]

            phantom["mat_list"] = mat_list
            phantom["num_voxels"] = num_voxels
            phantom["topleft"] = [x_voxels[0], y_voxels[0], z_voxels[0]]
            phantom["spacing"] = spacing

        return phantom
