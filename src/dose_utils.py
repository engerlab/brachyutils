from numpy import array as nparray, zeros as npzeros, reshape
from numpy import float as npfloat
from numpy import int as npint
from numpy import ma
from numpy import dtype
import re
import os

def load_pmc_dose(filename):
    return load_3ddose(filename)

def load_egsphant(filename):
    phant = {}
    with open(filename, "r") as egsphant:
        num_media = int(egsphant.readline().strip())
        phant["media"] = []
        for i in range(num_media):
            phant["media"].append(egsphant.readline().strip())

        # dummy line
        egsphant.readline()

        phant["num_voxels"] = [int(i) for i in egsphant.readline().strip().split()]
        phant["x_voxels"] = [float(x) for x in egsphant.readline().strip().split()]
        phant["y_voxels"] = [float(y) for y in egsphant.readline().strip().split()]
        phant["z_voxels"] = [float(z) for z in egsphant.readline().strip().split()]

        phant["mat_matrix"] = npzeros((phant["num_voxels"][2], phant["num_voxels"][1], phant["num_voxels"][0]), dtype=npint)
        phant["density_matrix"] = npzeros((phant["num_voxels"][2], phant["num_voxels"][1], phant["num_voxels"][0]), dtype=npfloat)

        for k in range(phant["num_voxels"][2]):
            for j in range(phant["num_voxels"][1]):
                phant["mat_matrix"][k][j] = list(egsphant.readline().strip())
            egsphant.readline()

        for k in range(phant["num_voxels"][2]):
            for j in range(phant["num_voxels"][1]):
                phant["density_matrix"][k][j] = egsphant.readline().strip().split()
            egsphant.readline()

    return phant

def load_3ddose(filename):
    # Load in the benchmark results.
    path = filename
    #print("Opening 3ddose at %s" % path)
    with open(path, "rb") as newfile:
        bench_voxels = [int(i) for i in newfile.readline().split()]
        bench_x_pos = nparray(newfile.readline().split(), dtype=npfloat)
        bench_y_pos = nparray(newfile.readline().split(), dtype=npfloat)
        bench_z_pos = nparray(newfile.readline().split(), dtype=npfloat)

        bench_x_spacing = (bench_x_pos[1] - bench_x_pos[0])
        bench_y_spacing = (bench_y_pos[1] - bench_y_pos[0])
        bench_slice_thick = (bench_z_pos[1] - bench_z_pos[0])

        huge_dose_array = nparray(newfile.readline().strip().split(), dtype=npfloat)
        bench_dose = reshape(huge_dose_array, (bench_voxels[2], bench_voxels[1], bench_voxels[0]))

        huge_uncert_array = nparray(newfile.readline().strip().split(), dtype=npfloat)
        bench_uncert = reshape(huge_uncert_array, (bench_voxels[2], bench_voxels[1], bench_voxels[0]))

        bench_dict = {}
        bench_dict["grid"] = bench_dose
        bench_dict["uncert"] = bench_uncert
        bench_dict["num_voxels"] = bench_voxels
        bench_dict["vox_size"] = [bench_x_spacing, bench_y_spacing, bench_slice_thick]
        bench_dict["topleft"] = [bench_x_pos[0], bench_y_pos[0], bench_z_pos[0]]

        # x_axis = bench_voxels[0] * bench_x
        # bench_dict["axis"] = 

    return bench_dict


def make_profile(dose, depth, axis = "x"):
    """
        Plots a profile at a given depth (z coordinate) inside a 3ddose file.
    """
    num_x, num_y, num_z = dose["num_voxels"]
    x_size, y_size, z_size = dose["vox_size"]
    topleft_x, topleft_y, topleft_z = dose["topleft"]
    depth_voxel = (depth - topleft_z) / z_size
    if axis == "x":
        off_axis_values = [topleft_x + (i + 0.5) * x_size for i in range(num_x)]
        mid_y = num_y / 2
        dose_values = [dose["grid"][depth_voxel][mid_y][i] for i in range(num_x)]
    elif axis == "y":
        off_axis_values = [topleft_y + (i + 0.5) * y_size for i in range(num_y)]
        mid_x = num_x / 2
        dose_values = [dose["grid"][depth_voxel][i][mid_x] for i in range(num_y)]
    else:
        raise("Only x or y axes are recognized")

    profile_dict = {}
    # Here, x and y axis refers to the axes on a graph, not
    # the dose axes.
    profile_dict["x_axis"] = off_axis_values
    profile_dict["y_axis"] = dose_values
    return profile_dict

def make_pdd(dose):
    mid_x, mid_y, mid_z = [int(vox/2) for vox in dose["num_voxels"]]
    x_size, y_size, z_size = dose["vox_size"]
    z_values = [(i + 0.5) * z_size for i in range(dose["num_voxels"][2])]
    dose_values = [dose["grid"][i][mid_y][mid_x] for i in range(dose["num_voxels"][2])]

    pdd_dict = {}
    if "uncert" in dose:
        uncert_values = [dose["uncert"][i][mid_y][mid_x] / 2.0 for i in range(dose["num_voxels"][2])]
        pdd_dict["uncert"] = uncert_values

    pdd_dict["x_axis"] = z_values
    pdd_dict["y_axis"] = nparray(dose_values)
    return pdd_dict


def get_average_uncert(dose):
    max_dose = dose["grid"].max()
    dose_mask = dose["grid"] < 0.2 * max_dose
    masked_uncert = ma.array(dose["uncert"], mask=dose_mask)
    masked_dose = ma.array(dose["grid"], mask=dose_mask)
    average_uncert = ma.average(masked_uncert / masked_dose) * 100
    return average_uncert


def get_average_uncert_benchmark(dose):
    max_dose = dose["grid"].max()
    dose_mask = dose["grid"] < 0.2 * max_dose
    masked_uncert = ma.array(dose["uncert"], mask=dose_mask)
    average_uncert = ma.average(masked_uncert) * 100
    return average_uncert
