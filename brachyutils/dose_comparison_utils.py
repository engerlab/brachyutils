import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
import sys
import pymedphys
import logging
from brachyutils import BrachyDose
import tkinter as tk
from tkinter import filedialog as fd

class DoseComparison:
    gamma_kwargs: dict = (
        {
            "lower_percent_dose_cutoff": 5,
            "interp_fraction": 10,
            "local_gamma": False,
            "global_normalisation": None,
            "skip_once_passed": False,
        },
    )

    def __init__(
        self,
        dose1: BrachyDose,
        dose2: BrachyDose,
        gamma_dose_percent_threshold: float,
        gamma_distance_threshold_mm: float,
        compute_percent_difference=True,
        compute_gamma_index=True,
        prescription_dose: float = None,
        max_gamma=None,
        path=None,
        gamma_kwargs: dict = gamma_kwargs,
    ):
        r"""
        Purpose:
            - to compare two BrachyDose objects. The comparison is done by computing the percent difference and gamma index.
            The gamma index is computed using the pymedphys gamma function. The result of the comparison is stored in the object and
            can be viewed using the plot_2d_dose_comparison function.
        Inputs:
            - dose1: BrachyDose object
            - dose2: BrachyDose object
            - gamma_dose_percent_threshold: float := the gamma dose percent threshold
            - gamma_distance_threshold_mm: float := the gamma distance threshold in mm
            - compute_percent_difference: bool := if True, the percent difference will be computed
            - compute_gamma_index: bool := if True, the gamma index will be computed
            - prescription_dose: float := the prescription dose of the dose grid
            - max_gamma: float := the maximum gamma index value
            - path: str := the path to the comparison object
            - gamma_kwargs: dict := the kwargs for the gamma index function
        Outputs:
            Object containing the following attributes:
                - dose1: BrachyDose object
                - dose2: BrachyDose object
                - voxel_centers: numpy array := the voxel centers of the dose grid
                - dose_2_grid_resampled: numpy array := the dose grid of dose2 resampled to the grid of dose1
                - percent_difference: BrachyDose object := the percent difference between dose1 and dose2
                - gamma_index: BrachyDose object := the gamma index between dose1 and dose2
                - gamma_dose_percent_threshold: float := the gamma dose percent threshold
                - gamma_distance_threshold: float := the gamma distance threshold in mm
                - gamma_kwargs: dict := the kwargs for the gamma index function

            and The following functions
                - compute_percent_difference: void := to compute the percent difference between dose1 and dose2
                - compute_gamma_index: void := to compute the gamma index between dose1 and dose2
                - plot_2d_dose_comparison: void := to plot the 2d dose comparison
                - save_comparison_object
                - load_comparison_object
        """
        # provide no dose to just load a file
        if dose1 is None and dose2 is None:
            self.load_comparison_object(path)
            return

        self.dose1 = dose1
        self.dose2 = dose2
        # axis is taken from the first dose provided
        self.voxel_centers = dose1.get_voxel_centers()
        self.dose_2_grid_resampled = self.dose2.extract_dose_values_from_coordinates(
            self.voxel_centers[2], self.voxel_centers[1], self.voxel_centers[0]
        )
        self.percent_difference: BrachyDose = None
        self.gamma_index: BrachyDose = None
        self.gamma_dose_percent_threshold = gamma_dose_percent_threshold
        self.gamma_kwargs = gamma_kwargs
        # we can index the dose cutoff to the prescription dose
        if isinstance(prescription_dose, float) or isinstance(prescription_dose, int):
            self.gamma_kwargs["global_normalisation"] = prescription_dose
        if isinstance(max_gamma, float) or isinstance(prescription_dose, int):
            self.max_gamma = max_gamma
            self.gamma_kwargs["max_gamma"] = max_gamma
        else:
            self.max_gamma = 2
        # axes values are assumed in cm from the 3ddose formalism
        # gamma distance thresholds are usually provided in mm
        # pymedphys documentation indicates that the threshold unit must match the axis
        # despite the name of the function input containing 'mm'
        self.gamma_distance_threshold = gamma_distance_threshold_mm / 10.0
        if compute_percent_difference:
            self.compute_percent_difference()
        if compute_gamma_index:
            self.compute_gamma_index()

    def plot_2d_dose_comparison(
        self,
        axis_1_coords: np.ndarray,
        axis_2_coords: np.ndarray,
        plane_coord: float,
        plane: str,
        plot_titles: tuple,
    ):
        # import itertools

        import matplotlib

        # from matplotlib.ticker import (
        # AutoMinorLocator,
        # FormatStrFormatter,
        # MultipleLocator,
        # )

        matplotlib.rcParams.update({"font.size": 8})
        plt.rcParams.update({"figure.dpi": 300})
        dose_1_profile = self.dose1.extract_profile_2d(
            axis_1_coords, axis_2_coords, plane_coord, plane
        )
        dose_2_profile = self.dose2.extract_profile_2d(
            axis_1_coords, axis_2_coords, plane_coord, plane
        )
        if self.percent_difference is not None:
            percent_difference_profile = self.percent_difference.extract_profile_2d(
                axis_1_coords, axis_2_coords, plane_coord, plane
            )
        if self.gamma_index is not None:
            gamma_index_profile = self.gamma_index.extract_profile_2d(
                axis_1_coords, axis_2_coords, plane_coord, plane
            )
        else:
            raise NotImplementedError(
                """Plotting of a comparison without computing the percent difference or
            gamma index is not supported"""
            )
        # we will plot a figure that is suitable as a double column figure for medical physics
        mm = 1.0 / 25.4  # define millimeters (relative to inches=1)
        fig, ax = plt.subplots(
            figsize=(180 * mm, 120 * mm), nrows=2, ncols=2, sharex=True, sharey=True
        )
        c00 = ax[0, 0].pcolormesh(
            axis_1_coords,
            axis_2_coords,
            dose_1_profile,
            vmin=0,
            vmax=30,
            cmap="turbo",
            rasterized=True,
            antialiased=True,
        )
        ax[0, 0].set_title(plot_titles[0], fontsize=12, pad=5, fontweight="bold")
        ax[0, 0].set_aspect("equal")
        cbar00 = fig.colorbar(c00, ax=ax[0, 0], shrink=0.9, pad=0.04)
        cbar00.set_label(label="Dose [Gy]", size=10, labelpad=5)
        # cbar00.mappable.set_clim(0, max_dose)
        ax[0, 0].invert_yaxis()
        ax[0, 0].set_ylabel("y (cm)", fontsize=10)
        c01 = ax[0, 1].pcolormesh(
            axis_1_coords,
            axis_2_coords,
            dose_2_profile,
            vmin=0,
            vmax=30,
            cmap="turbo",
            rasterized=True,
            antialiased=True,
        )
        ax[0, 1].set_title(plot_titles[1], fontsize=12, pad=5, fontweight="bold")
        ax[0, 1].set_aspect("equal")

        cbar01 = fig.colorbar(c01, ax=ax[0, 1], shrink=0.9, pad=0.04)
        cbar01.set_label(label="Dose [Gy]", size=10, labelpad=5)
        # cbar01.mappable.set_clim(0, max_dose)g
        ax[0, 1].invert_yaxis()
        c10 = ax[1, 0].pcolormesh(
            axis_1_coords,
            axis_2_coords,
            percent_difference_profile,
            vmin=0,
            vmax=200,
            cmap="turbo",
            rasterized=True,
            antialiased=True,
        )
        ax[1, 0].set_title("Percent Difference", fontsize=12, pad=5, fontweight="bold")
        ax[1, 0].set_aspect("equal")
        cbar10 = fig.colorbar(c10, ax=ax[1, 0], shrink=0.9, pad=0.04)
        cbar10.set_label(label="[%]", size=10, labelpad=5)
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_xlabel("x (cm)", fontsize=10)
        ax[1, 0].set_ylabel("y (cm)", fontsize=10)

        c11 = ax[1, 1].pcolormesh(
            axis_1_coords,
            axis_2_coords,
            gamma_index_profile,
            vmin=0,
            vmax=self.max_gamma,
            cmap="turbo",
            rasterized=True,
            antialiased=True,
        )
        ax[1, 1].set_title(
            f"Gamma ({self.gamma_dose_percent_threshold}% / {int(10.*self.gamma_distance_threshold)} mm)",
            fontsize=12,
            pad=5,
            fontweight="bold",
        )
        ax[1, 1].set_aspect("equal")
        #: Pass Rate = {np.round(self.gamma_pass_ratio*100,1)}%"
        cbar11 = fig.colorbar(c11, ax=ax[1, 1], shrink=0.9, pad=0.04)
        cbar11.set_label(label="Gamma", size=10, labelpad=5)
        ax[1, 1].invert_yaxis()
        ax[1, 1].set_xlabel("x (cm)", fontsize=10)
        plt.tight_layout()
        plt.savefig("dose_comparison.eps", dpi=300)
        plt.show()

    def compute_percent_difference(self):
        self.percent_difference = BrachyDose()
        self.percent_difference.grid = (
            np.abs(self.dose1.grid - self.dose_2_grid_resampled) / self.dose1.grid * 100
        )
        self.percent_difference.voxel_edges = self.dose1.voxel_edges
        self.percent_difference.voxel_size = self.dose1.voxel_size
        self.percent_difference.origin_coordinates = self.dose1.origin_coordinates
        self.percent_difference.num_voxels = self.dose1.num_voxels
        self.percent_difference.create_interpolation_function()

    def compute_gamma_index(self):
        print("Computing gamma index may take time")
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.gamma_index = BrachyDose()
        gamma_index_grid = pymedphys.gamma(
            tuple(self.voxel_centers),
            self.dose1.grid,
            tuple(self.voxel_centers),
            self.dose_2_grid_resampled,
            self.gamma_dose_percent_threshold,
            self.gamma_distance_threshold,
            **self.gamma_kwargs,
        )
        # cast the NaNs to 0s
        number_excluded = np.sum(np.isnan(gamma_index_grid))
        gamma_index_grid[np.isnan(gamma_index_grid)] = -1
        self.gamma_index.grid = gamma_index_grid
        self.gamma_index.voxel_edges = self.dose1.voxel_edges
        self.gamma_index.voxel_size = self.dose1.voxel_size
        self.gamma_index.origin_coordinates = self.dose1.origin_coordinates
        self.gamma_index.num_voxels = self.dose1.num_voxels
        self.gamma_pass_ratio = (
            np.sum(self.gamma_index.grid <= 1) - number_excluded
        ) / (self.gamma_index.grid.size - number_excluded)
        self.gamma_index.create_interpolation_function()

    def save_comparison_object(self, path: str = None):
        r"""
        Saves the dose comparison object to a file using the pickle module.

        Returns:
        None
        """
        if not isinstance(path, str):
            root = tk.Tk()
            root.withdraw()
            f = fd.asksaveasfile(
                mode="wb",
                defaultextension=".comp",
                initialdir=os.getcwd(),
                title="Save dose comparison object",
                confirmoverwrite=True,
            )
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            root.destroy()
            f.close()
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load_comparison_object(self, path: str = None):
        r"""
        Opens the dose comparison object file and updates the current object's attributes with the loaded object's attributes.

        Returns:
        None
        """
        if not isinstance(path, str):
            root = tk.Tk()
            root.withdraw()
            f = fd.askopenfile(
                mode="rb",
                parent=root,
                initialdir="$HOME",
                title="Select saved dose comparison file",
            )
            self.__dict__.update(pickle.load(f).__dict__)
            # print(calibration_object_file_path)
            root.destroy()
            f.close()
        else:
            with open(path, "rb") as f:
                self.__dict__.update(pickle.load(f).__dict__)
