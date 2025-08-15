import logging
import os
import pickle
import sys
import tkinter as tk
from tkinter import filedialog as fd
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from opentps.core.processing.dataComparison.gammaIndex import gammaIndex

from brachyutils.dose.dose_utils import BrachyDose


class BrachyDoseComparison:
    r"""
    A class to compare two BrachyDose objects by computing the percent difference and gamma index.
    Attributes:
        gamma_kwargs (dict): Keuword arguments for the pymedphys gamma index function. Please
            see pymedphys.gamma for more details
        dose1 (BrachyDose): The first BrachyDose object (reference).
        dose2 (BrachyDose): The second BrachyDose object (test), to resampled on the grid of dose1 with extrapolated points set to 0.
        voxel_centers (np.ndarray): The voxel centers of the dose grid.
        dose_2_grid_resampled (np.ndarray): The dose grid of dose2 resampled to the grid of dose1.
        percent_difference (BrachyDose): The percent difference between dose1 and dose2.
        gamma_index (BrachyDose): The gamma index between dose1 and dose2.
        gamma_dose_percent_threshold (float): The gamma dose percent threshold.
        gamma_distance_threshold (float): The gamma distance threshold in mm.
        gamma_kwargs (dict): The kwargs for the gamma index function.
        prescription_dose (float): The prescription dose of the dose grid.
        max_gamma (float): The maximum gamma index value.
        gamma_pass_ratio (float): The ratio of gamma index values passing the threshold.
    Methods:
        __init__(dose1, dose2, gamma_dose_percent_threshold, gamma_distance_threshold_mm, compute_percent_difference=True, compute_gamma_index=True, prescription_dose=None, max_gamma=None, path=None, gamma_kwargs=gamma_kwargs)
        plot_2d_dose_comparison(axis_1_coords, axis_2_coords, plane_coord, plane, plot_titles)
        compute_percent_difference()
        compute_gamma_index()
        save_comparison_object(path=None)
        load_comparison_object(path=None)

    """

    default_gamma_kwargs : dict = {
        "lower_percent_dose_cutoff": 5,
        "interp_fraction": 10,
        "local_gamma": False,
        "global_normalisation": None,
        "skip_once_passed": True,
        "max_gamma": 1.1
    }

    def __init__(
        self,
        dose1: BrachyDose,
        dose2: BrachyDose,
        gamma_dose_percent_threshold: float,
        gamma_distance_threshold_mm: float,
        compute_percent_difference=True,
        compute_gamma_index=True,
        path=None,
        gamma_kwargs: dict = default_gamma_kwargs,
        positive_percent_difference: bool = True,
        percent_difference_range: tuple = (0, 200),
        dose_mask: BrachyDose = None
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
            - positive_percent_difference: bool := if True, the percent difference will be computed with or without absolute value
            - percent_difference_range: tuple := the range of the percent difference used in plotting
            - dose_mask: BrachyDose := a mask to apply to the dose comparison, excluding any voxels where the mask is valued <0
        Outputs:
            Object containing the following attributes:
                - dose1: BrachyDose object
                - dose2: BrachyDose object, resampled on the grid of dose1 with extrapolated
                    points set to 0
                - voxel_centers: numpy array := the voxel centers of the dose grid
                - dose_2_grid_resampled: numpy array := the dose grid of dose2 resampled to the grid of dose1
                - percent_difference: BrachyDose object := the percent difference between dose1 and dose2
                - gamma_index: BrachyDose object := the gamma index between dose1 and dose2
                - gamma_dose_percent_threshold: float := the gamma dose percent threshold
                - gamma_distance_threshold: float := the gamma distance threshold in mm
                - gamma_kwargs: dict := the kwargs for the gamma index function
                - plot_max_dose_percent_of_prescription : int = default 300%, can be tuned to get a good dynamic range for plots

            and The following functions
                - compute_percent_difference: void := to compute the percent difference between dose1 and dose2
                - compute_gamma_index: void := to compute the gamma index between dose1 and dose2
                - plot_2d_dose_comparison: void := to plot the 2d dose comparison
                - save_comparison_object
                - load_comparison_object
        """
        # note: we will not use DoseComparisonImageProvider from OpenTPS
        # since the gamma index capabilities are not yet implemented
        # provide no dose to just load a file
        if dose1 is None and dose2 is None:
            self.load_comparison_object(path)
            return

        self.dose1 = dose1
        self.dose2 = dose2
        # axis is taken from the first dose provided
        self.voxel_centers = dose1.get_voxel_centers()
        # print("Before resample", self.dose2.dose_image is None)
        self.dose2.dose_image.resampleOn(dose1.dose_image, fillValue=0)
        # print("After resample", self.dose2.dose_image is None)
        self.percent_difference_local: BrachyDose = None
        self.percent_difference_global: BrachyDose = None
        # self.dose_comparision_image_provider = DoseComparisonImageProvider()
        self.gamma_index: BrachyDose = None
        self.gamma_dose_percent_threshold = gamma_dose_percent_threshold
        self.gamma_kwargs = BrachyDoseComparison.default_gamma_kwargs
        self.gamma_kwargs.update(gamma_kwargs) #in case the user wants to change the default
        self.plot_max_dose_percent_of_prescription : int = 200
        self.prescription_dose = gamma_kwargs.get("global_normalisation", 1.0)
        self.max_gamma = gamma_kwargs.get("max_gamma", 1.1)
        self.percent_difference_range = percent_difference_range
        self.dose_mask = dose_mask
        # axes values are assumed in cm from the 3ddose formalism
        # gamma distance thresholds are usually provided in mm
        # pymedphys documentation indicates that the threshold unit must match the axis
        # despite the name of the function input containing 'mm'
        self.gamma_distance_threshold = gamma_distance_threshold_mm
        if compute_percent_difference:
            self.compute_percent_difference(positive_percent_difference, local = True)
            self.compute_percent_difference(positive_percent_difference, local = False)
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
        """
        Plots a 2D dose comparison between two dose profiles, along with their percent difference and gamma index.
        Parameters:
        -----------
        axis_1_coords : np.ndarray
            Coordinates along the first axis (e.g., x-axis).
        axis_2_coords : np.ndarray
            Coordinates along the second axis (e.g., y-axis).
        plane_coord : float
            Coordinate of the plane in which the profiles are extracted.
        plane : str
            The plane in which the profiles are extracted (e.g., 'axial', 'sagittal', 'coronal').
        plot_titles : tuple
            Titles for the dose plots (dose 1 and dose 2).
        Raises:
        -------
        NotImplementedError
            If neither percent difference nor gamma index is computed.
        Notes:
        ------
        The function generates a figure suitable for a double column figure in medical physics publications.
        The figure is saved as an EPS file (user is prompted for the filename) and displayed.
        """

        # import itertools

        import matplotlib

        # from matplotlib.ticker import (
        # AutoMinorLocator,
        # FormatStrFormatter,
        # MultipleLocator,
        # )

        plot_vmax = self.plot_max_dose_percent_of_prescription / 100 * self.prescription_dose
        matplotlib.rcParams.update({"font.size": 8})
        plt.rcParams.update({"figure.dpi": 300})

        figure_colormap = plt.get_cmap("turbo")
        figure_colormap.set_bad(color="black", alpha=1.0)  # set bad values to black
    

        dummy_profile = np.zeros((len(axis_2_coords) - 1, len(axis_1_coords) - 1))

        dose_1_profile = self.dose1.extract_profile_2d(
            axis_1_coords, axis_2_coords, plane_coord, plane
        )
        dose_2_profile = self.dose2.extract_profile_2d(
            axis_1_coords, axis_2_coords, plane_coord, plane
        )
        if self.percent_difference_local is not None:
            percent_difference_profile = self.percent_difference_local.extract_profile_2d(
                axis_1_coords, axis_2_coords, plane_coord, plane
            )
        else:
            percent_difference_profile = dummy_profile
        if self.gamma_index is not None:
            gamma_index_profile = self.gamma_index.extract_profile_2d(
                axis_1_coords, axis_2_coords, plane_coord, plane
            )
        else:
            gamma_index_profile = dummy_profile

        if self.dose_mask is not None:
            dose_mask_profile = self.dose_mask.extract_profile_2d(
                axis_1_coords, axis_2_coords, plane_coord, plane
            )
            dose_1_profile[dose_mask_profile < 0] = np.nan
            dose_2_profile[dose_mask_profile < 0] = np.nan
            percent_difference_profile[dose_mask_profile < 0] = np.nan
            gamma_index_profile[dose_mask_profile < 0] = np.nan


        #flip the profiles 
        if plane == 'xy':
            dose_1_profile = np.flip(dose_1_profile, axis=0)
            dose_2_profile = np.flip(dose_2_profile, axis=0)
            percent_difference_profile = np.flip(percent_difference_profile, axis=0)
            gamma_index_profile = np.flip(gamma_index_profile, axis=0)


        # elif self.gamma_index is :
        #    raise NotImplementedError(
        #        """Plotting of a comparison without computing the percent difference or
        #    gamma index is not supported"""
        #    )
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
            vmax=plot_vmax,
            cmap=figure_colormap,
            rasterized=True,
            antialiased=True,
        )
        ax[0, 0].set_title(plot_titles[0], fontsize=12, pad=5, fontweight="bold")
        ax[0, 0].set_aspect("equal")
        cbar00 = fig.colorbar(c00, ax=ax[0, 0], shrink=0.9, pad=0.04)
        cbar00.set_label(label="Dose [Gy]", size=10, labelpad=5)
        # cbar00.mappable.set_clim(0, max_dose)
        ax[0, 0].invert_yaxis()
        ax[0, 0].set_ylabel(f"{plane[1]} [mm]", fontsize=10)
        c01 = ax[0, 1].pcolormesh(
            axis_1_coords,
            axis_2_coords,
            dose_2_profile,
            vmin=0,
            vmax=plot_vmax,
            cmap=figure_colormap,
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
            vmin=self.percent_difference_range[0],
            vmax=self.percent_difference_range[1],
            cmap=figure_colormap,
            rasterized=True,
            antialiased=True,
        )
        ax[1, 0].set_title("Percent Difference", fontsize=12, pad=5, fontweight="bold")
        ax[1, 0].set_aspect("equal")
        cbar10 = fig.colorbar(c10, ax=ax[1, 0], shrink=0.9, pad=0.04)
        cbar10.set_label(label="[%]", size=10, labelpad=5)
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_xlabel(f"{plane[0]} [mm]", fontsize=10)
        ax[1, 0].set_ylabel(f"{plane[1]} [mm]", fontsize=10)

        c11 = ax[1, 1].pcolormesh(
            axis_1_coords,
            axis_2_coords,
            gamma_index_profile,
            vmin=0,
            vmax=self.max_gamma,
            cmap=figure_colormap,
            rasterized=True,
            antialiased=True,
        )
        ax[1, 1].set_title(
            f"Gamma ({self.gamma_dose_percent_threshold}% / {int(self.gamma_distance_threshold)} mm)",
            fontsize=12,
            pad=5,
            fontweight="bold",
        )
        ax[1, 1].set_aspect("equal")
        #: Pass Rate = {np.round(self.gamma_pass_ratio*100,1)}%"
        cbar11 = fig.colorbar(c11, ax=ax[1, 1], shrink=0.9, pad=0.04)
        cbar11.set_label(label="Gamma", size=10, labelpad=5)
        ax[1, 1].invert_yaxis()
        ax[1, 1].set_xlabel(f"{plane[0]} [mm]", fontsize=10)
        plt.tight_layout()
        plt.show()

        root = tk.Tk()
        root.withdraw()
        f = fd.asksaveasfile(
            mode="wb",
            defaultextension=".eps",
            initialdir=os.getcwd(),
            title="Save dose comparison plots",
            confirmoverwrite=True,
        )
        if f is not None:
            plt.savefig(f, dpi=300)
            f.close()

        root.destroy()

    def compute_percent_difference(self, positive_percent_difference: bool = True, local: bool = True):
        """
        Compute the local or global percent difference between two dose distributions.
        self.dose1 is the reference and self.dose2 is the test dose.
        See TG-372 for exact definitions
        Returns:
            None
        """
        if local:
            self.percent_difference_local = BrachyDose.dose_with_empty_grid_like(self.dose1)
            if positive_percent_difference:
                self.percent_difference_local.dose_image.imageArray = (
                    np.abs(self.dose1.dose_image.imageArray - self.dose2.dose_image.imageArray)
                    / self.dose1.dose_image.imageArray
                    * 100.0
                )
            else:
                self.percent_difference_local.dose_image.imageArray = (
                    self.dose2.dose_image.imageArray - self.dose1.dose_image.imageArray
                ) * 100.0 / self.dose1.dose_image.imageArray
            if self.dose_mask is not None:
                self.percent_difference_local.dose_image.imageArray[self.dose_mask.dose_image.imageArray < 0] = np.nan

        else:
            self.percent_difference_global = BrachyDose.dose_with_empty_grid_like(self.dose1)
            if positive_percent_difference:
                self.percent_difference_global.dose_image.imageArray = (
                    np.abs(self.dose1.dose_image.imageArray - self.dose2.dose_image.imageArray)
                    / self.prescription_dose
                    * 100.0
                )
            else:
                self.percent_difference_global.dose_image.imageArray = (
                    self.dose2.dose_image.imageArray - self.dose1.dose_image.imageArray
                ) * 100.0 / self.prescription_dose

            if self.dose_mask is not None:
                self.percent_difference_global.dose_image.imageArray[self.dose_mask.dose_image.imageArray < 0] = np.nan

    def compute_gamma_index(self):
        """
        Compute the gamma index between two dose distributions.
        This method calculates the gamma index, which is a quantitative measure
        of the agreement between two dose distributions. The gamma index is
        computed using the dose images from `self.dose1` and `self.dose2`,
        along with specified dose percent and distance thresholds.
        The resulting gamma index is stored in `self.gamma_index`, and the
        pass ratio (the fraction of points with a gamma index less than or
        equal to 1) is stored in `self.gamma_pass_ratio`.
        Note:
            Computing the gamma index may take some time.
        Attributes:
            self.gamma_index (BrachyDose): A BrachyDose object with the computed
                gamma index.
            self.gamma_pass_ratio (float): The ratio of points with a gamma index
                less than or equal to 1.
        Raises:
            Any exceptions raised by the `gammaIndex` function or numpy operations
            will propagate up to the caller.
        """
        self.gamma_index = BrachyDose.dose_with_empty_grid_like(self.dose1)
        print("Computing gamma index may take time")
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        self.gamma_index.dose_image = gammaIndex(
            self.dose1.dose_image,
            self.dose2.dose_image,
            self.gamma_dose_percent_threshold,
            self.gamma_distance_threshold,
            **self.gamma_kwargs,
        )
        logger.setLevel(logging.INFO)
        self.gamma_index.dose_image.imageArray[self.dose_mask.dose_image.imageArray < 0] = np.nan
        # cast the NaNs to 0s
        gamma_index_grid = self.gamma_index.dose_image.imageArray
        number_excluded = np.sum(np.isnan(gamma_index_grid))
        gamma_index_grid[np.isnan(gamma_index_grid)] = -1
        self.gamma_pass_ratio = (np.sum(gamma_index_grid <= 1) - number_excluded) / (
            gamma_index_grid.size - number_excluded
        )
    
    def write_percent_difference_to_nrrd(self, dir: str = "./"):
        """
        Write the local and global percent difference to an NRRD file.
        Parameters:
            path (str): The path where the NRRD file will be saved.
        Returns:
            None
        """
        if self.percent_difference_local is not None and self.percent_difference_global is not None:
            self.percent_difference_local.write_to_nrrd(Path(f"{dir}/local_percent_difference.seq.nrrd"))
            self.percent_difference_global.write_to_nrrd(Path(f"{dir}/global_percent_difference.seq.nrrd"))
        else:
            raise ValueError("Local/global difference not computed. Call compute_percent_difference() first.")
    
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

    def plot_local_and_global_differences(
            self,
            axis_1_coords: np.ndarray,
            axis_2_coords: np.ndarray,
            plane_coord: float,
            plane: str,
            plot_titles: tuple,
        ):

        """
        Plot local and dose differences maps along both axes
        With the histograms below
        """

        # from matplotlib.ticker import (
        # AutoMinorLocator,
        # FormatStrFormatter,
        # MultipleLocator,
        # )
        import matplotlib

        local_vmax = 2
        global_vmax = 0.2

        matplotlib.rcParams.update({"font.size": 8})
        plt.rcParams.update({"figure.dpi": 300})

        image_cmap = plt.get_cmap('turbo')
        image_cmap.set_bad(color='black', alpha=1.0)  # set bad values to black
    

        local_difference_profile = self.percent_difference_local.extract_profile_2d(
            axis_1_coords, axis_2_coords, plane_coord, plane
        )
        global_difference_profile = self.percent_difference_global.extract_profile_2d(
            axis_1_coords, axis_2_coords, plane_coord, plane
        )

        mask_profile = self.dose_mask.extract_profile_2d(
            axis_1_coords, axis_2_coords, plane_coord, plane
        )

        local_difference_profile[mask_profile < 0] = np.nan
        global_difference_profile[mask_profile < 0] = np.nan

        #flip the profiles 
        if plane == 'xy':
            local_difference_profile = np.flip(local_difference_profile, axis=0)
            global_difference_profile = np.flip(global_difference_profile, axis=0)

        # we will plot a figure that is suitable as a double column figure for medical physics
        mm = 1.0 / 25.4  # define millimeters (relative to inches=1)
        fig, ax = plt.subplots(
            figsize=(360 * mm, 240 * mm), nrows=2, ncols=2, sharex=False, sharey=False, layout = "compressed",
            gridspec_kw={'width_ratios': [1, 1]}#,height_ratios=[1,1]
        )
        c00 = ax[0, 0].pcolormesh(
            axis_1_coords,
            axis_2_coords,
            local_difference_profile,
            vmin=-local_vmax,
            vmax=local_vmax,
            cmap=image_cmap,
            rasterized=True,
            antialiased=True,
        )

        ax[0, 0].set_title(plot_titles[0], fontsize=12, pad=5, fontweight="bold")
        ax[0, 0].set_aspect("equal")
        cbar00 = fig.colorbar(c00, ax=ax[0, 0], shrink=0.9, pad=0.04, location='right', panchor = False)
        cbar00.set_label(label="[%]", size=10, labelpad=5)
        cbar00.mappable.set_clim(-5, 5)
        ax[0, 0].invert_yaxis()
        ax[0, 0].set_xlabel(f"{plane[0]} [mm]", fontsize=10)
        ax[0, 0].set_ylabel(f"{plane[1]} [mm]", fontsize=10)

        c01 = ax[0, 1].pcolormesh(
            axis_1_coords,
            axis_2_coords,
            global_difference_profile,
            vmin=-global_vmax,
            vmax=global_vmax,
            cmap=image_cmap,
            rasterized=True,
            antialiased=True,
        )
        cbar01 = fig.colorbar(c01, ax=ax[0, 1], shrink=0.9, pad=0.04, location='right' )
        cbar01.set_label(label="[%]", size=10, labelpad=5)
        cbar01.mappable.set_clim(-1, 1)
        ax[0, 1].invert_yaxis()
        ax[0, 1].set_xlabel(f"{plane[0]} [mm]", fontsize=10)
        ax[0, 1].set_title(plot_titles[1], fontsize=12, pad=5, fontweight="bold")
        ax[0, 1].set_aspect("equal")



        #now plot the local and global percent differences histograms
        #use a bin width of 0.1%
        local_bin_width = 0.1
        global_bin_width = 0.001
        local_hist, local_bins = np.histogram(
            self.percent_difference_local.dose_image.imageArray, bins=np.arange(-local_vmax, local_vmax, local_bin_width)
        )
        global_hist, global_bins = np.histogram(
            self.percent_difference_global.dose_image.imageArray, bins=np.arange(-global_vmax, global_vmax, global_bin_width)
        )

        ax[1, 0].bar(local_bins[:-1], 100 * local_hist / self.percent_difference_local.dose_image.imageArray.size, width=local_bin_width, color="blue", alpha=0.5)#, fill=False)
        ax[1, 1].bar(global_bins[:-1], 100 * global_hist / self.percent_difference_global.dose_image.imageArray.size, width=global_bin_width, color="red", alpha=0.5)#, fill=False)
        ax[1, 0].set_box_aspect(aspect=1)
        ax[1, 1].set_box_aspect(aspect=1)

        ax[1, 0].set_ylabel("Voxels [%]", fontsize=10)
        ax[1, 0].set_xlabel(fr"$\Delta D_{{\mathrm{{LOCAL}}}} [\%]$", fontsize=10)
        ax[1, 1].set_xlabel(fr"$\Delta D_{{\mathrm{{GLOBAL}}}} [\%]$", fontsize=10)

        #plt.subplots_adjust(left = 0.184, bottom = 0.136, right = 0.813, top = 0.892, wspace = 0.219, hspace = 0.222)

        plt.show()

        root = tk.Tk()
        root.withdraw()
        f = fd.asksaveasfile(
            mode="wb",
            defaultextension=".eps",
            initialdir=os.getcwd(),
            title="Save dose difference plots",
            confirmoverwrite=True,
        )
        if f is not None:
            ext = os.path.splitext(f.name)[1][1:]  # get extension without dot
            plt.savefig(f, dpi=300, format=ext)
            f.close()
        root.destroy()


