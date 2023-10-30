import pickle
import tkinter as tk
from tkinter import filedialog as fd
import sys
import os
import dose_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import tifffile
import pymedphys
from scipy import ndimage
from scipy.optimize import curve_fit
import cv2


class CalibrationCurve:
    """
    A class used to represent a calibration curve for film dosimetry.

    Attributes:
    - doses (numpy.ndarray): doses used to create calibration curve
    - r_pv (numpy.ndarray): red channel pixel values on calibration films
    - g_pv (numpy.ndarray): green channel pixel values
    - b_pv (numpy.ndarray): blue channel pixel values
    - r_std (numpy.ndarray): standard deviation of red channel pixel values
    - g_std (numpy.ndarray): standard deviation of green channel pixel values
    - b_std (numpy.ndarray): standard deviation of blue channel pixel values
    - calibration_curve_type (str): type of calibration curve
    - r_opt (numpy.ndarray): optimal parameters for red channel calibration curve
    - g_opt (numpy.ndarray): optimal parameters for green channel calibration curve
    - b_opt (numpy.ndarray): optimal parameters for blue channel calibration curve
    - rp_cov (numpy.ndarray): covariance matrix for red channel fit
    - gp_cov (numpy.ndarray): covariance matrix for green channel fit
    - bp_cov (numpy.ndarray): covariance matrix for blue channel fit
    - dose_to_pv (function): function to convert dose to pixel value
    - pv_to_dose (function): function to convert pixel value to dose
    - r_pv_fit (numpy.ndarray): red channel pixel values normalized to the zero dose pixel value
    - g_pv_fit (numpy.ndarray): green channel pixel values normalized to the zero dose pixel value
    - b_pv_fit (numpy.ndarray): blue channel pixel values normalized to the zero dose pixel value (for Lewis) or net pixel value (for Devic)

    Methods:
    - perform_fit(): performs the fit for the calibration curve
    - define_fit_parameters_for_calibration_type(): defines the fit parameters for the calibration type based on the calibration curve type
    - plot_calibration_and_response_curve(): plots the calibration and response curves for the film dosimetry system
    """

    def __init__(self, doses, r_pv, g_pv, b_pv, r_std, g_std, b_std, calibration_curve_type):
        self.doses: np.array = doses #doses used to create calibration curve
        self.r_pv: np.array = r_pv #red channel pixel values on calibration films
        self.g_pv: np.array = g_pv #green channel pixel values
        self.b_pv: np.array = b_pv #blue channel pixel values
        self.r_std: np.array = r_std #standard deviation of red channel pixel values
        self.g_std: np.array = g_std #standard deviation of green channel pixel values
        self.b_std: np.array = b_std #standard deviation of blue channel pixel values
        self.calibration_curve_type = calibration_curve_type #type of calibration curve
        self.r_opt: np.array = None #optimal parameters for red channel calibration curve
        self.g_opt: np.array = None #optimal parameters for green channel calibration curve
        self.b_opt: np.array = None #optimal parameters for blue channel calibration curve
        self.rp_cov: np.array = None #covariance matrix for red channel fit
        self.gp_cov: np.array = None #covariance matrix for green channel fit
        self.bp_cov: np.array = None #covariance matrix for blue channel fit
        self.dose_to_pv = None #function to convert dose to pixel value
        self.pv_to_dose = None #function to convert pixel value to dose
        self.r_pv_fit: np.array = None #mean red channel pixel values to fit; normalized to the zero dose pixel value
        self.g_pv_fit: np.array = None #mean green channel pixel values to fit; normalized to the zero dose pixel value
        self.b_pv_fit: np.array = None #mean blue channel pixel values to fit; normalized to the zero dose pixel value
        self.define_fit_parameters_for_calibration_type()
        self.perform_fit()
        self.plot_calibration_and_response_curve()

    def perform_fit(self):
        """
        Perform curve fitting to obtain optimal parameters and covariance matrices for red, green, and blue channels.

        Returns:
        - r_opt: array_like
        Optimal parameters for red channel.
        - rp_cov: 2-D array
        Covariance matrix for the estimated parameters for red channel.
        - g_opt: array_like
        Optimal parameters for green channel.
        - gp_cov: 2-D array
        Covariance matrix for the estimated parameters for green channel.
        - b_opt: array_like
        Optimal parameters for blue channel.
        - bp_cov: 2-D array
        Covariance matrix for the estimated parameters for blue channel.
        """
        self.r_opt, self.rp_cov = curve_fit(self.dose_to_pv, self.doses, self.r_pv_fit, sigma=self.r_std)
        self.g_opt, self.gp_cov = curve_fit(self.dose_to_pv, self.doses, self.g_pv_fit, sigma=self.g_std)
        self.b_opt, self.bp_cov = curve_fit(self.dose_to_pv, self.doses, self.b_pv_fit, sigma=self.b_std)

    def define_fit_parameters_for_calibration_type(self):
        """
        Defines the fit parameters for the calibration type based on the calibration curve type.

        Args:
        - doses (numpy.ndarray): Array of doses
        - r_pv (numpy.ndarray): Array of red channel pixel values
        - g_pv (numpy.ndarray): Array of green channel pixel values
        - b_pv (numpy.ndarray): Array of blue channel pixel values

        Returns:
        - Tuple containing:
        - dose_to_pv (function): Function to convert dose to pixel value
        - pv_to_dose (function): Function to convert pixel value to dose
        - r_pv_fit (numpy.ndarray): Array of red channel pixel values normalized to the zero dose pixel value
        - g_pv_fit (numpy.ndarray): Array of green channel pixel values normalized to the zero dose pixel value
        - b_pv_fit (numpy.ndarray): Array of blue channel pixel values normalized to the zero dose pixel value (for Lewis) or net pixel value (for Devic)
        """
        if self.calibration_curve_type == "Lewis":
            self.dose_to_pv = FilmCalibration.lewis_dose_to_pv
            self.pv_to_dose = FilmCalibration.lewis_pv_to_dose
            # normalize to the zero dose pv for Lewis
            #print(r_pv, doses==0)
            self.r_pv_fit = self.r_pv/(self.r_pv[self.doses == 0])
            self.g_pv_fit = self.g_pv/(self.g_pv[self.doses == 0])
            self.b_pv_fit = self.b_pv/(self.b_pv[self.doses == 0])

        elif self.calibration_curve_type == "Devic":
            self.dose_to_pv = FilmCalibration.devic_dose_to_pv
            self.pv_to_dose = FilmCalibration.devic_pv_to_dose
            # Devic uses net pvs
            self.r_pv_fit = np.abs(np.array(self.r_pv) - self.r_pv[self.doses == 0])
            self.g_pv_fit = np.abs(np.array(self.g_pv) - self.g_pv[self.doses == 0])
            self.b_pv_fit = np.abs(np.array(self.b_pv) - self.b_pv[self.doses == 0])

    def plot_calibration_and_response_curve(self):
        """
        Plots the calibration and response curves for the film dosimetry system.
        The calibration curve relates the measured pixel values (PV) to the actual
        radiation dose delivered to the film. The response curve relates the
        normalized PV to the actual radiation dose delivered to the film.
        """

        #doses = self.calibration_curve_params['doses']
        #r_pv = self.calibration_curve_params['r_pv']
        #g_pv = self.calibration_curve_params['g_pv']
        #b_pv = self.calibration_curve_params['b_pv']
        #r_opt = self.calibration_curve_params['r_opt']
        #g_opt = self.calibration_curve_params['g_opt']
        #b_opt = self.calibration_curve_params['b_opt']

        fig, ax = plt.subplots(2, 1, dpi=150, sharex=True)
        # Remove space so awesome sharex visual
        fig.subplots_adjust(hspace=0.03)
        ax[0].plot(self.doses, self.r_pv, 'r', label="Red")
        ax[0].plot(self.doses, self.g_pv, 'g', label="Green")
        ax[0].plot(self.doses, self.b_pv, 'b', label="Blue")
        yl1 = ax[0].set_ylabel("PV", labelpad=25)
        yl1.set_rotation(0)
        ax[0].grid()
        ax[0].legend()

        # Plot derivative to highlight usable ranges per channel
        d_dose = (np.array(self.doses)[:-1] + np.array(self.doses)[1:]) / 2
        d_r_pv = np.diff(self.r_pv) / np.diff(self.doses)
        d_g_pv = np.diff(self.g_pv) / np.diff(self.doses)
        d_b_pv = np.diff(self.b_pv) / np.diff(self.doses)
        ax[1].plot(d_dose, -d_r_pv, 'r', label="Red")
        ax[1].plot(d_dose, -d_g_pv, 'g', label="Green")
        ax[1].plot(d_dose, -d_b_pv, 'b', label="Blue")
        ax[1].set_xlabel("Dose (Gy)")
        yl2 = ax[1].set_ylabel(
        r"$-\frac{\partial{PV}}{\partial Dose}$", labelpad=25)
        yl2.set_rotation(0)
        ax[1].set_yscale('log')
        ax[1].grid()
        ax[1].legend()

        dose_array = np.linspace(0, 40, 100)
        r_pv_array = np.linspace(0.12, 1, 100)
        g_pv_array = np.linspace(0.15, 1, 100)
        b_pv_array = np.linspace(0.30, 1, 100)

        fig, ax = plt.subplots(1, 3, dpi=150, figsize=(20, 5))

        legend_elements = [Line2D([0], [0], marker='.', markersize=10, markerfacecolor='black', color='w', label='Experiment'),
        Line2D([0], [0], color='black', label='Fit')]

        # Response Curve
        ax[0].plot(self.doses, self.r_pv_fit, 'r.')
        ax[0].plot(self.doses, self.g_pv_fit, 'g.')
        ax[0].plot(self.doses, self.b_pv_fit, 'b.')

        ax[0].plot(dose_array, self.dose_to_pv(
        dose_array, *self.r_opt), 'r', label="Red fit")
        ax[0].plot(dose_array, self.dose_to_pv(
        dose_array, *self.g_opt), 'g', label="Green fit")
        ax[0].plot(dose_array, self.dose_to_pv(
        dose_array, *self.b_opt), 'b', label="Blue fit")

        ax[0].grid()
        ax[0].set_xlabel("Dose (Gy)")
        ax[0].set_ylabel("Normalized PV")
        ax[0].legend(loc='best', handles=legend_elements)
        ax[0].set_title('Response Curve')

        # Calibration Curve
        ax[1].plot(self.r_pv_fit, self.doses, 'r.')
        ax[1].plot(self.g_pv_fit, self.doses, 'g.')
        ax[1].plot(self.b_pv_fit, self.doses, 'b.')

        ax[1].plot(r_pv_array, self.pv_to_dose(r_pv_array, *self.r_opt), 'r')
        ax[1].plot(g_pv_array, self.pv_to_dose(g_pv_array, *self.g_opt), 'g')
        ax[1].plot(b_pv_array, self.pv_to_dose(b_pv_array, *self.b_opt), 'b')

        ax[1].grid()
        ax[1].set_xlabel("Normalized PV")
        ax[1].set_ylabel("Dose (Gy)")
        ax[1].legend(loc='best', handles=legend_elements)
        ax[1].set_title('Calibration Curve')

        # Dose Residual Plot
        ax[2].plot(self.doses, 100*(self.doses - self.pv_to_dose(self.r_pv_fit, *self.r_opt))/self.doses, 'r.')
        ax[2].plot(self.doses, 100*(self.doses - self.pv_to_dose(self.g_pv_fit, *self.g_opt))/self.doses, 'g.')
        ax[2].plot(self.doses, 100*(self.doses - self.pv_to_dose(self.b_pv_fit, *self.b_opt))/self.doses, 'b.')

        ax[2].axhline(-2, linestyle='--')
        ax[2].axhline(2, linestyle='--')

        ax[2].set_ylabel("Given Dose - Measured Dose (%)")
        ax[2].set_xlabel("Dose (Gy)")
        ax[2].grid()
        ax[2].set_title('Dose Error')

class FilmCalibration:
    r"""
    A utility class for processing film images in the context of brachytherapy.

    This class provides methods for loading and processing calibration images, as well as for applying
    calibration curves to film images. It also stores various parameters related to the calibration process,
    such as the pixel range, the calibration file directory, and the ROI bounds.

    Attributes:
    pixel_range (int): The maximum pixel value for the film images.
    calibration_file_dict (dict): A dictionary mapping dose values to calibration file paths and calibration
    coefficients.
    calibration_images (dict): A dictionary mapping dose values to calibration images.
    calibration_curve_type (str): The type of calibration curve used to fit the calibration data.
    calibration_curve_params (dict): A dictionary of parameters for the calibration curve.
    film_to_mc_offset (tuple): The offset between the film and the Monte Carlo (MC) simulation.
    roi_bounds (tuple): The bounds of the region of interest (ROI) for the film images.
    calibration_file_directory (str): The directory where the calibration files are stored.
    calibration_object_file_path (str): The path to the calibration object file.
    """

    possible_calibrations = ["Lewis", "Devic"]

    def __init__(self):
        self.pixel_range: int = None
        self.calibration_file_dict: dict = {}  # [float, tuple]
        self.calibration_images: dict = {}  # [float, np.ndarray]
        self.calibration_curve_type: str = None
        self.calibration_curve: CalibrationCurve = None
        self.film_to_mc_offset: tuple = None
        self.roi_bounds: tuple = (100, 100, 150, 150)  # x1 y1 x2 y2
        self.calibration_file_directory: str = "$HOME"
        self.calibration_object_file_path: str = "$HOME"


    def configure_calibration(self):
        r"""
        Configures the calibration settings for the film.

        Prompts the user to choose the calibration curve type, 
        enter the number of possible pixel values,
        and select calibration film files for a given dose. 
        Then loads and plots the calibration data.
        """
        while self.calibration_curve_type not in FilmCalibration.possible_calibrations:
            print("Choose calibration curve type (Lewis or Devic):")
            self.calibration_curve_type = input()
        while not isinstance(self.pixel_range, int):
            print(
            "Enter the number of possible pixel values (e.g. 2^16 for 16 bit images):")
            pixel_range_str = input()
        try:
            self.pixel_range = int(pixel_range_str)
        except ValueError:
            print("Invalid input. Please enter an integer.")
        print("Begin selecting calibration film files. Enter a dose and select the calibration films at that dose or enter \'Done\' to finish.")
        current_dose_str = input()
        while current_dose_str != "Done":
            try:
                current_dose = float(current_dose_str)
                self.add_calibration_files_for_dose(current_dose)
            except ValueError:
                print("Invalid input. Please enter a float or \'Done\' to finish.")
            current_dose_str = input()
        self.load_calibration()

    def add_calibration_files_for_dose(self, dose: float):
        r"""Opens a file dialog to select calibration film files corresponding to a given dose.

        Args:
        dose (float): The dose in Gy for which to select calibration files.

        Returns:
        None

        Side effects:
        - Modifies the `calibration_file_dict` attribute of the object to include the selected files.
        - Modifies the `calibration_file_directory` attribute of the object to the directory of the last selected file.
        """
        root = tk.Tk()
        root.withdraw()
        filenames_for_dose = fd.askopenfilenames(parent=root, initialdir=self.calibration_file_directory,
        title='Choose calibration films corresponding to a dose of ' + str(dose) + ' Gy')
        if len(filenames_for_dose) > 0:
            # start navigation of directory of last selected file
            self.calibration_file_directory = os.path.dirname(
            filenames_for_dose[0])
            self.calibration_file_dict[dose] = filenames_for_dose
        root.destroy()

    def load_calibration(self):
        r"""
        Loads calibration images from a dictionary of file paths and calculates the mean pixel value for each image.

        Raises:
        ValueError: If the calibration file dictionary is not set.

        Returns:
        None
        """
        if self.calibration_file_dict is None:
            raise ValueError("calibration file dictionary not set")
        else:
            self.calibration_images = {j: np.mean([tifffile.imread(
            i) for i in self.calibration_file_dict[j]], axis=0) / self.pixel_range for j in self.calibration_file_dict.keys()}


    def display_calibration_films(self):
        r"""
        Plots the calibration images with the region of interest bounds.
        """
        if self.calibration_images is None:
            raise ValueError("calibration images not loaded")
        else:
            nplots = len(self.calibration_images.keys())
            _, axes = plt.subplots(
            1, nplots,  dpi=124, figsize=(15, 10), squeeze=True)
            i = 0
            if nplots == 1:
                plt.imshow(self.calibration_images[list(
                self.calibration_images.keys())[i]][:, :, 0], cmap='trubo')
                plt.title(str(list(self.calibration_images.keys())[i]))
                plt.axvline(self.roi_bounds[0], color='r')
                plt.axhline(self.roi_bounds[1], color='r')
                plt.axvline(self.roi_bounds[2], color='r')
                plt.axhline(self.roi_bounds[3], color='r')
            else:
                for ax in axes.flatten():
                    ax.imshow(self.calibration_images[list(
                    self.calibration_images.keys())[i]][:, :, 0], cmap='turbo')
                    ax.set_title(str(list(self.calibration_images.keys())[i]))
                    ax.axvline(self.roi_bounds[0], color='r')
                    ax.axhline(self.roi_bounds[1], color='r')
                    ax.axvline(self.roi_bounds[2], color='r')
                    ax.axhline(self.roi_bounds[3], color='r')
                    i += 1

    def set_roi_bounds(self, new_bounds: tuple):
        r"""
        Set the region of interest (ROI) bounds for the film.

        Args:
        new_bounds (tuple): A tuple of length 4 containing the x and y coordinates of the top-left corner of the ROI,
        followed by the width and height of the ROI.

        Raises:
        ValueError: If the length of new_bounds is not 4 or if any element in new_bounds is not an integer.
        """
        if len(new_bounds) != 4:
            raise ValueError("ROI bounds must be a tuple of length 4")
        for i in new_bounds:
            if not isinstance(i, int):
                raise ValueError("ROI bounds must be a tuple of integers")
        self.roi_bounds = new_bounds

    def create_calibration_curve(self):
        r"""
        Create a calibration curve for a film dosimetry image.

        The calibration curve is created by analyzing a set of calibration images
        with known doses and calculating the mean pixel value in a region of
        interest (ROI) for each color channel (red, green, blue) as a function of
        dose. The calibration curve is then fitted to a response curve and a
        calibration curve, which are plotted along with a dose residual plot.

        Returns:
        None
        """
        doses = np.array(list(self.calibration_images.keys()))
        r_pv = np.array([])
        g_pv = []
        b_pv = []

        r_std = []
        g_std = []
        b_std = []

        # populate arrays with mean pixel value in ROI as a function of dose
        r_pv = np.array([self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3], self.roi_bounds[0]:self.roi_bounds[2], 0].mean() for dose in doses])
        g_pv = np.array([self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3], self.roi_bounds[0]:self.roi_bounds[2], 1].mean() for dose in doses])
        b_pv = np.array([self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3], self.roi_bounds[0]:self.roi_bounds[2], 2].mean() for dose in doses])

        r_std = np.array([self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3], self.roi_bounds[0]:self.roi_bounds[2], 0].std() for dose in doses])
        g_std = np.array([self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3], self.roi_bounds[0]:self.roi_bounds[2], 1].std() for dose in doses])
        b_std = np.array([self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3], self.roi_bounds[0]:self.roi_bounds[2], 2].std() for dose in doses])

        #print('red channel:', rstd)
        #print('green channel:', gstd)
        #print('blue channel:', bstd)
        self.calibration_curve = CalibrationCurve(doses, r_pv, g_pv, b_pv, r_std, g_std, b_std, self.calibration_curve_type)


    def normalize_image_by_calibration_type(self, image: np.array):
        """
        Normalizes the given image based on the calibration curve type.

        Args:
        - image: A numpy array representing the image to be normalized, with shape (height, width, 3).

        Returns:
        A tuple containing the pv_to_dose conversion function and the normalized image.

        Throws: 
        ValueError: If the image does not have 3 channels.
        """
        if image.shape[2] !=3:
            raise ValueError("Image must have 3 channels")
        norm_image = np.zeros(image.shape)
        doses = self.calibration_curve['doses']
        r_pv = self.calibration_curve['r_pv']
        g_pv = self.calibration_curve['g_pv']
        b_pv = self.calibration_curve['b_pv']
        if self.calibration_curve_type == "Lewis":
            pv_to_dose = FilmCalibration.lewis_pv_to_dose
            norm_image[:,:,0] = image[:,:,0] / r_pv[doses == 0]
            norm_image[:,:,1] = image[:,:,1] / g_pv[doses == 0]
            norm_image[:,:,2] = image[:,:,2] / b_pv[doses == 0]
        elif self.calibration_curve_type == "Devic":
            pv_to_dose = FilmCalibration.devic_pv_to_dose
            norm_image[:,:,0] = np.abs(image[:,:,0] - r_pv[doses == 0])
            norm_image[:,:,1] = np.abs(image[:,:,1] - g_pv[doses == 0])
            norm_image[:,:,2] = np.abs(image[:,:,2] - b_pv[doses == 0])
        else:
            raise ValueError("Invalid calibration curve type")
        return (pv_to_dose, norm_image)

    @staticmethod
    def lewis_dose_to_pv(d, a, b, c):
        r"""
        Calculates the PV value for a given dose using the Lewis model.

        Args:
        d (float): The dose value.
        a (float): The a parameter of the Lewis model.
        b (float): The b parameter of the Lewis model.
        c (float): The c parameter of the Lewis model.

        Returns:
        float: The PV value calculated using the Lewis model.
        """
        return a + b/(d + c)

    @staticmethod
    def lewis_pv_to_dose(pv, a, b, c):
        r"""
        Converts a given PV value to a dose value using the Lewis formula.

        Args:
        pv (float): The PV value to convert to dose.
        a (float): The 'a' parameter of the Lewis formula.
        b (float): The 'b' parameter of the Lewis formula.
        c (float): The 'c' parameter of the Lewis formula.

        Returns:
        float: The dose value corresponding to the given PV value.
        """
        return -c + b/(pv - a)

    @staticmethod
    def devic_pv_to_dose(pv, a, b):
        r"""
        Converts a pixel value (PV) to a dose value using the Devic formula.

        Args:
        pv (float): Pixel value
        a (float): Calibration coefficient
        b (float): Calibration coefficient

        Returns:
        float: Dose value
        """
        return (a*pv)/(1 - b*pv)

    @staticmethod
    def devic_dose_to_pv(d, a, b):
        r"""
        Converts a measured dose value to a pixel value using the Devic formula.

        Args:
        d (float): The measured dose value.
        a (float): The a parameter of the Devic formula.
        b (float): The b parameter of the Devic formula.

        Returns:
        float: The pixel value corresponding to the measured dose value.
        """
        return d/(a + b*d)

    def convert_image_to_dose(self, image:np.array):
        r"""
        Converts an image to a dose map using the calibration curve parameters.

        Args:
        image (np.array): The input image to convert, with shape (height, width, 3).

        Returns:
        np.array: The dose map generated from the input image.

        Throws: 
        ValueError: If the image does not have 3 channels.
        """
        if image.shape[2] !=3:
            raise ValueError("Image must have 3 channels")
        pv_to_dose, normed_image= self.normalize_image_by_calibration_type(image)
        dose = np.zeros(image.shape)
        dose[:,:,0] = pv_to_dose(normed_image[:,:,0], *self.calibration_curve['r_opt'])
        dose[:,:,1] = pv_to_dose(normed_image[:,:,1], *self.calibration_curve['g_opt'])
        dose[:,:,2] = pv_to_dose(normed_image[:,:,2], *self.calibration_curve['b_opt'])
        return dose

    def save_calibration_object(self):
        r"""
        Saves the calibration object to a file using the pickle module.

        Returns:
        None
        """
        f = fd.asksaveasfile(mode='wb',
        defaultextension=".cal", initialdir=self.calibration_file_directory, title='Save calibration object', confirmoverwrite=True)
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def open_calibration_object(self):
        r"""
        Opens the calibration object file and updates the current object's attributes with the loaded object's attributes.

        Returns:
        None
        """
        with open(self.calibration_object_file_path, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
            self.__dict__.update(pickle.load(f).__dict__)

    @staticmethod
    def create_or_load_calibration_object():
        r"""
        Prompts the user to create a new calibration or load an existing calibration.
        If the user chooses to create a new calibration, a new FilmCalibration object is created,
        configured, and saved. The calibration curve is also plotted.
        If the user chooses to load an existing calibration, the user is prompted to select a saved
        calibration file. The selected file is then loaded into a new FilmCalibration object, and
        the calibration curve is plotted.
        Raises a ValueError if the user enters an invalid input.
        """
        root = tk.Tk()
        root.withdraw()
        print("Please enter N to create a new calibration or L to load an existing calibration:")
        load_or_new_calibration_str = input()
        film_calibration = FilmCalibration()
        if load_or_new_calibration_str == "N":
            film_calibration.configure_calibration()
            film_calibration.create_calibration_curve()
            film_calibration.save_calibration_object()
            root.destroy()
        elif load_or_new_calibration_str == "L":
            calibration_object_file_path = fd.askopenfilename(
            parent=root, initialdir="$HOME", title='Select saved calibration file')
            #print(calibration_object_file_path)
            film_calibration.calibration_object_file_path = calibration_object_file_path
            film_calibration.open_calibration_object()
            root.destroy()
        else:
            raise ValueError("Invalid input. Please enter N or L.")
        film_calibration.display_calibration_films()

def test_create_lewis_calibration_curve():
    test_curve_type = "Lewis"
    test_doses = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    test_r_pv = np.power(2.0, -test_doses)
    test_g_pv = np.copy(test_r_pv)
    test_b_pv = np.copy(test_r_pv)
    test_r_std = np.ones(test_r_pv.shape)
    test_g_std = np.ones(test_g_pv.shape)
    test_b_std = np.ones(test_b_pv.shape)
    test_calibration_curve = CalibrationCurve(test_doses, test_r_pv, test_g_pv, test_b_pv, test_r_std, test_g_std, test_b_std, test_curve_type)
    fit_ground_truth = np.array([-0.27395561,  1.85194717,  1.45033183]) #ground truth extracted using a separate python script
    assert np.allclose(test_calibration_curve.r_opt, fit_ground_truth, rtol=1e-6)
    assert np.allclose(test_calibration_curve.g_opt, fit_ground_truth, rtol=1e-6)
    assert np.allclose(test_calibration_curve.b_opt, fit_ground_truth, rtol=1e-6)

def test_create_devic_calibration_curve():
    test_curve_type = "Devic"
    test_doses = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    test_r_pv = np.power(2.0, -test_doses)
    test_g_pv = np.copy(test_r_pv)
    test_b_pv = np.copy(test_r_pv)
    test_r_std = np.ones(test_r_pv.shape)
    test_g_std = np.ones(test_g_pv.shape)
    test_b_std = np.ones(test_b_pv.shape)
    test_calibration_curve = CalibrationCurve(test_doses, test_r_pv, test_g_pv, test_b_pv, test_r_std, test_g_std, test_b_std, test_curve_type)
    fit_ground_truth = np.array([1.14490056, 0.78425477]) #ground truth extracted using a separate python script
    assert np.allclose(test_calibration_curve.r_opt, fit_ground_truth, rtol=1e-6)
    assert np.allclose(test_calibration_curve.g_opt, fit_ground_truth, rtol=1e-6)
    assert np.allclose(test_calibration_curve.b_opt, fit_ground_truth, rtol=1e-6)

def test_load_calibration_films():
    test_calibration_films_path = "../data_test/test_calibration_films/"
    pixel_range = 65535. #2^32 -1 
    test_file_dict = dict() #dict maps dose to array of file names
    test_file_dict[0] = ["0Gy062.tif", "0Gy063.tif"]
    test_file_dict[0.25] = ["0_25_1Gy058.tif", "0_25_1Gy059.tif", "0_25_2Gy060.tif", "0_25_2Gy061.tif"]
    test_file_dict[0.5] = ["0_5_1Gy053.tif", "0_5_1Gy054.tif", "0_5_2Gy056.tif", "0_5_2Gy057.tif"]
    test_file_dict[1] = ["1_1Gy049.tif", "1_1Gy050.tif", "1_2Gy051.tif", "1_2Gy052.tif"]
    test_file_dict[1.5] = ["1_5_1Gy045.tif", "1_5_1Gy046.tif", "1_5_2Gy047.tif", "1_5_2Gy048.tif"]
    test_file_dict[2] = ["2Gy043.tif", "2Gy044.tif"]
    test_file_dict[2.5] = ["2_5Gy041.tif", "2_5Gy042.tif"]
    test_file_dict[3] = ["3Gy039.tif", "3Gy040.tif"]
    test_file_dict[4] = ["4Gy037.tif", "4Gy038.tif"]
    test_file_dict[5] = ["5Gy035.tif", "5Gy036.tif"]
    test_file_dict[6] = ["6Gy033.tif", "6Gy034.tif"]
    test_file_dict[7] = ["7Gy031.tif", "7Gy032.tif"]
    test_file_dict[8] = ["8Gy029.tif", "8Gy030.tif"]
    test_file_dict[9] = ["9Gy027.tif", "9Gy028.tif"]
    test_file_dict[10] = ["10Gy025.tif", "10Gy026.tif"]
    test_file_dict[12] = ["12Gy023.tif", "12Gy024.tif"]
    test_file_dict[14] = ["14Gy021.tif", "14Gy022.tif"] #ignoring _end files, of unknown purpose
    test_file_dict[16] = ["16Gy019.tif", "16Gy020.tif"]
    test_file_dict[18] = ["18Gy017.tif", "18Gy018.tif"]
    test_file_dict[20] = ["20Gy015.tif", "20Gy016.tif"]
    test_file_dict[24] = ["24Gy013.tif", "24Gy014.tif"]
    test_file_dict[28] = ["28Gy011.tif", "28Gy012.tif"]
    test_file_dict[32] = ["32Gy008.tif", "32Gy009.tif"]
    test_file_dict[36] = ["36Gy006.tif", "36Gy007.tif"]
    test_file_dict[40] = ["40Gy004.tif", "40Gy005.tif"]
    test_calibration = FilmCalibration()
    test_calibration.pixel_range = pixel_range
    test_calibration.calibration_curve_type = "Lewis"
    for d in test_file_dict.keys():
        test_calibration.add_calibration_files_for_dose(test_file_dict[d])
    test_calibration.load_calibration()
    test_calibration.create_calibration_curve()
    r_opt_ground_truth = np.array([0.09951261, 2.66066357, 3.07706001]) 
    g_opt_ground_truth = np.array([0.05982643, 5.28318534, 5.74549658])
    b_opt_ground_truth = np.array([ 0.16486698, 10.10602592, 12.20989423])
    assert np.allclose(test_calibration.calibration_curve.r_opt, r_opt_ground_truth, rtol=1e-6)
    assert np.allclose(test_calibration.calibration_curve.g_opt, g_opt_ground_truth, rtol=1e-6)
    assert np.allclose(test_calibration.calibration_curve.b_opt, b_opt_ground_truth, rtol=1e-6)


def test_create_calibration_curve_from_calibration_films():
    pass

def test_load_calibraton_object():
    pass

def main():
    r"""
    This main function creates or loads a calibration object for film calibration.

    Returns:
    None
    """
    FilmCalibration.create_or_load_calibration_object()
    plt.show()


if __name__ == "__main__":
    main()
