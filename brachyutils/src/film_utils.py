import pickle
import tkinter as tk
from tkinter import filedialog as fd
import sys
import os
import dose_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tifffile
from matplotlib.ticker import FormatStrFormatter
import pymedphys
from scipy import ndimage
from scipy.optimize import curve_fit
import cv2


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
        self.calibration_curve_params: dict = None
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
        for dose in doses:
            r_pv = np.array([self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3], self.roi_bounds[0]:self.roi_bounds[2], 0].mean() for dose in doses])
            g_pv = np.array([self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3], self.roi_bounds[0]:self.roi_bounds[2], 1].mean() for dose in doses])
            b_pv = np.array([self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3], self.roi_bounds[0]:self.roi_bounds[2], 2].mean() for dose in doses])

            r_std = np.array([self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3], self.roi_bounds[0]:self.roi_bounds[2], 0].std() for dose in doses])
            g_std = np.array([self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3], self.roi_bounds[0]:self.roi_bounds[2], 1].std() for dose in doses])
            b_std = np.array([self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3], self.roi_bounds[0]:self.roi_bounds[2], 2].std() for dose in doses])

        #print('red channel:', rstd)
        #print('green channel:', gstd)
        #print('blue channel:', bstd)
        dose_to_pv, pv_to_dose, r_pv_fit, g_pv_fit, b_pv_fit = self.define_fit_parameters_for_calibration_type(doses, r_pv, g_pv, b_pv)

        r_opt, rp_cov = curve_fit(dose_to_pv, doses, r_pv_fit, sigma=r_std)
        g_opt, gp_cov = curve_fit(dose_to_pv, doses, g_pv_fit, sigma=g_std)
        b_opt, bp_cov = curve_fit(dose_to_pv, doses, b_pv_fit, sigma=b_std)
        
        self.calibration_curve_params = {'doses': doses, 'r_pv': r_pv, 'g_pv' : g_pv, 'b_pv': b_pv, 'r_opt': r_opt, 'rp_cov': rp_cov, 'g_opt': g_opt, 'gp_cov': gp_cov, 'b_opt': b_opt, 'bp_cov': bp_cov}

    def define_fit_parameters_for_calibration_type(self, doses, r_pv, g_pv, b_pv):
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
            dose_to_pv = FilmCalibration.lewis_dose_to_pv
            pv_to_dose = FilmCalibration.lewis_pv_to_dose
            # normalize to the zero dose pv for Lewis
            #print(r_pv, doses==0)
            r_pv_fit = r_pv/(r_pv[doses == 0])
            g_pv_fit = g_pv/(g_pv[doses == 0])
            b_pv_fit = b_pv/(b_pv[doses == 0])

        elif self.calibration_curve_type == "Devic":
            dose_to_pv = FilmCalibration.devic_dose_to_pv
            pv_to_dose = FilmCalibration.devic_pv_to_dose
            # Devic uses net pvs
            r_pv_fit = np.abs(np.array(r_pv) - r_pv[doses == 0])
            g_pv_fit = np.abs(np.array(g_pv) - r_pv[doses == 0])
            b_pv_fit = np.abs(np.array(b_pv) - r_pv[doses == 0])

        return (dose_to_pv, pv_to_dose, r_pv_fit, g_pv_fit, b_pv_fit)

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
        doses = self.calibration_curve_params['doses']
        r_pv = self.calibration_curve_params['r_pv']
        g_pv = self.calibration_curve_params['g_pv']
        b_pv = self.calibration_curve_params['b_pv']
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

    def plot_calibration_and_response_curve(self):
        """
        Plots the calibration and response curves for the film dosimetry system.
        The calibration curve relates the measured pixel values (PV) to the actual
        radiation dose delivered to the film. The response curve relates the
        normalized PV to the actual radiation dose delivered to the film.
        """

        doses = self.calibration_curve_params['doses']
        r_pv = self.calibration_curve_params['r_pv']
        g_pv = self.calibration_curve_params['g_pv']
        b_pv = self.calibration_curve_params['b_pv']
        r_opt = self.calibration_curve_params['r_opt']
        g_opt = self.calibration_curve_params['g_opt']
        b_opt = self.calibration_curve_params['b_opt']
        
        dose_to_pv, pv_to_dose, r_pv_fit, g_pv_fit, b_pv_fit = self.define_fit_parameters_for_calibration_type(doses, r_pv, g_pv, b_pv)

        fig, ax = plt.subplots(2, 1, dpi=150, sharex=True)
        # Remove space so awesome sharex visual
        fig.subplots_adjust(hspace=0.03)
        ax[0].plot(doses, r_pv, 'r', label="Red")
        ax[0].plot(doses, g_pv, 'g', label="Green")
        ax[0].plot(doses, b_pv, 'b', label="Blue")
        yl1 = ax[0].set_ylabel("PV", labelpad=25)
        yl1.set_rotation(0)
        ax[0].grid()
        ax[0].legend()

        # Plot derivative to highlight usable ranges per channel
        d_dose = (np.array(doses)[:-1] + np.array(doses)[1:]) / 2
        d_r_pv = np.diff(r_pv) / np.diff(doses)
        d_g_pv = np.diff(g_pv) / np.diff(doses)
        d_b_pv = np.diff(b_pv) / np.diff(doses)
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

        doses = np.array(list(self.calibration_images.keys()))
        dose_array = np.linspace(0, 40, 100)
        r_pv_array = np.linspace(0.12, 1, 100)
        g_pv_array = np.linspace(0.15, 1, 100)
        b_pv_array = np.linspace(0.30, 1, 100)

        fig, ax = plt.subplots(1, 3, dpi=150, figsize=(20, 5))

        legend_elements = [Line2D([0], [0], marker='.', markersize=10, markerfacecolor='black', color='w', label='Experiment'),
                            Line2D([0], [0], color='black', label='Fit')]

        # Response Curve
        ax[0].plot(doses, r_pv_fit, 'r.')
        ax[0].plot(doses, g_pv_fit, 'g.')
        ax[0].plot(doses, b_pv_fit, 'b.')

        ax[0].plot(dose_array, dose_to_pv(
            dose_array, *r_opt), 'r', label="Red fit")
        ax[0].plot(dose_array, dose_to_pv(
            dose_array, *g_opt), 'g', label="Green fit")
        ax[0].plot(dose_array, dose_to_pv(
            dose_array, *b_opt), 'b', label="Blue fit")

        ax[0].grid()
        ax[0].set_xlabel("Dose (Gy)")
        ax[0].set_ylabel("Normalized PV")
        ax[0].legend(loc='best', handles=legend_elements)
        ax[0].set_title('Response Curve')

        # Calibration Curve
        ax[1].plot(r_pv_fit, doses, 'r.')
        ax[1].plot(g_pv_fit, doses, 'g.')
        ax[1].plot(b_pv_fit, doses, 'b.')

        ax[1].plot(r_pv_array, pv_to_dose(r_pv_array, *r_opt), 'r')
        ax[1].plot(g_pv_array, pv_to_dose(g_pv_array, *g_opt), 'g')
        ax[1].plot(b_pv_array, pv_to_dose(b_pv_array, *b_opt), 'b')

        ax[1].grid()
        ax[1].set_xlabel("Normalized PV")
        ax[1].set_ylabel("Dose (Gy)")
        ax[1].legend(loc='best', handles=legend_elements)
        ax[1].set_title('Calibration Curve')

        # Dose Residual Plot
        ax[2].plot(doses, 100*(doses - pv_to_dose(r_pv_fit, *r_opt))/doses, 'r.')
        ax[2].plot(doses, 100*(doses - pv_to_dose(g_pv_fit, *g_opt))/doses, 'g.')
        ax[2].plot(doses, 100*(doses - pv_to_dose(b_pv_fit, *b_opt))/doses, 'b.')

        ax[2].axhline(-2, linestyle='--')
        ax[2].axhline(2, linestyle='--')

        ax[2].set_ylabel("Given Dose - Measured Dose (%)")
        ax[2].set_xlabel("Dose (Gy)")
        ax[2].grid()
        ax[2].set_title('Dose Error')

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
        dose[:,:,0] = pv_to_dose(normed_image[:,:,0], *self.calibration_curve_params['r_opt'])
        dose[:,:,1] = pv_to_dose(normed_image[:,:,1], *self.calibration_curve_params['g_opt'])
        dose[:,:,2] = pv_to_dose(normed_image[:,:,2], *self.calibration_curve_params['b_opt'])
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
        film_calibration.plot_calibration_and_response_curve()

    

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
