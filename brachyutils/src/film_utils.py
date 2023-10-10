import dose_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tifffile
from matplotlib.ticker import FormatStrFormatter
import tkinter as tk
from tkinter import filedialog as fd
import sys

import pymedphys 
from scipy import ndimage

import cv2

class FilmMeasurement:
    r"""
    to write
    """

    pixel_range:int
    calibration_file_dict:dict#[float, tuple]
    calibration_images:dict#[float, np.ndarray]
    calibration_curve_type:str

    #calibration_folder:str
    #calibration_file_dict:dict#[float, tuple[str, ...]]

    def __init__(self):
        self.pixel_range = None
        self.calibration_curve_type = None
        self.callibration_file_dict = {}
        self.calibration_images = None

        #to write
        pass 

    def configure_calibration(self):
        possible_calibrations = ["Lewis", "Devic"]
        while(self.calibration_curve_type not in possible_calibrations):
            print("Choose calibration curve type (Lewis or Devic):")
            self.calibration_curve_type = input()
        while(type(self.pixel_range) is not int):
            print("Enter the number of possible pixel values (e.g. 2^16 for 16 bit images):")
            pixel_range_str = input()
            try:
                self.pixel_range = int(pixel_range_str)
            except ValueError:
                print("Invalid input. Please enter an integer.")
            print(type(self.pixel_range))
        print("Begin selecting calibration film files. Enter a dose and select the calibration films at that dose or enter \'Done\' to finish.")
        current_dose_str = ""
        while(current_dose_str != "Done"):
            current_dose_str = input()
            try:
                current_dose = float(current_dose_str)
                self.add_calibration_files_for_dose(current_dose)
            except ValueError:
                print("Invalid input. Please enter a float or \'Done\' to finish.")
        self.load_calibration()
        self.plot_calibration()

    def add_calibration_files_for_dose(self, dose:float):
        root = tk.Tk()
        root.withdraw()
        filenames_for_dose = fd.askopenfilenames(parent=root, title='Choose calibration films corresponding to a dose of ' + str(dose) + ' Gy')
        self.calibration_file_dict[dose] = filenames_for_dose
        

    def load_calibration(self): 
        if calibration_folder is None: 
            raise ValueError("Path for calibration films not set")
        elif calibration_file_dict is None:
            raise ValueError("calibration file dictionary not set")
        else: 
            self.calibration_images = {j: np.mean([tifffile.imread(i) for i in self.calibration_file_dict[j]], axis=0 ) / self.pixel_range for j in self.calibration_file_dict.keys()}

    def plot_calibration(self): 
        if(self.calibration_images is None): 
            raise ValueError("calibration images not loaded")
        else: 
            fig, axes = plt.subplots(4, 6, dpi = 124, figsize=(15,10))
            for i in self.calibration_images.keys(): 
                plt.plot(self.calibration_images[i].flatten(), label=str(i))
            plt.legend()
            plt.show()

def main(): 
    film = FilmMeasurement()
    film.configure_calibration()

if __name__ == "__main__":
    main()