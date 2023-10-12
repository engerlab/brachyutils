import dose_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tifffile
from matplotlib.ticker import FormatStrFormatter
import tkinter as tk
from tkinter import filedialog as fd
import sys
import os
import pymedphys 
from scipy import ndimage

import cv2

class FilmMeasurement:
    r"""
    to write
    """



    #calibration_folder:str
    #calibration_file_dict:dict#[float, tuple[str, ...]]

    def __init__(self):
        self.pixel_range:int = None
        self.calibration_file_dict:dict = {}#[float, tuple]
        self.calibration_images:dict = {}#[float, np.ndarray]
        self.calibration_curve_type:str = None
        self.calibration_curve:np.ndarray = None
        self.film_to_mc_offset:tuple = None
        self.roi_bounds:tuple = (100, 100, 150, 150) #x1 y1 x2 y2
        self.calibration_file_directory:str = "$HOME"

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
        print("Begin selecting calibration film files. Enter a dose and select the calibration films at that dose or enter \'Done\' to finish.")
        current_dose_str = input()
        while(current_dose_str != "Done"):
            try:
                current_dose = float(current_dose_str)
                self.add_calibration_files_for_dose(current_dose)
            except ValueError:
                print("Invalid input. Please enter a float or \'Done\' to finish.")
            current_dose_str = input()
        self.load_calibration()
        self.plot_calibration()

    def add_calibration_files_for_dose(self, dose:float):
        root = tk.Tk()
        root.withdraw()
        filenames_for_dose = fd.askopenfilenames(parent=root, initialdir = self.calibration_file_directory, title='Choose calibration films corresponding to a dose of ' + str(dose) + ' Gy')
        if len(filenames_for_dose) > 0:
            self.calibration_file_directory = os.path.dirname(filenames_for_dose[0]) #start navigation of directory of last selected file
            self.calibration_file_dict[dose] = filenames_for_dose 
        root.destroy()
        

    def load_calibration(self): 
        if self.calibration_file_dict is None:
            raise ValueError("calibration file dictionary not set")
        else: 
            self.calibration_images = {j: np.mean([tifffile.imread(i) for i in self.calibration_file_dict[j]], axis=0 ) / self.pixel_range for j in self.calibration_file_dict.keys()}

    def plot_calibration(self): 
        if(self.calibration_images is None): 
            raise ValueError("calibration images not loaded")
        else: 
            nplots = len(self.calibration_images.keys())
            fig, axes = plt.subplots(1, nplots,  dpi = 124, figsize=(15,10), squeeze = True)
            i = 0
            if nplots == 1: 
                plt.imshow(self.calibration_images[list(self.calibration_images.keys())[i]][:,:,0], cmap='trubo')
                plt.title(str(list(self.calibration_images.keys())[i]))
                plt.axvline(self.roi_bounds[0], color='r')
                plt.axhline(self.roi_bounds[1], color='r')
                plt.axvline(self.roi_bounds[2], color='r') 
                plt.axhline(self.roi_bounds[3], color='r')
            else: 
                for ax in axes.flatten():
                    ax.imshow(self.calibration_images[list(self.calibration_images.keys())[i]][:,:,0], cmap='turbo')
                    ax.set_title(str(list(self.calibration_images.keys())[i]))
                    ax.axvline(self.roi_bounds[0], color='r')
                    ax.axhline(self.roi_bounds[1], color='r')
                    ax.axvline(self.roi_bounds[2], color='r') 
                    ax.axhline(self.roi_bounds[3], color='r')
                    i += 1
            plt.show()

    def set_roi_bounds(self, new_bounds:tuple):
        if(len(new_bounds) != 4): 
            raise ValueError("ROI bounds must be a tuple of length 4")
        for i in new_bounds:
            if type(i) is not int:
                raise ValueError("ROI bounds must be a tuple of integers")
        self.roi_bounds = new_bounds

def main(): 
    film = FilmMeasurement()
    film.configure_calibration()

if __name__ == "__main__":
    main()