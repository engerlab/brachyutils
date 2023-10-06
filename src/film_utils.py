import dose_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tifffile
from matplotlib.ticker import FormatStrFormatter
import tkinter as tk
from tkinter import filedialog as fd

import pymedphys 
from scipy import ndimage

import cv2

class FilmMeasurement:
    r"""
    to write
    """

    pixel_range:int
    #callibration_folder:str
    callibration_file_dict:dict[float, tuple[str, ...]]
    callibration_images:dict[float, np.ndarray]
    callibration_curve_type:str

    def __init__(self):
        pixel_range = None
        callibration_curve_type = None
        #to write
        pass 

    def configure_callibration(self):
        possible_callibrations = ["Lewis", "Devic"]
        while(callibration_curve_type not in possible_callibrations):
            print("Choose callibration curve type (Lewis or Devic):")
            self.callibration_curve_type = input()
        while(pixel_range is not int):
            print("Enter the number of possible pixel values (e.g. 2^16 for 16 bit images):")
            pixel_range_str = input()
            try:
                pixel_range = int(pixel_range_str)
            except ValueError:
                print("Invalid input. Please enter an integer.")

    def add_callibration_files_for_dose(self, dose:float):
        root = tk.Tk()
        root.withdraw()
        filenames_for_dose = fd.askopenfilenames(parent=root, title='Choose callibration films corresponding to a dose of ' + dose + ' Gy')
        self.callibration_file_dict[dose] = filenames_for_dose
        

    def load_callibration(self): 
        if callibration_folder is None: 
            raise ValueError("Path for callibration films not set")
        elif callibration_file_dict is None:
            raise ValueError("Callibration file dictionary not set")
        else: 
            self.callibration_images = {j: np.mean([tifffile.imread(i) for i in self.callibration_file_dict[j]], axis=0 ) / self.pixel_range for j in self.callibration_file_dict.keys()}

    def plot_callibration(self): 
        if(self.callibration_images is None): 
            raise ValueError("Callibration images not loaded")
        else: 
            fig, axes = plt.subplots(4, 6, dpi = 124, figsize=(15,10))
            for i in self.callibration_images.keys(): 
                plt.plot(self.callibration_images[i].flatten(), label=str(i))
            plt.legend()
            plt.show()