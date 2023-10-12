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
from scipy.optimize import curve_fit


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
        self.calibration_curve_params = None
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

    def set_roi_bounds(self, new_bounds:tuple):
        if(len(new_bounds) != 4): 
            raise ValueError("ROI bounds must be a tuple of length 4")
        for i in new_bounds:
            if type(i) is not int:
                raise ValueError("ROI bounds must be a tuple of integers")
        self.roi_bounds = new_bounds

    def create_calibration_curve(self): 
        doses = np.array(list(self.calibration_images.keys()))
        rPV=[]
        gPV=[]
        bPV=[]

        rSTD=[]
        gSTD=[]
        bSTD=[]

        #populate arrays with mean pixel value in ROI as a function of dose
        for dose in doses:
            rPV.append((self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3],self.roi_bounds[0]:self.roi_bounds[2],0]).mean())
            gPV.append((self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3],self.roi_bounds[0]:self.roi_bounds[2],1]).mean())
            bPV.append((self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3],self.roi_bounds[0]:self.roi_bounds[2],2]).mean())

            rSTD.append((self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3],self.roi_bounds[0]:self.roi_bounds[2],0]).std())
            gSTD.append((self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3],self.roi_bounds[0]:self.roi_bounds[2],1]).std())
            bSTD.append((self.calibration_images[dose][self.roi_bounds[1]:self.roi_bounds[3],self.roi_bounds[0]:self.roi_bounds[2],2]).std())

        print('red channel:', rSTD)
        print('green channel:', gSTD)
        print('blue channel:', bSTD)

        fig, ax = plt.subplots(2,1,dpi=150,sharex=True)
        fig.subplots_adjust(hspace=0.03) # Remove space so awesome sharex visual
        ax[0].plot(doses,rPV,'r',label="Red")
        ax[0].plot(doses,gPV,'g',label="Green")
        ax[0].plot(doses,bPV,'b',label="Blue")
        yL1 = ax[0].set_ylabel("PV", labelpad=25)
        yL1.set_rotation(0)
        ax[0].grid()
        ax[0].legend()

        # Plot derivative to highlight usable ranges per channel
        dDose = (np.array(doses)[:-1] + np.array(doses)[1:]) / 2
        drPV = np.diff(rPV) / np.diff(doses)
        dgPV = np.diff(gPV) / np.diff(doses)
        dbPV = np.diff(bPV) / np.diff(doses)
        ax[1].plot(dDose,-drPV, 'r',label="Red")
        ax[1].plot(dDose,-dgPV, 'g',label="Green")
        ax[1].plot(dDose,-dbPV, 'b',label="Blue")
        ax[1].set_xlabel("Dose (Gy)")
        yL2 = ax[1].set_ylabel(r"$-\frac{\partial{PV}}{\partial Dose}$", labelpad=25)
        yL2.set_rotation(0)
        ax[1].set_yscale('log')
        ax[1].grid()
        ax[1].legend()

        if self.calibration_curve_type == "Lewis":
            dose2PV = Lewis_dose2PV
            #normalize to the zero dose PV for Lewis
            rPV_fit = rPV/rPV[rPV == 0]
            gPV_fit = gPV/gPV[rPV == 0]
            bPV_fit = bPV/bPV[rPV == 0]

        elif self.calibration_curve_type == "Devic": 
            dose2PV = Devic_dose2PV
            #Devic uses net PVs
            rPV_fit = np.abs(np.array(rPV) - rPV[rPV == 0])
            gPV_fit = np.abs(np.array(gPV) - rPV[rPV == 0])
            bPV_fit = np.abs(np.array(bPV) - rPV[rPV == 0])

        rOpt, rPcov = curve_fit(dose2PV, doses, rPV_fit, sigma=rSTD)
        gOpt, gPcov = curve_fit(dose2PV, doses, gPV_fit, sigma=gSTD)
        bOpt, bPcov = curve_fit(dose2PV, doses, bPV_fit, sigma=bSTD)

        dose_array = np.linspace(0,40,100)
        rPV_array = np.linspace(0.12,1,100)
        gPV_array = np.linspace(0.15,1,100)
        bPV_array = np.linspace(0.30,1,100)

        fig, ax = plt.subplots(1,3, dpi=150, figsize=(20,5))

        legend_elements = [Line2D([0], [0], marker='.', markersize=10, markerfacecolor='black', color='w', label='Experiment'),
                        Line2D([0], [0], color='black', label='Fit')]

        # Response Curve
        ax[0].plot(doses,rPV_norm,'r.')
        ax[0].plot(doses,gPV_norm,'g.')
        ax[0].plot(doses,bPV_norm,'b.')

        ax[0].plot(dose_array, dose2PV(dose_array, *rOpt), 'r', label="Red fit")
        ax[0].plot(dose_array, dose2PV(dose_array, *gOpt), 'g', label="Green fit")
        ax[0].plot(dose_array, dose2PV(dose_array, *bOpt), 'b', label="Blue fit")

        ax[0].grid()
        ax[0].set_xlabel("Dose (Gy)")
        ax[0].set_ylabel("Normalized PV")
        ax[0].legend(loc='best', handles=legend_elements)
        ax[0].set_title('Response Curve')

        # Calibration Curve
        ax[1].plot(rPV_norm,doses,'r.')
        ax[1].plot(gPV_norm,doses,'g.')
        ax[1].plot(bPV_norm,doses,'b.')

        ax[1].plot(rPV_array, PV2dose(rPV_array, *rOpt), 'r')
        ax[1].plot(gPV_array, PV2dose(gPV_array, *gOpt), 'g')
        ax[1].plot(bPV_array, PV2dose(bPV_array, *bOpt), 'b')

        ax[1].grid()
        ax[1].set_xlabel("Normalized PV")
        ax[1].set_ylabel("Dose (Gy)")
        ax[1].legend(loc='best', handles=legend_elements)
        ax[1].set_title('Calibration Curve')

        # Dose Residual Plot
        ax[2].plot(doses, 100*(doses - PV2dose(rPV_norm, *rOpt))/doses, 'r.')
        ax[2].plot(doses, 100*(doses - PV2dose(gPV_norm, *gOpt))/doses, 'g.')
        ax[2].plot(doses, 100*(doses - PV2dose(bPV_norm, *bOpt))/doses, 'b.')

        ax[2].axhline(-2, linestyle='--')
        ax[2].axhline(2, linestyle='--')


        ax[2].set_ylabel("Given Dose - Measured Dose (%)")
        ax[2].set_xlabel("Dose (Gy)")
        ax[2].grid()
        ax[2].set_title('Dose Error')

    def Lewis_dose2PV(D, a, b, c):
        return a + b/(D + c)

    def Lewis_PV2dose(PV, a, b, c):     
        return -c + b/(PV - a)

    def Devic_PV2dose(PV, a, b):      
        return (a*PV)/(1 - b*PV)

    def Devic_dose2PV(D, a, b):
        return D/(a + b*D)

def main(): 
    film = FilmMeasurement()
    film.configure_calibration()
    film.create_calibration_curve()
    plt.show()

if __name__ == "__main__":
    main()