"""
Date 
        2021/9/9
Purpose
        To compare the dose that was simulated according to TG-186 by MCTPS to the dose that was presented by AAPM as ground truth. 
        This script will loop through the AAPM cases, and validates our dose for each case to the ground truth. 
Author
        Hossein Jafarzadeh
        Enger Lab
        McGill University
Inputs
        a mother directory that contains the dicom files and simulated dose for each case. Shown below:
                -mother_dir
                        |-Case1/
                                |-dicom
                                        RD-...-.dcm
                                |-simResults 
                                        combined.3ddose
                        |-Case2/
                        |-Case3/
                        |etc...
                
Dependencies
        The following external packages:
                1. numpy
                2. glob
                3. matplotlib.pyplot
                4. dicompylercore (needs installation!)

        The following internal packages:
                1. workplace
                2. scroll_dose
                3. dose_utils
Outputs
        1. terminal outputs saying dose loading was successful
        2. two scrollable plots of the dose matrix; one for ground truth dose and one for our simulated dose
        3. result of gamma index analysis (to be added) 
"""

# needed libraries
import dose_utils
import os
import numpy as np
from dicompylercore import dose
import workplace
import scroll_dose
import glob
import matplotlib.pyplot as plt
import pymedphys
import pickle


# a function to test the dose loading from .3ddose was successful
def testSimLoadSuccess(dose):
        try:
                print("Type of the dose is: ", type(dose["grid"]))
                print("-----------------")
                print("dimensions of the dose is:", np.shape(dose["grid"]))
                print("-----------------")
                print("the average uncertainty of the dose is:", dose_utils.get_average_uncert(dose))
                print("-----------------")
                print("3ddose was loaded successfully")
                print("------------------")                
                return 1
        except:
                print("3ddose file did not load \n")
                print("-----------------")
                return 0
# a function to test the loading from dicom files
def testDicomLoadSuccess(dicom_object):
        try:
                print("here is the shape of the dose \n")
                print(dicom_object.shape)
                print("-----------------")
                print("this is the type of the dicom_object:\n")
                print(type(dicom_object))
                print("-----------------")
                # {{for debugging only}}
                # print("here is the first few dose values \n")
                # print(dicom_object.dose_grid[0, 0, 98:101])
                # print("-----------------")
                return 1
        except:
                print("DICOM dose file did not load\n")
                print("-----------------")
                return 0
# this function shows images of the input 3D matrix. the images are 2D planes (transverse, sagital and coronal)
def triPlanar_snapshot(matrix, name, unit="(Gy)", voxelSize=[1,1,1]):
        # get the dimensions of the input matrix and find the coordinates of the center
        dimensions = np.shape(matrix)
        coord_center = np.array([dimensions[0]/2, dimensions[1]/2, dimensions[2]/2], dtype = int)

        xy_plane = matrix[:, :, coord_center[2]]
        xz_plane = matrix[:, coord_center[1], :]
        yz_plane = matrix[coord_center[0], :, :]

        x_axis_tick = np.arange(-1*coord_center[0]*voxelSize[0]+1, coord_center[0]*voxelSize[0], voxelSize[0])
        y_axis_tick = np.arange(-1*coord_center[1]*voxelSize[1]+1, coord_center[1]*voxelSize[1], voxelSize[1])
        z_axis_tick = np.arange(-1*coord_center[2]*voxelSize[2]+1, coord_center[2]*voxelSize[2], voxelSize[2])
        
        # print(str(x_axis_tick))
        # quit()

        # generate a subplot that has 3 maps in a row. the maps are transverse (xy), sagital(yz) and coronal(xz) planes.
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        xy_map = ax1.imshow(xy_plane)
        xz_map = ax2.imshow(xz_plane)
        yz_map = ax3.imshow(yz_plane)
        
        ax1.title.set_text("Transverse Plane")
        ax1.set_xlabel("y-axis")
        ax1.set_ylabel("x-axis")

        ax2.title.set_text("Coronal Plane")
        ax2.set_xlabel("z-axis")
        ax2.set_ylabel("x-axis")

        ax3.title.set_text("Sagital Plane")
        ax3.set_xlabel("z-axis")
        ax3.set_ylabel("y-axis")

        fig.colorbar(xy_map, ax=ax1)
        fig.colorbar(xz_map, ax=ax2)
        fig.colorbar(yz_map, ax=ax3)

        ''' the tick labels are turned into coordinates instead of slide number (this does not work at the moment).
        plt.setp(ax1, xticklabels=str(x_axis_tick), yticklabels=str(y_axis_tick))
        plt.setp(ax2, xticklabels=str(x_axis_tick), yticklabels=str(y_axis_tick))
        plt.setp(ax3, xticklabels=str(x_axis_tick), yticklabels=str(y_axis_tick))
        '''
        fig.suptitle("Tri-Planar Snapshot of "+name+ " at center" + unit, fontsize=20)

        # plt.show()
        return fig


# set directory of the project data
mother_dir = workplace._workplace(workplace.askForLocation())
# simfileDir = simfileDir + "/simResults"
print("------------------")
print("looking at the mother directory: \n")
print(mother_dir)
print("------------------")

# ask what kinds of plots are wanted?
scroll_dicome = input("whould u like to scroll through DICOM dose? yes or no \n")
print("------------------")
scroll_simulated = input("whould u like to scroll through simulated dose? yes or no \n")
print("------------------")
scroll_gamma = input("whould u like to scroll through gamma map? yes or no \n")
print("------------------")
scroll_doseRatio = input("whould u like to scroll through dose ratio between simulated over ground truth map? yes or no \n")
print("------------------")

# a for loop to iterate through dose files in the specified directory
for case in os.listdir(mother_dir):
        if "Case4" in case:
# let's load the ground truth from DICOM files
                dicomDoseFile = glob.glob(mother_dir + "/" + case + "/dicom/RD*")[0]
                print("looking at the DICOM file: \n")
                print(dicomDoseFile)
                print("------------------")
        
                dicom_object = dose.DoseGrid(dicomDoseFile)

# check if loading was successful
                testDicomLoadSuccess(dicom_object)
# plot the dicom dose file
                if scroll_dicome == "yes":
                        scroll_dose.plot_scrollable(dicom_object.dose_grid, "DICOM")
                        # pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))
                        # pickle.dump(scroll_dose.plot_scrollable(dicom_object.dose_grid, "DICOM"), open('doseGroundTruth.pickle', 'wb'))
                
                elif scroll_dicome == "no":
                        print("------------------")
                        print("not scrolling through ground truth dose and moving on")
                else:
                        print("------------------")
                        print("invalid input, rerun the script and pick either yes or no")
                        quit()

# obtain shape of dose in DICOM file to crop the dose in 3ddose file accordingly 
                dicom_shape = np.shape(dicom_object.dose_grid)
                print("here is the shape of the dicom dose file: \n", dicom_shape)
                # dicom axis object
                dicom_axes = tuple(dicom_object.axes)
                # {{for debugging}}
                # print("------------------")    
                # print("here is the axis of the dicom file: \n")
                # print(dicom_axes)
                # print("------------------")    
# # test triplanar
#                 triPlanar_snapshot(dicom_object.dose_grid, "ground truth", "Gy")
#                 quit()
# # ############################################
                simDoseFile = glob.glob(mother_dir + "/" + case + "/simResults/*.3ddose")[0]
                print("looking at the file: \n")
                print(simDoseFile)
                print("------------------")
                             
# extract the dose data out of the .3ddose file
                simDose = dose_utils.load_3ddose(simDoseFile)

# test in the dose loading was successful
                testSimLoadSuccess(simDose)
                
# cropping the dose grid of 3ddose file to match the dose of DICOM file
                a3ddose_shape = np.shape(simDose["grid"])

                # let's get the range of values from size of dicom dose
                crop_out = (a3ddose_shape[0] - dicom_shape[0])/2
                lower_bound = int(crop_out-1)
                upper_bound = int(a3ddose_shape[0]-crop_out-1)
                simDose["grid"] = simDose["grid"][lower_bound:upper_bound, lower_bound:upper_bound, lower_bound:upper_bound]
                
                print("------------------")
                print("here is the size of croped out 3ddose::::: \n", np.shape(simDose["grid"]))

# scroll through the 3ddose files
                if scroll_simulated == "yes":
                        scroll_dose.plot_scrollable(simDose["grid"], "3ddose")
                elif scroll_simulated == "no":
                        print("------------------")
                        print("not scrolling through simulated dose and moving on")
                else:
                        print("------------------")
                        print("invalid input, rerun the script and pick either yes or no")
                        quit()

# time to get % error
                ''' at the moment, %error does not work since the sizes of the arrays do not match
                mean_abs_percent_err = np.mean(np.abs((dicom_object.dose_grid - simDose["grid"])/dicom_object.dose_grid))*100
                print("The mean absolute percent error between the simulations and the ground truth is: \n")
                print(mean_abs_percent_err)
                '''

# let's do Gamma Variate analysis
                gamma_matrix = pymedphys.gamma(dicom_axes, dicom_object.dose_grid, dicom_axes, simDose["grid"], 2., 2.)
                print("------------------")
                print("here is the shape of the gamma_matrix \n", np.shape(gamma_matrix))
                print("------------------")
                print("here is the type of the gamma_matrix \n", type(gamma_matrix))
                if scroll_gamma == "yes":
                        scroll_dose.plot_scrollable(gamma_matrix, "gamma")
                elif scroll_gamma == "no":
                        print("------------------")
                        print("not scrolling through gamma map and moving on")
                else:
                        print("------------------")
                        print("invalid input, rerun the script and pick either yes or no")
                        quit()
   
                print("------------------")
                print("here is the result of gamma 2%/2mm: ", ((gamma_matrix < 1).sum() / len(gamma_matrix)))

# let's get the dose ratios
                doseRatio_sim_dicom = simDose["grid"]/dicom_object.dose_grid
                print("------------------")
                print("here is the shape of the dose ratio \n", np.shape(doseRatio_sim_dicom))
                print("------------------")
                print("here is the type of the gamma_matrix \n", type(doseRatio_sim_dicom))
                if scroll_doseRatio == "yes":
                        scroll_dose.plot_scrollable(doseRatio_sim_dicom, "D simulated/groundTruth")
                elif scroll_doseRatio == "no":
                        print("------------------")
                        print("not scrolling through dose ratio map and moving on")
                else:
                        print("------------------")
                        print("invalid input, rerun the script and pick either yes or no")
                        quit()

# ############################################

# let's bring all the graphs in one place so the users do not have to scroll all the time.
# tri-planaer subplot of the dose distributions
                groundTruth_triplanar = triPlanar_snapshot(dicom_object.dose_grid, "ground truth dose", "(Gy)")
                simulated_triplanar = triPlanar_snapshot(simDose["grid"], "Simulated dose", "(Gy)")
                gamma_triplanar =  triPlanar_snapshot(gamma_matrix, "Gamma Matrix between simulated dose and ground truth", "")
                doseRatio_triplanar = triPlanar_snapshot(doseRatio_sim_dicom, "D(sim/truth)", "")
                plt.show()
        