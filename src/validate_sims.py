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
import csv


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
def triPlanar_snapshot(matrix, name, unit="(Gy)", source_coord=[0, 0, 0], voxelSize=[1,1,1], normalize="yes"):
        # get the dimensions of the input matrix and find the coordinates of the center
        dimensions = np.shape(matrix)
        coord_center = np.array([dimensions[0]/2, dimensions[1]/2, dimensions[2]/2], dtype = int)
        snapshot_center = coord_center + source_coord
        if normalize=="yes":
                normalization_point = list(snapshot_center + [0, 10, 0])
                xy_plane = np.swapaxes(matrix[:, :, snapshot_center[2]]/matrix[normalization_point[0]][normalization_point[1]][normalization_point[2]], 0, 1)
                xz_plane = np.swapaxes(matrix[:, snapshot_center[1], :]/matrix[normalization_point[0]][normalization_point[1]][normalization_point[2]], 0, 1)
                yz_plane = np.swapaxes(matrix[snapshot_center[0], :, :]/matrix[normalization_point[0]][normalization_point[1]][normalization_point[2]], 0, 1) 
        else: 
                xy_plane = np.swapaxes(matrix[:, :, snapshot_center[2]], 0, 1)
                xz_plane = np.swapaxes(matrix[:, snapshot_center[1], :], 0, 1)
                yz_plane = np.swapaxes(matrix[snapshot_center[0], :, :], 0, 1)
        # {{for debugging
        # print("**Here is the normalization point**", normalization_point)
        # print("**Here is the dose value at the normalization point**", matrix[normalization_point[0]][normalization_point[1]][normalization_point[2]])
        # # print(np.shape(matrix))
        # print("**Here is the snap shot location**", snapshot_center)
        # quit()
        # }}
        
# x y and z ticks. it does not work, ignore for now or fix it if you can ;) 
        # x_axis_tick = np.arange(-1*snapshot_center[0]*voxelSize[0]+1, snapshot_center[0]*voxelSize[0], voxelSize[0])
        # y_axis_tick = np.arange(-1*snapshot_center[1]*voxelSize[1]+1, snapshot_center[1]*voxelSize[1], voxelSize[1])
        # z_axis_tick = np.arange(-1*snapshot_center[2]*voxelSize[2]+1, snapshot_center[2]*voxelSize[2], voxelSize[2])
        
        # print(str(x_axis_tick))
        # quit()

        # generate a subplot that has 3 maps in a row. the maps are transverse (xy), sagital(yz) and coronal(xz) planes.
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        xy_map = ax1.imshow(xy_plane)
        xz_map = ax2.imshow(xz_plane)
        yz_map = ax3.imshow(yz_plane)
        
        ax1.title.set_text("Transverse Plane")
        ax1.set_xlabel("x-axis")
        ax1.set_ylabel("y-axis")

        ax2.title.set_text("Coronal Plane")
        ax2.set_xlabel("x-axis")
        ax2.set_ylabel("z-axis")

        ax3.title.set_text("Sagital Plane")
        ax3.set_xlabel("y-axis")
        ax3.set_ylabel("z-axis")

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

def crop_dose_matrix(dose_in, wanted_dimensions):
        # only for matrixes with 3 dimensions of equal size
        shape_in = np.shape(dose_in)
        crop_out = ((shape_in[0] - wanted_dimensions[0])/2)
        if crop_out < 1:
                raise("cropped volume should be smaller than the input volume. please check your input and wanted dimensions")

        # get the lower and upper index to crop from the input
        lower_bound = int(crop_out)
        upper_bound = int(shape_in[0]-crop_out)
        return dose_in[lower_bound:upper_bound, lower_bound:upper_bound, lower_bound:upper_bound]

def percent_error(points, sim, gtd):
        points += int(np.shape(sim)[0]/2)
        points = np.ravel_multi_index(points.T, np.shape(sim))
        d_sim = sim.take(points)
        d_gtd = gtd.take(points)
        pe = 100*(d_sim - d_gtd)/d_gtd
        return pe, d_sim, d_gtd

def _test_save_to_csv():
        a = np.arange(0, 12)
        b = np.arange(12, 24)
        c = np.arange(24, 36)
        file = "/home/majd/data/TG186 Vallidation/Elekta/Case1-OCB-MCNP6/simResults/test.csv"
        save_to_csv(file, a, b, c)

def save_to_csv(filename, pe, d_sim, d_gtd):
        rows = zip(pe, d_sim, d_gtd)
        with open(filename, 'w') as f:
                writer = csv.writer(f)
                for row in rows:
                        writer.writerow(row)

def validate_sims(dir_to_case, scroll_yes_no, simDoseFile=None, source_coord_in=[0,0,0], do_gamma="no", normalize_dose = "yes"):
        ### First openning DICOM files ###
        dicomDoseFile = glob.glob(dir_to_case + "/dicom/RD*")[0]
        print("looking at the DICOM file: \n")
        print(dicomDoseFile)
        print("------------------")
        dicom_object = dose.DoseGrid(dicomDoseFile)
        
        # check if loading was successful
        testDicomLoadSuccess(dicom_object)
        
        # scroll through the 
        if scroll_yes_no['scroll_dicome'] == "yes":
                scroll_dose.plot_scrollable(dicom_object.dose_grid, "DICOM")
                # pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))
                # pickle.dump(scroll_dose.plot_scrollable(dicom_object.dose_grid, "DICOM"), open('doseGroundTruth.pickle', 'wb'))
                        
        elif scroll_yes_no['scroll_dicome'] == "no":
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
                # {{for debugging
                # groundTruth_triplanar = triPlanar_snapshot(dicom_object.dose_grid, "ground truth dose", "(Gy)", source_coord=source_coord_in)
                # plt.show()
                # quit()
                # print("------------------")    
                # print("here is the axis of the dicom file: \n")
                # print(dicom_axes)
                # print("------------------")
                # examin_points = np.array([
                #         [-10, 0, 0], [10, 0, 0], [0, -10, 0], [0, 10, 0], [0, 0, -10], [0, 0, 10], [-50, 0, 0], [50, 0, 0], [0, -50, 0], [0, 50, 0], [0, 0, -50], [0, 0, 50]
                # ])
                # print("**here is the percent error!**")
                # print(percent_error(examin_points, dicom_object.dose_grid, dicom_object.dose_grid))  
                # quit()
                # }}
                ### simulated dose files ###
        
        #############################
        if simDoseFile==None:
                simDoseFile = glob.glob(dir_to_case + "/simResults/*.3ddose")[0]
        
        print("looking at the file: \n")
        print(simDoseFile)
        print("------------------")
                                
        # extract the dose data out of the .3ddose file
        simDose = dose_utils.load_3ddose(simDoseFile)
        # test in the dose loading was successful
        testSimLoadSuccess(simDose)

        # dose utils loads the images az zyx, but I want them to be xyz
        simDose["grid"] = np.swapaxes(simDose["grid"], 0, 2)

        # cropping the dose grid of 3ddose file to match the dose of DICOM file
        simDose["grid"] = crop_dose_matrix(simDose['grid'], dicom_shape)              
        print("------------------")
        print("here is the size of croped out 3ddose::::: \n", np.shape(simDose["grid"]))

        # scroll through the 3ddose files
        if scroll_yes_no['scroll_simulated'] == "yes":
                scroll_dose.plot_scrollable(simDose["grid"], "3ddose")
        elif scroll_yes_no['scroll_simulated'] == "no":
                print("------------------")
                print("not scrolling through simulated dose and moving on")
        else:
                print("------------------")
                print("invalid input, rerun the script and pick either yes or no")
                quit()

        # let's do Gamma Variate analysis
        # do_gamma = "no"
        if do_gamma == "yes":
                gamma_matrix = pymedphys.gamma(dicom_axes, dicom_object.dose_grid, dicom_axes, simDose["grid"], 2., 2.)
                print("------------------")
                print("here is the shape of the gamma_matrix \n", np.shape(gamma_matrix))
                print("------------------")
                print("here is the type of the gamma_matrix \n", type(gamma_matrix))
                if scroll_yes_no['scroll_gamma'] == "yes":
                        scroll_dose.plot_scrollable(gamma_matrix, "gamma")
                elif scroll_yes_no['scroll_gamma'] == "no":
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
        if scroll_yes_no['scroll_doseRatio'] == "yes":
                scroll_dose.plot_scrollable(doseRatio_sim_dicom, "D simulated/groundTruth")
        elif scroll_yes_no['scroll_doseRatio'] == "no":
                print("------------------")
                print("not scrolling through dose ratio map and moving on")
        else:
                print("------------------")
                print("invalid input, rerun the script and pick either yes or no")
                quit()

       
        # ############################################


        # let's bring all the graphs in one place so the users do not have to scroll all the time.
        # tri-planaer subplot of the dose distributions
        groundTruth_triplanar = triPlanar_snapshot(dicom_object.dose_grid, "ground truth dose", "(Gy)", source_coord=source_coord_in, normalize=normalize_dose)
        simulated_triplanar = triPlanar_snapshot(simDose["grid"], "Simulated dose", "(Gy)", source_coord=source_coord_in, normalize=normalize_dose)
        if do_gamma=="yes":
                gamma_triplanar =  triPlanar_snapshot(gamma_matrix, "Gamma Matrix between simulated dose and ground truth", "", source_coord=source_coord_in, normalize="no")
        doseRatio_triplanar = triPlanar_snapshot(doseRatio_sim_dicom, "D(sim/truth)", "", source_coord=source_coord_in, normalize="no")

        plt.show()
       
        # ############################################
        # let's get the simulated dose, GTD and the percent error at the specific locations
        examin_points = np.array([
                [-10, 0, 0], [10, 0, 0], [0, -10, 0], [0, 10, 0], [0, 0, -10], [0, 0, 10], [-50, 0, 0], [50, 0, 0], [0, -50, 0], [0, 50, 0], [0, 0, -50], [0, 0, 50]
        ])
        

        return [dicom_object.dose_grid, simDose["grid"], percent_error(examin_points, simDose["grid"], dicom_object.dose_grid)]
        # # {for debugging
        # print("**HERE is the perecent errors**")
        # print(percent_error(examin_points, simDose["grid"], dicom_object.dose_grid))
        # quit()
        # # }

if __name__ =="__main__":
        # set directory of the project data
        # mother_dir = workplace._workplace(workplace.askForLocation())
        mother_dir = "/home/majd/data/TG186 Vallidation/Elekta/"
        print("------------------")
        print("looking at the mother directory: \n")
        print(mother_dir)
        print("------------------")

        # _test_save_to_csv()
        # quit()

        # ask what kinds of plots are wanted?
        scroll_yes_no = {}
        scroll_yes_no['scroll_dicome']=  "no"
        scroll_yes_no['scroll_simulated']= "no"
        scroll_yes_no['scroll_gamma']= "no"
        scroll_yes_no['scroll_doseRatio']= "no"
        # simFileCase1 = (mother_dir+"Case1-OCB-MCNP6/simResults/source_along_z/combined.3ddose")
        # _, _, p_error1 = validate_sims(mother_dir+"Case1-OCB-MCNP6", simFileCase1, scroll_yes_no, [0, 0, 0])
        # save_to_csv(mother_dir+"Case1-OCB-MCNP6/simResults/source_along_z/case1_z.csv", p_error1[0], p_error1[1], p_error1[2])
        
        simFileCase2 = (mother_dir+"Case2-OCB-MCNP6/simResults/source_along_z/combined.3ddose")
        _, _, p_error2 = validate_sims(mother_dir+"Case2-OCB-MCNP6", scroll_yes_no, simFileCase2)
        save_to_csv(mother_dir+"Case2-OCB-MCNP6/simResults/source_along_z/case1_z.csv", p_error2[0], p_error2[1], p_error2[2])

        # validate_sims(mother_dir+"Case3-OCB-MCNP6", scroll_yes_no, [70, 0, 0])
        # validate_sims(mother_dir+"Case4-OCB-MCNP6", scroll_yes_no, [0, 0, 0])


