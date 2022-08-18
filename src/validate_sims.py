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
        3. result of gamma index analysis 
"""

# needed libraries
import string
from typing import Any
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
import pandas as pd

def _test_QA_TG186_init(with_return=False):
        test_path = "/home/majd/data/TG186 Vallidation/rapidBrachyMCTPS/RapidBrachyMCTPS_Merged_DoseComparison/alana_newmuens_combined.3ddose"
        groundTruth_path = "/home/majd/data/TG186 Vallidation/rapidBrachyMCTPS/RapidBrachyMCTPS_Merged_DoseComparison/merged_combined.3ddose"
        qa_object = QA_TG186(test_path, groundTruth_path)
        if with_return:
                return qa_object

def _test_QA_TG186_dose_ratio():
        qa_object = _test_QA_TG186_init(True)
        qa_object.dose_ratio()
        print(qa_object.results_df)
        
def _test_QA_TG186_dose_percent_error():
        qa_object = _test_QA_TG186_init(True)
        qa_object.dose_percent_error()
        print(qa_object.results_df['mean_dose_ratio'])

def _test_QA_TG186_gamma_index():
        qa_object = _test_QA_TG186_init(True)
        qa_object.gamma_index()
        print(qa_object.results_df['mean_gamma_matrix'])
        print(qa_object.plot_results())
def _test_QA_TG186_run_QA():
        qa_object = _test_QA_TG186_init(True)
        qa_object.run_QA()


class QA_TG186:
        """ an object containing the following attributes:
        - two dose objects:
                - test dose: 3ddose or dicom format
                - ground truth dose: 3ddose or dicom format

        - functions that compare the equality of the dose objects:
                - dose ratio analysis
                - percent error analysis
                - gamma variate analysis
                - dose point analysis
                - dose line profile analysis

        - functions that visualize the result of the analysis
                - scroll dose
                - tri-planar snapshot (views transverse, sagital and coronal views of a 3D matrix)
        - pandas data frame that contains the result of the analysis
        """
        test_dose = None
        groundTruth_dose = None
        test_dose_axes = None
        groundTruth_dose_axes = None
        results_df = {}
        
        def __init__(self, path2testDose:string, path2GroundTruth:string) -> None:
                # load test dose
                test_extension = path2testDose.split('.')[-1]
                if test_extension == "3ddose":
                        dose_dictionary = dose_utils.load_3ddose(path2testDose)
                        self.test_dose = dose_dictionary['grid']
                        self.test_dose_axes = dose_dictionary['axis']
                        testSimLoadSuccess(self.test_dose)
                elif test_extension == "dcm":
                        self.test_dose = dose.DoseGrid(path2GroundTruth).dose_grid
                        testDicomLoadSuccess(self.test_dose)
                else:
                        raise Exception("input testing dose must have 3ddose or DICOM RD format")

                # load ground truth dose
                groundTruth_extension = path2GroundTruth.split('.')[-1]
                if groundTruth_extension == "3ddose":
                        dose_dictionary = dose_utils.load_3ddose(path2GroundTruth)
                        self.groundTruth_dose = dose_dictionary['grid']
                        self.groundTruth_dose_axes = dose_dictionary['axis']
                        testSimLoadSuccess(self.groundTruth_dose)
                elif groundTruth_extension == "dcm":
                        self.groundTruth_dose = dose.DoseGrid(path2GroundTruth)
                        testDicomLoadSuccess(self.groundTruth_dose).dose_grid
                else:
                        raise Exception("input testing dose must have 3ddose or DICOM RD format")


        # these tests compare 2 dose matrices. the format of the metrices should be Z,Y,X
        
        def dose_ratio(self):
                '''get elemetwise ratio between the test dose and the ground truth dose. add the results to the results_df'''
                dose_ratio =  self.test_dose/self.groundTruth_dose
                self.results_df["dose_ratio"] = dose_ratio
                self.results_df['max_dose_ratio']=np.nanmax(dose_ratio)
                self.results_df['mean_dose_ratio']=np.nanmean(dose_ratio)
                self.results_df['std_dose_ratio']=np.nanstd(dose_ratio)
        
        def dose_percent_error(self):
                '''get elementwise percent error between the dose and the ground truth dose. add the results to the results_df'''
                dose_PE= np.abs(self.test_dose - self.groundTruth_dose)/self.groundTruth_dose * 100
                self.results_df["dose_percent_error"] = dose_PE
                self.results_df['max_dose_percent_error']=np.nanmax(dose_PE)
                self.results_df['mean_dose_percent_error']=np.nanmean(dose_PE)
                self.results_df['std_dose_percent_error']=np.nanstd(dose_PE)
        
        def gamma_index(self):
                '''get gamma matrix between the dose and the ground truth dose. add the results to the results_df'''
                gamma_matrix = pymedphys.gamma(self.groundTruth_dose_axes, self.groundTruth_dose, self.test_dose_axes, self.test_dose, 20., 2.)
                self.results_df["gamma_matrix"] = gamma_matrix
                self.results_df['max_gamma_matrix']=np.nanmax(gamma_matrix)
                self.results_df['mean_gamma_matrix']=np.nanmean(gamma_matrix)
                self.results_df['std_gamma_matrix']=np.nanstd(gamma_matrix)
        
        def run_QA(self,):
                self.dose_ratio()
                self.dose_percent_error()
                self.gamma_index()

                triPlanar_snapshot(self.results_df['dose_ratio'], "dose ratio")
                triPlanar_snapshot(self.results_df['dose_percent_error'], "dose dose_percent_error")
                triPlanar_snapshot(self.results_df['gamma_matrix'], "dose gamma_matrix")

                scroll_dose.plot_scrollable(self.results_df['dose_ratio'], "dose ratio")
                scroll_dose.plot_scrollable(self.results_df['dose_percent_error'], "dose dose_percent_error")



# a function to test the dose loading from .3ddose was successful
def testSimLoadSuccess(dose):
        try:
                print("Type of the input dose is: ", type(dose))
                print("-----------------")
                print("dimensions of the dose is:", np.shape(dose))
                print("-----------------")
                # print("the average uncertainty of the dose is:", dose_utils.get_average_uncert(dose))
                # print("-----------------")
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
                print("Type of the input dose is: ", type(dicom_object))
                print("-----------------")
                print("here is the shape of the dose \n")
                print(dicom_object.shape)
                print("-----------------")
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
def triPlanar_snapshot(matrix, name, unit="(Gy)", source_coord=[0, 0, 0],):
        # get the dimensions of the input matrix and find the coordinates of the center
        dimensions = np.shape(matrix)
        coord_center = np.array([dimensions[2]/2, dimensions[1]/2, dimensions[0]/2], dtype = int)
        snapshot_center = coord_center + source_coord
       
        xy_plane = matrix[snapshot_center[2], :, :]
        xz_plane = matrix[:, snapshot_center[1], :]
        yz_plane = matrix[:, :, snapshot_center[0]]
      
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
        plt.show()
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

def validate_sims(dir_to_dicom, scroll_yes_no, simDoseFile=None, source_coord_in=[0,0,0], do_gamma="no", normalize_dose = "yes"):
        '''THIS FUNCTION IS NOT WORKING AND IS NOT NEEDED'''
        ### First openning DICOM files ###
        dicomDoseFile = glob.glob(dir_to_dicom + "/RD*")[0]
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
#      _test_QA_TG186_init()                            # test passed
        # _test_QA_TG186_dose_ratio()                   # test passed
        # _test_QA_TG186_dose_percent_error()           # test passed
        # _test_QA_TG186_gamma_index()                    # test passed but what does gamma mean???!!
        _test_QA_TG186_run_QA()
