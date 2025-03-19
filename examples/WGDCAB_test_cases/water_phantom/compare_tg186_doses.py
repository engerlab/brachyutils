import numpy as np
from brachyutils import BrachyDose, DoseComparison
from matplotlib import pyplot as plt
import os

TG186_VALIDATION_DIR = '/home/jonathan/Documents/TG186_Validation/'
os.chdir(TG186_VALIDATION_DIR)

def transform_dose_for_case(i, dose_image):
    if i in [0, 1]:
        dose_image.imageArray = np.swapaxes(dose_image.imageArray, 1, 2)
    elif i in [2]:
        pass
    elif i in [3]:
        pass


for i in range(4):
    path_i_rapidbrachy = f'TestCase{i+1}-Elekta/Case-{i+1}-RapidBrachy/testcase{i+1}.nrrd'
    path_i_tg186_dicom_dir = f'TestCase{i+1}-Elekta/Case-{i+1}-OCB/Case-{i+1}-OCB/'
    files = os.listdir(TG186_VALIDATION_DIR + path_i_tg186_dicom_dir)
    rd_file = [f for f in files if f.startswith('RD')][0]
    path_i_tg186_dicom = os.path.join(TG186_VALIDATION_DIR + path_i_tg186_dicom_dir, rd_file)
    dose_i_rapidbrachy = BrachyDose(path_i_rapidbrachy)
    dose_i_tg186 = BrachyDose(path_i_tg186_dicom)
    #print(dose_i_tg186.dose_image is None)

    #print(dose_i_tg186.get_voxel_edges(), dose_i_rapidbrachy.get_voxel_edges())
    transform_dose_for_case(i, dose_i_tg186.dose_image)
    

    gamma_kwargs = {'lower_percent_dose_cutoff': 1,
        'interp_fraction': 10, 'max_gamma': 1.1, 'global_normalisation': 1}
    comparison = DoseComparison(dose_i_tg186, dose_i_rapidbrachy, 1, 1, True, True, TG186_VALIDATION_DIR, gamma_kwargs)
    axis_coords = np.arange(-100, 100.1, 1)
    print("Case", i+1, " Pass Ratio: ", comparison.gamma_pass_ratio * 100, "%")
    comparison.plot_2d_dose_comparison(axis_coords, axis_coords, 0.0, 'xy', (f"Case {i+1} TG-186", f"Case {i+1} RapidBrachyMCTPS") )
    comparison.save_comparison_object("comparison_case_" + str(i+1) + ".comp")

