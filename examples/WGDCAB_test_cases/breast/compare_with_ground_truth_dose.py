from brachyutils import BrachyDose, BrachyDoseComparison
import os
import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np
import copy

def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2.)
    )


os.chdir(os.path.dirname(__file__))
#ref_dose_path = "/home/jonathan/Documents/Breast_DL_Dose_Prediction_Benchmarking/Original_Dataset/Reference_TPS/RD.1.2.246.352.71.7.686590568890.1092.20151125134239.dcm"
ref_dose_path = "/home/jonathan/Documents/Breast_DL_Dose_Prediction_Benchmarking/Original_Dataset/Reference_MC/RD_EGS_Brachy_BreastCase.dcm"
dl_dose_path = "/home/jonathan/Documents/Breast_DL_Dose_Prediction_Benchmarking/DL_MC_Doses/combined_DL_MC_dose.nrrd"
tg43_dose_path = "/home/jonathan/Documents/Breast_DL_Dose_Prediction_Benchmarking/TG43_Dose/combined.seq.nrrd"


ref_dose = BrachyDose(ref_dose_path)
tg43_dose = BrachyDose(tg43_dose_path)
dl_dose_raw = sitk.ReadImage(dl_dose_path)

dl_dose = BrachyDose.dose_with_empty_grid_like(tg43_dose)
dl_dose.dose_image.imageArray = np.swapaxes(sitk.GetArrayFromImage(dl_dose_raw), 0, 2)

trash_dose = copy.deepcopy(dl_dose)
dl_dose_padded = BrachyDose.dose_with_empty_grid_like(ref_dose)
trash_dose.dose_image.resampleOn(ref_dose.dose_image)
dl_dose_padded.dose_image.imageArray = trash_dose.dose_image.imageArray
dl_dose_padded.write_to_nrrd("./dl_benchmark_padded.seq.nrrd")
dl_dose_padded.write_to_3ddose("./dl_benchmark_padded.3ddose")

#resample the ref dose to the smaller one
ref_dose.dose_image.resampleOn(dl_dose.dose_image)




gamma_kwargs = {"global_normalisation": 4.3, "max_gamma": 2}

path = "./dl_benchmark.comp"

path = None

if path is not None:
    comp = BrachyDoseComparison(None, None, None, None, path = path)
else:
    comp = BrachyDoseComparison(ref_dose, dl_dose, 1, 1, True, True, positive_percent_difference=False, percent_difference_range=(-20, 20), gamma_kwargs=gamma_kwargs)

comp.prescription_dose = 4.3

xx = comp.dose1.get_voxel_centers()[0]
yy = comp.dose1.get_voxel_centers()[1]
zz = comp.dose1.get_voxel_centers()[2]

print("GAMMA PASS RATE ", comp.gamma_pass_ratio)
#comp.compute_percent_difference()
comp.plot_2d_dose_comparison(xx, yy, -247., 'xy', ('MC Reference Dose', 'DL Dose'))
comp.save_comparison_object("dl_benchmark.comp")
bins = np.arange(-100, 100, 1)
difference_values = comp.percent_difference.dose_image.imageArray.flatten()
plt.hist(difference_values, bins=bins, density = True)
mean_difference = np.mean(difference_values)
std_difference = np.std(difference_values)
plt.xlabel("Percent Difference")
plt.ylabel("Frequency")
plt.axvline(mean_difference, color='r', linestyle='dashed', linewidth=1, label=fr'$\mu$: {mean_difference:.2f}')
#plt.plot(bins, gaussian(bins, mean_difference, std_difference), ls='solid', color='k')
print(std_difference)
print(np.max(difference_values), np.min(difference_values))
plt.legend()
plt.savefig("dl_benchmark_percent_difference_histogram.png")
plt.show()