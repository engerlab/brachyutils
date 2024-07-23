from brachyutils.src.egsphant_utils import BrachyEgsphant
import numpy as np

abs_path = "/home/alana/Documents/RectumPatients/Simulations/Simulations_Win180/P3F2/"
path_eggers = "Final_Balloon/ct.egsphant"

eggers = BrachyEgsphant(abs_path+path_eggers)

print(eggers.axis, eggers.voxel_size, eggers.topleft)
crop_coords = np.array([[-209.2862, 91.8934],[-362.542, -60.051],[77.4847, 245.485]]) / 10.
crop_indices = crop_coords.copy()

for i in range(3):
    origin = eggers.topleft[i]
    for j in range(2):
        crop_indices[i][j] = int((crop_coords[i][j]-origin)/eggers.voxel_size[i])


print(crop_indices)
cropped_eggers = eggers.crop_by_index(crop_indices, inplace=False)

cropped_eggers.write_to_ctegsphant(abs_path+ "Final_Balloon_Crop_15cm/ct.egsphant")
