from brachyutils.dose import BrachyUtilsTG43
from brachyutils.planning import BrachyPlan
from brachyutils.geometry import BrachyPhantom
from brachyutils.geometry import BrachyApplicator
from brachyutils.geometry.catheter_utils import DwellPosition
from brachyutils.dose import BrachyDose
from opentps.core.processing.imageProcessing.imageTransform3D  import applyTransform3D, translateDataByChangingOrigin
from opentps.core.data.images import DoseImage

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
from typing import Optional, Union, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging


class BrachyUtilsTG43S(BrachyUtilsTG43):
    """
    """
    def __init__(self,
        dir_shielding_kernels : Union[Path, str],
        dir_tg43_parameters: Optional[Union[Path, str]] = "GenericHDR",
        dir_output : Optional[Union[Path, str]] = Path(),
        **calc_parameter_kwargs
        ) -> None:
        """
        """
        #initialize the BrachyUtilsTG43 paremt class
        super().__init__(dir_tg43_parameters=dir_tg43_parameters, dir_output=dir_output, calc_parameter_kwargs=calc_parameter_kwargs)
        self.dir_shielding_kernels = Path(dir_shielding_kernels)


    def run_dose_generation(
        self,
        dir_export: str | Path = None,
        plan: BrachyPlan = None,
        generate_dose_rate_maps: bool = True,
        export_combined_dose:bool = False) -> BrachyPlan:
        if not generate_dose_rate_maps:
            raise ValueError("generate_dose_rate_maps must be True in BrachyUtilsTG43.")

        self.load_from_brachyplan(plan)
        self.load_and_initialize_tg43()
        self.validate_inputs()
        self.initialize_shielding_kernels()
        if len(plan.applicator_list) == 0:
            raise ValueError("TG-43S requires a BrachyApplicator.")

        #with ProcessPoolExecutor() as executor:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(calculate_dwell_dose_tg43s, dwell, self.tg43_dose_rate_kernel, self.brachyphantom, self.shielding_kernels, plan.applicator_list): dwell for dwell in self.brachyplan.catheter_table.all_dwells}
            for action in tqdm(
                as_completed(futures),
                total = len( self.brachyplan.catheter_table.all_dwells),
                desc = "Calculating dwell doses:"):
                    try:
                        action.result()
                    except Exception as exc:
                        failed_dwell = futures[action]
                        raise ValueError(f"TG-43S dose calculation failed for dwell {failed_dwell.name_id}") from exc

        logging.info("TG-43S dose calculation complete.")
        if export_combined_dose:
            combined_dose = self.brachyplan.combined_dose
            if dir_export is None:
                dir_export = Path(self.dir_plan_export)
            else:
                dir_export = Path(dir_export)
            pth_output = dir_export / "combined_TG43S.seq.nrrd"
            logging.info("Writing combined TG-43S dose to %s.", pth_output)
            combined_dose.write_brachydose_to_file(pth_output)

    def generate_dose(self, pth_output: Optional[Path] = None):
        raise NotImplementedError("generate_dose() not implemented for BrachyUtilsTG43. Call run_dose_generation() instead.")

    def initialize_shielding_kernels(self):
        #shielding kernel pattern: S_{zsource}mm.seq.nrrd
        #we want to populate a dict of z_source : path to the kernel
        logging.info("Initializing shielding kernels from directory %s.", self.dir_shielding_kernels)
        pth_kernels = list(self.dir_shielding_kernels.glob("*seq.nrrd"))
        if len(pth_kernels) == 0:
            raise ValueError(f"No shielding kernels found in directory {self.dir_shielding_kernels}.")
        self.shielding_kernels = {}
        for pth_kernel in pth_kernels:
            kernel_name = pth_kernel.stem[:-6] #remove mm.seq and .nrrd
            z_source = int(kernel_name.split("_")[1]) #remove S_ and then mm
            self.shielding_kernels[z_source] = BrachyDose(pth_dose_file = pth_kernel, load_uncertainty=False, dtype=np.float16).dose_image

        #sort the dict by z_source
        self.shielding_kernels = dict(sorted(self.shielding_kernels.items()))
        logging.info("Loaded %d shielding kernels.", len(self.shielding_kernels))

def calculate_dwell_dose_tg43s(dwell : DwellPosition, dose_rate_kernel: BrachyDose, phantom : BrachyPhantom, shielding_kernels : dict, applicator_list : List[BrachyApplicator] ) ->  None:
    rotation_matrix = calculate_dwell_rotation_matrix(dwell, applicator_list)
    dose_rate_kernel_image = dose_rate_kernel.dose_image.copy()
    shielding_kernel = calculate_shielding_kernel(dwell, shielding_kernels, applicator_list)
    shielding_kernel.resampleOn(dose_rate_kernel_image, fillValue=0, tryGPU=False)
    dose_rate_kernel_image.imageArray *= shielding_kernel.imageArray
    applyTransform3D(dose_rate_kernel_image, rotation_matrix, fillValue=0,
        outputBox = 'keepAll', rotCenter = [0.0, 0.0, 0.0], interpOrder = 1),# translation=dwell.position)
    translateDataByChangingOrigin(dose_rate_kernel_image, dwell.position)
    dose_rate_kernel_image.resampleOn(phantom.image_obj, fillValue=0, tryGPU=False)
    if dwell.dose_rate is None:
        dwell.dose_rate = BrachyDose(dtype=np.float16)
    dwell.dose_rate.dose_image = dose_rate_kernel_image

def calculate_dwell_rotation_matrix( dwell : DwellPosition, applicator_list) -> np.ndarray:
    #build an affine matrix with an extrinsic rotation around Z->Y->X then the translation to the dwellcc
    dwell_rot = dwell.rotation
    dwell_angle = float(dwell.angle) #the spin of the applicator around its central axis after placement
    applicator_spin_angle = applicator_list[0].rotation[0] #the spin of the applicator STL around its central axis for initial placement
    total_angle = applicator_spin_angle - dwell_angle #don't ask
    applicator_spin = Rotation.from_euler('z', total_angle, degrees=True)
    dwell_rot_rotation = Rotation.align_vectors(dwell_rot, [0, 0, 1])[0]
    return (applicator_spin * dwell_rot_rotation).as_matrix()

def calculate_shielding_kernel(dwell, shielding_kernels, applicator_list) -> DoseImage:
    z_source = int(dwell.relativePos)
    if z_source > max(shielding_kernels.keys()) or z_source < min(shielding_kernels.keys()):
        raise ValueError(f"Dwell relative position {z_source} is out of bounds for available shielding kernels.")
    if z_source in shielding_kernels.keys():
        return shielding_kernels[dwell.relativePos].copy()

    #otherwise, linearly interpolate between the two nearest kernels
    shielding_kernel = shielding_kernels.values()[0].copy() #just to get the metadata, we'll overwrite the data with the interpolation

    z_source_lower = max([z for z in shielding_kernels.keys() if z < z_source])
    z_source_upper = min([z for z in shielding_kernels.keys() if z > z_source])
    kernel_lower = shielding_kernels[z_source_lower]
    kernel_upper = shielding_kernels[z_source_upper]
    weight_upper = (z_source - z_source_lower) / (z_source_upper - z_source_lower)
    weight_lower = 1 - weight_upper
    #perform the interpolation
    kernel_interp = weight_lower * kernel_lower.imageArray + weight_upper * kernel_upper.imageArray
    shielding_kernel.imageArray = kernel_interp.astype(np.float16)
    return shielding_kernel

