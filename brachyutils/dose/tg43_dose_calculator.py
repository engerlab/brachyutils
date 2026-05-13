import logging
import argparse
from pathlib import Path
from typing import Union, Callable, Optional

import numpy as np
from tqdm import tqdm

from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation
from opentps.core.data.images import DoseImage
from opentps.core.processing.imageProcessing.imageTransform3D  import applyTransform3D, translateDataByChangingOrigin

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


from brachyutils.planning.plan_utils import BrachyPlan
from brachyutils.geometry.catheter_utils import DwellPosition
from brachyutils.dose.dose_utils import BrachyDose
from brachyutils.dose.dose_generation_utils import BrachyDoseGenerator
from brachyutils.geometry.phantom_utils import BrachyPhantom
from brachyutils.planning.simulation_utils import BrachySource

#unit constants
CGY = 0.01 #Gy
CM = 10. #mm
HR = 3600. #s
U = CGY * CM * CM / HR
CI = 3.7e10 #Bq
RAD = 180./np.pi

class BrachyUtilsTG43(BrachyDoseGenerator):
    """
    """
    def __init__(self,
        dir_tg43_parameters: Optional[str] = "microSelectron-v2_Consensus",
        dir_output : Optional[Union[Path, str]] = Path(),
        **calc_parameter_kwargs
        ) -> None:
        """
        """
        #input
        super().__init__(dir_output, None)
        
        self.dir_tg43_parameters : Path = (
            Path(__file__).parent.parent.parent
            / "admin/constants/TG43_Parameter_Data"
            / dir_tg43_parameters
            )

        #store meta-parameters about the calculation in this dictionary
        self.calc_parameters = {
            "kernel_max_dose_rate": 10, #Gy/s
            "epsilon": 1e-3, #just a little nudge to certain values :)
            "auto_kernel_phantom": True #if not True, you must set the next two values and resize the phantom yourself
            #"kernel_half_width" : 10 * CM, #half width to calculate dose rate kernel
            #"kernel_res" : 0.1 * CM, #resolution to calculate the dose rate kernel
        }
        self.calc_parameters.update(calc_parameter_kwargs)

        #populate attributes to the validated brachyplan input
        self.brachyplan = None
        self.brachyphantom : BrachyPhantom = None
        self.brachysource : BrachySource = None#
        self.source_name : str = None#self.brachysource.source_geometry
        self.is_hdr : bool = True#self.brachysource.treatment_type == "HDR"

        #tg43 parameters
        self.active_length : float = None
        self.air_kerma_strength : float = None
        self.activity : float = None
        self.dose_rate_constant : float = None
        self.radial_dose_function: Callable[[float], float] = None #TODO: fix function signatures
        self.geometry_function: Callable[[float, float], float] = None
        self.anisotropy_function : Callable[[float, float], float] = None

        self.tg43_dose_rate_kernel : BrachyDose = None #dose rate distribution in 3D with centered source

        #outputs

    def validate_brachyplan(self, plan : BrachyPlan) -> None:
        if plan is None:
            raise ValueError("Input BrachyPlan is None.")
        if plan.phantom is None:
            raise ValueError("Input BrachyPhantom has no BrachyPhantom.")
        if plan.simulation_setup is None:
            raise ValueError("Input BrachyPlan has no BrachySimulation.")
        if plan.simulation_setup.brachy_source is None:
            raise ValueError("Input BrachyPlan's BrachySimulation has no BrachySource.")

    
    def load_from_brachyplan(self, plan : BrachyPlan) -> None:
        self.validate_brachyplan(plan)
        self.brachyplan = plan
        self.brachyphantom : BrachyPhantom = self.auto_calculate_phantom_kernel() if self.calc_parameters["auto_kernel_phantom"] else plan.phantom
        self.brachysource : BrachySource = plan.simulation_setup.brachy_source
        self.source_name : str = self.brachysource.source_geometry
        self.is_hdr : bool = self.brachysource.treatment_type == "HDR"
        #a little hard-coded fix :)
        if self.source_name == "MicroSelectronV2":
            self.source_name = "microSelectron-v2"
        self.air_kerma_strength = self.brachysource.reference_air_kerma_rate * U
        self.activity = self.brachysource.activity #can specify the (total) activity in place of the AKS
        if self.activity is not None:
            self.activity *= CI

    AUTO_KERNEL_RES = 1.0 #mm
    def auto_calculate_phantom_kernel(self) -> BrachyPhantom:
        #automatically calculate the kernel size based on the phantom size and spacing
        #and will create a phantom of the appropriate size containing all OARs at 1 mm spacing
        logging.info("Automatically calculating kernel and phantom for TG-43 calcualtion.")
        structures_to_crop_by = [structure.name for structure in self.brachyplan.structure_list if structure.name.lower() not in "body" or structure.name.lower() in "external"]
        logging.debug(f"Cropping phantom to {structures_to_crop_by}.")
        new_phantom = self.brachyplan.phantom.crop_by_contour(contour_name = structures_to_crop_by, inplace = False, marginInMM = 20.0)
        new_phantom.cached_structure_masks = None #a little hack to prevent resampling of structure masks (useless)
        new_phantom.resample_to(spacing=np.array([1.0, 1.0, 1.0]), inplace=True)

        new_phantom_dimensions = new_phantom.image_obj.gridSize * new_phantom.image_obj.spacing
        logging.debug(f"New phantom dimensions: {new_phantom_dimensions}")
        new_kernel_half_width = max(new_phantom_dimensions) * 1.2 * 0.5 #make the kernel big enough to cover the whole phantom (with some padding)
        self.calc_parameters["kernel_half_width"] = new_kernel_half_width if new_kernel_half_width <  150.0 else 150.0 #cap the kernel size at 150 mm to prevent excessive memory usage
        self.calc_parameters["kernel_res"] = self.AUTO_KERNEL_RES
        logging.info(f"Automatically calculated kernel half width {self.calc_parameters["kernel_half_width"]} mm.")
        return new_phantom

    def load_and_initialize_tg43(self) -> None:
        #Load and initialize all of the TG-43 parameters and functions
        #NOTE: all distance dimensions should be converted 
        logging.info("Loading TG-43 parameters.")
        self.load_and_initialize_source_info()
        self.load_and_initialze_aks_drc()
        self.initialize_geometry_function()
        self.load_and_initialize_radial_dose_function()
        self.load_and_initialize_anisotropy_function()
        logging.info("TG-43 parameters initialized. Computing dose-rate kernel.")
        if logging.root.isEnabledFor(logging.DEBUG):
            self.compute_tg43_dose_rate_kernel(debug_pth_out=Path("./TG43_dose_rate_kernel.seq.nrrd"))
        else:
            self.compute_tg43_dose_rate_kernel()
        logging.info("TG-43 dose-rate kernel initialized.")

    def load_and_initialize_source_info(self) -> None:
        file_path = self.dir_tg43_parameters / "source.csv"
        with open(file_path, encoding='utf-8') as file:
            file_data = file.readlines()[0].split(',')
        file_data[2] = file_data[2][:-1] #cut out newline
        if file_data[0] != self.source_name:
            raise ValueError(f"Potential mismatch! Loaded parameters for source {file_data[0]} \
but source name is {self.source_name}.")
        source_core_from_plan = self.brachysource.core_material.split('_')[1] + "-" + str(self.brachysource.mass_number)
        if file_data[2] != source_core_from_plan:
            raise ValueError(f"Potential mismatch! Loaded parameters for source isotope ###{file_data[2]}### \
                but source core is ###{source_core_from_plan}###.")
        self.active_length = float(file_data[1]) * CM
        logging.debug("Active length %s mm", self.active_length)

    def load_and_initialze_aks_drc(self) -> None:
        file_path = self.dir_tg43_parameters / f"{self.source_name}_AKS_DRC.csv"
        file_data = np.loadtxt(file_path, dtype=np.float32, delimiter = ',', skiprows=1, usecols = [1, 2])
        if self.air_kerma_strength is None:
            self.air_kerma_strength = self.activity * file_data[0, 0] * U
            logging.info("Updated air-kerma strength to %s from activity.", self.air_kerma_strength)
        self.dose_rate_constant = file_data[0, 1] / (CM * CM)
        logging.debug("AKS: %s Gy mm^2 s^-1; DRC: %s mm^-2", self.air_kerma_strength, self.dose_rate_constant)

    def initialize_geometry_function(self) -> None:
        def geometry_function(r_theta : np.ndarray) -> np.array:
            r = r_theta[:,0]
            theta = r_theta[:, 1]
            branch_1 = np.logical_and(theta > 0.0, theta < 180.0)
            r_branch_1 = r[branch_1]
            r_branch_2 = r[np.logical_not(branch_1)]
            theta_branch_1 = theta[branch_1]
            theta_branch_1_rad = np.deg2rad(theta_branch_1)
            ell_over_two = 0.5 * self.active_length

            G = np.zeros(r_theta.shape[0])

            #branch 1 - 0 < theta < 180
            top11 = r_branch_1 * np.cos(theta_branch_1_rad) - ell_over_two
            top12 = np.sqrt(r_branch_1 * r_branch_1 + ell_over_two * ell_over_two - (self.active_length * r_branch_1 * np.cos(theta_branch_1_rad)))
            top21 = r_branch_1 * np.cos(theta_branch_1_rad) + ell_over_two
            top22 = np.sqrt(r_branch_1 * r_branch_1  + ell_over_two*ell_over_two + self.active_length * r_branch_1 * np.cos(theta_branch_1_rad))
            bottom = self.active_length * r_branch_1 * np.sin(theta_branch_1_rad)
            #top1 = np.clip(top11/top12, a_min = -1.0, a_max = 1.0) #some rounding errors causing issues with acos
            #top2 = np.clip(top21/top22, a_min = -1.0, a_max = 1.0)
            top = np.arccos(top11/top12)-np.arccos(top21/top22)
            G[branch_1] = (top/bottom)#[branch_1]

            #branch 2 - theta = 0 or theta = 180
            G[~branch_1] = 1.0 / (r_branch_2 * r_branch_2 - ell_over_two * ell_over_two)

            return G
        self.geometry_function = geometry_function

    def load_and_initialize_radial_dose_function(self) -> None:
        file_path = self.dir_tg43_parameters / f"{self.source_name}_radialdosefunction.csv"
        file_data = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype = np.float16)
        radii = file_data[:,0] * CM
        radii_meshgrid = np.meshgrid(radii)
        
        radial_dose_function_data = file_data[:,1]
        #print(radii_mm.shape, radial_dose_function_data.shape)
        if self.is_hdr:
            radial_dose_function_interpolator = RegularGridInterpolator(radii_meshgrid, radial_dose_function_data, method='linear', bounds_error=False)
            def radial_dose_function(r: np.array) -> np.array:
                gr = radial_dose_function_interpolator(r)
                if np.all(~np.isnan(gr)):
                    return gr
                gr[r < radii[0]] = radial_dose_function_data[0] #r<rmin nearest neighbor extrapolation
                gr[r > radii[-1]] = \
                radial_dose_function_data[-2] + \
                ((radial_dose_function_data[-1] -radial_dose_function_data[-2]) / (radii[-1] - radii[-2])) \
                * (r[r > radii[-1]] - radii[-2]) #r>rmax linear extrapolation
                return gr
        else:
            raise NotImplementedError("TG-43 for LDR not yet implemented.")
        self.radial_dose_function = radial_dose_function

    def load_and_initialize_anisotropy_function(self) -> None:
        #load the data
        file_path = self.dir_tg43_parameters / f"{self.source_name}_2danisotropyfunction.csv"
        file_data = np.genfromtxt(file_path,dtype=str, delimiter=',')
        file_data_F = file_data[:file_data.shape[0] // 2, :] #exclude the uncertainty
        file_data_F[file_data_F == '-'] = "-1" #points inside the source
        radii = np.array(file_data_F[1:, 0], dtype=np.float16) * CM
        thetas = np.array(file_data_F[0, 1:], dtype = np.float16)
        anisotropy_function_data = np.array(file_data_F[1:, 1:], dtype = np.float16)

        #nearest neigbor extrapolate the points of the anisotropy function in the source
        for ir in range(radii.size):
            Fr = anisotropy_function_data[ir, :]
            if(np.all(Fr > 0)):
                continue
            ithetamin = np.min(np.where(Fr > 0))
            ithetamax = np.max(np.where(Fr > 0))
            thetamin = thetas[ithetamin]
            thetamax = thetas[ithetamax]
            Fr[thetas < thetamin] = Fr[ithetamin]
            Fr[thetas > thetamax] = Fr[ithetamax]
            anisotropy_function_data[ir] = Fr

        anisotropy_function_interpolator = RegularGridInterpolator((radii, thetas), anisotropy_function_data, method='linear', bounds_error=False)
        def anisotropy_function(r_theta : np.ndarray) -> np.array: #N x 2 array -> N array
            F = anisotropy_function_interpolator(r_theta)
            for ir_in in np.where(np.isnan(F))[0]:
                itheta = np.argmin(np.abs(r_theta[ir_in,1] - thetas))
                r_in = r_theta[ir_in, 0]
                if r_in < radii[0]:
                    ir = 0
                else:
                    ir = -1
                F[ir_in] = anisotropy_function_data[ir, itheta]
            return F
        self.anisotropy_function = anisotropy_function

    def compute_tg43_dose_rate_kernel(self, debug_pth_out : Path = None) -> None:

        kernel_half_width = self.calc_parameters["kernel_half_width"]
        kernel_res = self.calc_parameters["kernel_res"]
        epsilon = self.calc_parameters["epsilon"]
        kernel_axis = np.arange(-kernel_half_width, kernel_half_width + epsilon, kernel_res, dtype = np.float32)
        kernel_axis_size = kernel_axis.size
        kernel_shape = (kernel_axis_size, kernel_axis_size, kernel_axis_size)
        kernel_x, kernel_y, kernel_z = np.meshgrid(kernel_axis, kernel_axis, kernel_axis)
        kernel_r = np.sqrt(kernel_x * kernel_x + kernel_y * kernel_y + kernel_z * kernel_z)
        kernel_r[np.isclose(kernel_r, 0., epsilon)] = epsilon
        #theta is off the z_axis (0, 0, 1), angle between two vectors a,b is arccos(a*b/(|a||b|))
        #this means that theta is just arccos(z/r)
        kernel_theta = np.arccos(kernel_z / kernel_r) * RAD
    
        del kernel_x
        del kernel_y
        del kernel_z

        logging.debug("Initializing AKS * DRC grid.")
        kernel_array = self.air_kerma_strength * self.dose_rate_constant / self.geometry_function(np.array([[10., 90.0]])) * np.ones(kernel_shape, dtype = np.float16)

        kernel_r_theta = np.column_stack((kernel_r.flatten(), kernel_theta.flatten()))
        #del kernel_r
        #del kernel_theta
        logging.debug("AKS * DRC grid initialized.")
        #geometry fn
        logging.debug("Initializing geometry function grid.")
        kernel_geometry_function = np.reshape(self.geometry_function(kernel_r_theta), kernel_shape)
        if np.any(kernel_geometry_function == 0):
            where_zero = kernel_geometry_function == 0
            logging.warn(f"Kernel's geometry function has 0s at r={kernel_r[where_zero]}/theta={kernel_theta[where_zero]}.")
        kernel_array *= kernel_geometry_function
        del kernel_geometry_function

        logging.debug("Geometry function grid initialized.")
        #radial dose fn
        logging.debug("Initializing radial dose function grid.")
        kernel_radial_dose_function = np.reshape(self.radial_dose_function(kernel_r_theta[:,0]), kernel_shape)
        kernel_array *= kernel_radial_dose_function
        del kernel_radial_dose_function
        logging.debug("Radial dose functiong grid initialized.")

        #anistropy fn
        logging.debug("Initializing 2D anisotropy function grid.")
        kernel_anisotropy_function = np.reshape(self.anisotropy_function(kernel_r_theta), kernel_shape)
        kernel_array *= kernel_anisotropy_function
        logging.debug("2D anisotropy function grid initialized.")
        if np.any(kernel_anisotropy_function == 0):
            where_zero = kernel_anisotropy_function == 0
            logging.warn(f"Kernel's anisotropy function has 0s at r={kernel_r[where_zero]}/theta={kernel_theta[where_zero]}.")
        del kernel_anisotropy_function
        #make the dose image
        if np.any(np.isnan(kernel_array)):
            print(kernel_r[np.where(np.isnan(kernel_array))], kernel_theta[np.where(np.isnan(kernel_array))])
            raise ValueError("NaN dose rate values in the TG-43 dose rate kernel.")

        np.clip(kernel_array, a_min = 0.0, a_max = self.calc_parameters["kernel_max_dose_rate"], out=kernel_array)
        logging.debug("Generating TG-43 dose rate kernel image.")
        kernel_dose_image = DoseImage(imageArray=kernel_array, name='TG-43 Dose Kernel',
            origin=(-kernel_half_width,-kernel_half_width,-kernel_half_width), spacing = (kernel_res, kernel_res, kernel_res))
        self.tg43_dose_rate_kernel = BrachyDose(dtype=np.float16)
        self.tg43_dose_rate_kernel.dose_image = kernel_dose_image
        if debug_pth_out is not None:
            logging.debug("Writing dose rate kernel to path %s", debug_pth_out)
            self.tg43_dose_rate_kernel.write_brachydose_to_file(debug_pth_out)
        logging.debug("TG-43 dose rate kernel generation complete.")

    def validate_inputs(self) -> None:
        #check that the brachyplan has the required info before proceeding
        #it should have a BrachyPhantom (providing the dose grid for calculation),
        #a CatheterTable (for dwell positions/times)
        #and a BrachySimulatinos
        #TODO: Implement this
        if self.air_kerma_strength is None and self.activity is None:
            raise ValueError("Either air kerma strength or activity should be set in the source dict.")

        if self.dose_rate_constant is None:
            raise ValueError("Dose rate constant not set.")

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

        #with ProcessPoolExecutor() as executor:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(calculate_dwell_dose_tg43, dwell, self.tg43_dose_rate_kernel, self.brachyphantom): dwell for dwell in self.brachyplan.catheter_table.all_dwells}
            for action in tqdm(
                as_completed(futures),
                total = len( self.brachyplan.catheter_table.all_dwells),
                desc = "Calculating dwell doses:"):
                    try:
                        action.result()
                    except Exception as exc:
                        failed_dwell = futures[action]
                        raise ValueError(f"TG-43DoseCalculator failed for dwell {failed_dwell.name_id}") from exc

        logging.info("TG-43 dose calculation complete.")
        if export_combined_dose:
            combined_dose = self.brachyplan.combined_dose
            if dir_export is None:
                dir_export = Path(self.dir_plan_export)
            else:
                dir_export = Path(dir_export)
            pth_output = dir_export / "combined_TG43.seq.nrrd"
            logging.info("Writing combined TG-43 dose to %s.", pth_output)
            combined_dose.write_brachydose_to_file(pth_output)

    def generate_dose(self, pth_output: Optional[Path] = None):
        raise NotImplementedError("generate_dose() not implemented for BrachyUtilsTG43. Call run_dose_generation() instead.")

def calculate_dwell_dose_tg43(dwell : DwellPosition, dose_rate_kernel: BrachyDose, phantom : BrachyPhantom ) ->  None:
    rotation_matrix = calculate_dwell_rotation_matrix(dwell)
    dose_rate_kernel = dose_rate_kernel.dose_image.copy()
    applyTransform3D(dose_rate_kernel, rotation_matrix, fillValue=0,
        outputBox = 'keepAll', rotCenter = [0.0, 0.0, 0.0], interpOrder = 1),# translation=dwell.position)
    translateDataByChangingOrigin(dose_rate_kernel, dwell.position)
    dose_rate_kernel.resampleOn(phantom.image_obj, fillValue=0, tryGPU=False)
    if dwell.dose_rate is None:
        dwell.dose_rate = BrachyDose(dtype=np.float16)
    dwell.dose_rate.dose_image = dose_rate_kernel

def calculate_dwell_rotation_matrix( dwell : DwellPosition) -> np.ndarray:
    #build an affine matrix with an extrinsic rotation around Z->Y->X then the translation to the dwell
    dwell_rot = dwell.rotation
    dwell_angle = float(dwell.angle) #todo: perform the Z rotation first
    return Rotation.align_vectors(dwell_rot, [0, 0, 1])[0].as_matrix()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #TODO:, parse inputs
