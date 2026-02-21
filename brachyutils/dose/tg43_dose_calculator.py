import logging
import argparse
import numpy as np

from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
from typing import Union, Callable, Optional
from opentps.core.data.images import DoseImage
from opentps.core.processing.imageProcessing.imageTransform3D  import transform3DMatrixFromTranslationAndRotationsVectors



from brachyutils.planning.plan_utils import BrachyPlan
from brachyutils.geometry.catheter_utils import CatheterTable, DwellPosition
from brachyutils.dose.dose_utils import BrachyDose
from brachyutils.dose.dose_generation_utils import BrachyDoseGenerator
from brachyutils.geometry.phantom_utils import BrachyPhantom
from brachyutils.planning.simulation_utils import BrachySimulation, BrachySource

#unit constants
CGY = 0.01 #Gy
CM = 10. #mm
HR = 3600. #s
U = CGY * CM * CM / HR
CI = 3.7e10 #Bq

class TG43DoseCalculator(BrachyDoseGenerator):
    """
    """
    def __init__(self,
        brachyplan: BrachyPlan,
        dir_tg43_parameters: Optional[str] = "SourceParameters/microSelectron-v2",
        output_dose_per_dwell : Optional[Union[bool, str]] = False,
        dir_output : Optional[Union[Path, str]] = Path()
        ) -> None:
        """
        """
        #input
        super().__init__(dir_output, None)
        self.brachyplan : BrachyPlan = brachyplan
        self.dir_tg43_parameters : Path = dir_tg43_parameters
        self.output_dose_per_dwell : Union[bool, str] = output_dose_per_dwell
        if isinstance(dir_output, str):
            dir_output = Path(dir_output)
        self.dir_output = dir_output

        #check that the brachyplan has the required info before proceeding
        #it should have a BrachyPhantom (providing the dose grid for calculation),
        #a CatheterTable (for dwell positions/times)
        #and a BrachySimulatinos
        self.validate_brachyplan()

        #populate attributes to the validated brachyplan input
        self.brachyphantom : BrachyPhantom = self.brachyplan.phantom
        self.brachysource : BrachySource =  self.brachyplan.simulation_setup.brachy_source
        self.source_name : str = self.brachysource.source_geometry
        self.is_hdr : bool = self.brachysource.treatment_type == "HDR"

        #a little hard-coded fix :)
        if self.source_name == "MicroSelectronV2":
            self.source_name = "microSelectron-v2"

        #tg43 parameters
        self.active_length : float = None
        print(self.brachysource.reference_air_kerma_rate )
        self.air_kerma_strength : float = self.brachysource.reference_air_kerma_rate * U
        self.activity : float = self.brachysource.activity #can specify the (total) activity in place of the AKS
        if self.activity is not None:
            self.activity *= CI
        self.dose_rate_constant : float = None
        self.radial_dose_function: Callable[[float], float] = None
        self.geometry_function: Callable[[float, float], float] = None
        self.anisotropy_function : Callable[[float, float], float] = None

        self.tg43_dose_rate_kernel : BrachyDose = None #dose rate distribution in 3D with centered source

        self.load_and_initialize_tg43()

        #outputs

        self.combined_dose : BrachyDose = None

    def validate_brachyplan(self) -> None:
        if self.brachyplan is None:
            raise ValueError("Input BrachyPlan is None.")
        if self.brachyplan.phantom is None:
            raise ValueError("Input BrachyPhantom has no BrachyPhantom.")
        if self.brachyplan.simulation_setup is None:
            raise ValueError("Input BrachyPlan has no BrachySimulation.")
        if self.brachyplan.simulation_setup.brachy_source is None:
            raise ValueError("Input BrachyPlan's BrachySimulation has no BrachySource.")
    
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
        with open(file_path) as file:
            file_data = file.readlines()[0].split(',')
        file_data[2] = file_data[2][:-1] #cut out newline
        if file_data[0] != self.source_name:
            raise ValueError(f"Potential mismatch! Loaded parameters for source {file_data[0, 0]} \
but source name is {self.source_name}.")
        source_core_from_plan = self.brachysource.core_material.split('_')[1] + "-" + str(self.brachysource.mass_number)
        if file_data[2] != source_core_from_plan:
            raise ValueError(f"Potential mismatch! Loaded parameters for source isotope ###{file_data[2]}### \
                but source core is ###{source_core_from_plan}###.")
        self.active_length = float(file_data[1]) * CM
        logging.debug(f"Active length %s mm", self.active_length)

    def load_and_initialze_aks_drc(self) -> None:
        file_path = self.dir_tg43_parameters / f"{self.source_name}_AKS_DRC.csv"
        file_data = np.loadtxt(file_path, dtype=np.float32, delimiter = ',', skiprows=1, usecols = [1, 2])
        if self.air_kerma_strength is None:
            self.air_kerma_strength = self.activity * file_data[0, 0] * U
            logging.info("Updated air-kerma strength to %s from activity.", self.air_kerma_strength)
        self.dose_rate_constant = file_data[0, 1] / (CM * CM)
        logging.debug("AKS: %s Gy mm^2 s^-1; DRC: %s mm^-2", self.air_kerma_strength, self.dose_rate_constant)

    def initialize_geometry_function(self) -> None:
        def geometry_function(r: np.float16, theta_deg: np.float16):
            theta_rad = np.deg2rad(theta_deg)
            ell_over_two = 0.5 * self.active_length
            top11 = r * np.cos(theta_rad) - ell_over_two
            top12 = np.sqrt(r*r + ell_over_two * ell_over_two - (self.active_length * r * np.cos(theta_rad)))
            top21 = r * np.cos(theta_rad) + ell_over_two
            top22 = np.sqrt(r*r + ell_over_two*ell_over_two + self.active_length * r * np.cos(theta_rad))
            bottom = self.active_length * r * np.sin(theta_rad)
            top = np.acos(top11/top12) - np.acos(top21/top22)
            return top/bottom
        self.geometry_function = np.vectorize(geometry_function, otypes = [np.float16])

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
            if(np.all(~np.isnan(F))):
                return F
            for ir_in in np.where(np.isnan(F)):
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
        kernel_max_r = 40 * CM
        kernel_res = 5 * CM
        kernel_axis = np.arange(-kernel_max_r, kernel_max_r + 1e-8, kernel_res, dtype = np.float32)
        kernel_axis_size = kernel_axis.size
        kernel_shape = (kernel_axis_size, kernel_axis_size, kernel_axis_size)
        kernel_x, kernel_y, kernel_z = np.meshgrid(kernel_axis, kernel_axis, kernel_axis)
        kernel_r = np.sqrt(kernel_x * kernel_x + kernel_y * kernel_y + kernel_z * kernel_z)
        #theta is off the z_axis (0, 0, 1), angle between two vectors a,b is arccos(a*b/(|a||b|))
        #this means that theta is just arccos(z/r)
        kernel_theta = np.arccos(kernel_z / kernel_r)
    
        del kernel_x
        del kernel_y
        del kernel_z

        kernel_array = self.air_kerma_strength * self.dose_rate_constant * np.ones(kernel_shape, dtype = np.float16)

        kernel_r_theta = np.column_stack((kernel_r.flatten(), kernel_theta.flatten()))
        #geometry fn
        kernel_geometry_function = np.reshape(self.geometry_function(kernel_r_theta), kernel_shape)
        kernel_array *= kernel_geometry_function 
        del kernel_geometry_function
        #radial dose fn
        kernel_radial_dose_function = np.reshape(self.radial_dose_function(kernel_r_theta[:,0]), kernel_shape)
        kernel_array *= kernel_radial_dose_function
        del kernel_radial_dose_function
        #anistropy fn
        kernel_anisotropy_function = np.reshape(self.anisotropy_function(kernel_r_theta), kernel_shape)
        kernel_array *= kernel_anisotropy_function
        del kernel_anisotropy_function
        #make the dose image
        kernel_dose_image = DoseImage(imageArray=kernel_array, name='TG-43 Dose Kernel',
            origin=(-kernel_max_r,-kernel_max_r,-kernel_max_r), spacing = (kernel_res, kernel_res, kernel_res))
        self.tg43_dose_rate_kernel = BrachyDose()
        dose_out.dose_image = kernel_dose_image
        if debug_pth_out is not None:

            dose_out.write_brachydose_to_file(debug_pth_out)

    def validate_inputs(self) -> None:
        if self.air_kerma_strength is None and self.activity is None:
            raise ValueError("Either air kerma strength or activity should be set in the source dict.")

        if self.dose_rate_constant is None:
            raise ValueError("Dose rate constant not set.")

    def generate_dose(self, pth_output: Optional[Path] = None):
        pass

    def calculate_dwell_dose_tg43(self, dwell_position : DwellPosition):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #to do, parse inputs

