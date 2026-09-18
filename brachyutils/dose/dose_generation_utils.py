from abc import ABC, abstractmethod
import subprocess
import sys
import os
from glob import glob
from pathlib import Path
from typing import Literal, Optional, Union
from brachyutils.planning.plan_utils import BrachyPlan
from brachyutils.planning.plan_export_configs import ExportConfig_BrachyPlan, ExportConfig_Egsphant
from brachyutils.dose.dose_utils import BrachyDose

class BrachyDoseGenerator(ABC):
    def __init__(
        self,
        dir_plan_export: Union[Path, str],
        pth_dose_executable: Union[Path, str],
    ) -> None:
        r"""
        ### Purpose:
        - A generic class to wrap around all sorts of dose generators. Each generator should
        support the attributes of this class and implements its abstract methods.

        ### Attributes:
        - dir_plan_export: Union[Path, str]: The path to the dose setup directory.
        - pth_dose_executable: Union[Path, str]: The path to the dose executable.

        ### Inputs:
        - dir_plan_export: Union[Path, str]: The path to the dose setup directory.
        - pth_dose_executable: Union[Path, str]: The path to the dose executable.

        ### Functions:
        - generate_dose(): generates the dose distribution as well as its uncertaity per voxel.
        - validate_inputs(): validates the dose setup directory.
        """
        self.dir_plan_export: Path = Path(dir_plan_export)
        self.pth_dose_executable: Path = pth_dose_executable

    @abstractmethod
    def validate_inputs(self):
        r"""
        ### Purpose:
        - Abstract method to validate the inputs of the dose generator.
        Each dose generator should implement this method.
        """
        pass

    @abstractmethod
    def generate_dose(self, pth_output: Optional[Path] = None):
        r"""
        ### Purpose:
        - Abstract method to generate the dose distribution.
        Each dose generator should implement this method.

        ### Inputs:
            - pth_output: Optional[Path]: If provided, the dose distribution will be saved to this path.
        """
        pass
    
    @abstractmethod
    def run_dose_generation(
        self,
        dir_export: str | Path = None,
        plan: BrachyPlan = None,
        generate_dose_rate_maps: bool = False,
        ) -> BrachyPlan:
        r"""
        ### Purpose:
        - to run the dose generation for the plan and return a plan with combined dose filled as well
        as the dose rate dictionary if desired.

        ### Inputs:
        - dir_export := The directory used for exporting the dosimetry setup and the generated dose maps.
        - plan:= The treatment plan for which we want to generate the dose. 
        - generate_dose_rate_maps := whether to generate dose rate maps for each dwell position.
        If True, the dose_rate_dict will be populated with the dose rate maps for each dwell position.

        ### Output:
        - plan: BrachyPlan := The brachy plan with the combined dose and optionally the dose rate dict filled.
        """
        pass

class RapidBrachyMC(BrachyDoseGenerator):
    def __init__(
        self,
        dir_plan_export: Path | str,
        pth_dose_executable: Path | str="http://192.168.1.11:8000/calculate_dose_mc",
    ) -> None:
        r"""
        ### Purpose:
        - A class to generate dose distribution using Monte Carlo simulations.
        This class uses RapidBrachyMC to calculate the dose distribution.
        """
        super().__init__(dir_plan_export, pth_dose_executable)

    def validate_inputs(self):
        r"""
        ### Purpose:
        - Validate the inputs of the Monte Carlo dose generator.
        """
        pass
    
    def generate_batch_plans():
        r"""
        ### Purpose:
        - Generate the batch plans for the Monte Carlo simulation to achieve the 
        desired uncertainty with much less time.
        """
        raise NotImplementedError("This feature is not implemented yet.")
    
    def generate_dose(
        self,
        pth_mac: Path = None,
        random_seed: int = 556677,
    ):
        r"""
        ### Purpose:
        - Generate the dose distribution for a given mac file.
        
        ### Inputs:
        - pth_mac: Path := The path to the mac file for which the dose distribution
        will be generated. If None, the function will search for all mac files in the dir_plan_export directory and generate dose for each of them.
        - random_seed: int := The random seed for the Monte Carlo simulation. The default is 556677
        """

        if pth_mac is None:
            print("No mac file is provided. Will use all mac files in the directory. except the combined.mac")
            pth_all_mac = list(self.dir_plan_export.glob("*.mac")) #glob(str(self.dir_plan_export / "*.mac"))
            if len(pth_all_mac) == 0:
                raise ValueError(
                    f"No mac file is found at {self.dir_plan_export}."
                )
            for pth_mac in pth_all_mac:
                if str(pth_mac.name) == "combined.mac":
                    continue
                self.generate_dose(
                    pth_mac=pth_mac,
                    random_seed=random_seed,
                )
        else:
            if "http" in str(self.pth_dose_executable):
                pth_mac = str(pth_mac.resolve())
                # use fast api post to request the dose calculation
                import requests
                response = requests.post(
                    self.pth_dose_executable,
                    json={
                        "pth_mac": str(pth_mac),
                        "random_seed": str(random_seed),
                    },
                    timeout=None,
                )
            elif ".py" in str(self.pth_dose_executable):
                # use subprocess to run the python script
                raise NotImplementedError("This feature is not implemented yet.")
            else:
                cmd = [str(self.pth_dose_executable), str(pth_mac), str(random_seed)]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self.dir_plan_export,
                )
                # iterate over stdout lines as they become available
                if proc.stdout is not None:
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                proc.wait()
                response = subprocess.CompletedProcess(args=cmd, returncode=proc.returncode)
                if proc.returncode != 0:
                    raise RuntimeError(f"MC dose calculation calculation process exited with code {proc.returncode}")
            return response
        

    def run_dose_generation(
        self,
        dir_export: str | Path = None,
        plan: BrachyPlan = None,
        generate_dose_rate_maps: bool = False,
        export_config_brachyplan: ExportConfig_BrachyPlan = None,
        ) -> BrachyPlan:
        r"""
        ### Purpose:
        - to run the dose generation for the plan and return a plan with combined dose filled as well
        as the dose rate dictionary if desired.

        ### Inputs:
        - dir_export := The directory used for exporting the dosimetry setup and the generated dose maps.
        - plan:= The treatment plan for which we want to generate the dose. 
        - generate_dose_rate_maps := whether to generate dose rate maps for each dwell position.
        If True, the dose_rate_dict will be populated with the dose rate maps for each dwell position.

        ### Output:
        - plan: BrachyPlan := The brachy plan with the combined dose and optionally the dose rate dict filled.
        TODO examples/benchmarks/eval_dose_generation.py has the code to fill this function. will do it when 
        I need it again!
        """
        pass
