from brachyutils import BrachyDose, BrachyPlan, BrachyPhantom, CatheterTable, DwellPosition, BrachyDoseGenerator
from pathlib import Path
from typing import List, Literal, Union, Dict, Tuple, Callable, Optional
import argparse


class TG43DoseCalculator(BrachyDoseGenerator):
    """
    """
    def __init__(self,
        brachyplan: BrachyPlan,
        output_dose_per_dwell : Optional[Union[bool, str]] = False,
        dir_source_parameters: Optional[str] = "SourceParameters/microSelectron-v2",
        ):
        """
        """
        #input
        super().__init__(dir_plan_export, pth_dose_executable)
        self.brachyplan : BrachyPlan = brachyplan
        self.output_dose_per_dwell : Union[bool, str] = output_dose_per_dwell

        #check that the brachyplan has the required info before proceeding
        #it should have a BrachyPhantom (providing the dose grid for calculation),
        #a CatheterTable (for dwell positions/times)
        #and a BrachySimulatinos
        self.validate_brachyplan()

        #taken from the input

        self.brachyphantom : BrachyPhantom = self.brachyplan.phantom
        self.brachysource : BrachYSimuol
        self.param_dir : Path = None
        self.output_dir : Path = None
        self.source_name : str = None

        #tg43 parameters
        self.air_kerma_strength : float = None
        self.activity : float = None #can specify the (total) activity in place of the AKS
        self.dose_rate_constant : float
        self.radial_dose_function: Callable[[float], float]
        self.geometry_function: Callable[[float, float], float]
        self.anisotropy_function : Callable[[float, float], float]

        #outputs
        self.combined_dose : BrachyDose = None

    def validate_brachyplan(self):
        if self.brachyplan is None:
            raise ValueError("Input BrachyPlan is None.")
        if self.brachyplan.phantom is None:
            raise ValueError("Input BrachyPhantom has no BrachyPhantom.")
        if self.brachyplan.simulation_setup is None:
            raise ValueError("Input BrachyPlan has no BrachySimulation.")
        if self.brachyplan.simulation_setup.brachy_source is None:
            raise ValueError("Input BrachyPlan's BrachySimulation has no BrachySource")
        





    def validate_inputs(self):
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

    pass
    #to do, parse inputs

