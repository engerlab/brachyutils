from brachyutils import BrachyDose, BrachyPlan, BrachyPhantom, DwellPosition, BrachyDoseGenerator
from pathlib import Path
from typing import Union, Callable, Optional
import argparse


class TG43DoseCalculator(BrachyDoseGenerator):
    """
    """
    def __init__(self,
        brachyplan: BrachyPlan,
        dir_tg43_parameters: Optional[str] = "SourceParameters/microSelectron-v2",
        output_dose_per_dwell : Optional[Union[bool, str]] = False,
        dir_output : Optional[Union[Path, str]] = None
        ):
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
        self.brachysource : self.brachyplan.simulation_setup.brachy_source
        self.source_name : str = self.brachysource.source_geometry

        #tg43 parameters
        self.air_kerma_strength : float = self.brachysource.reference_air_kerma_rate
        self.activity : float = self.brachysource.activity #can specify the (total) activity in place of the AKS
        self.dose_rate_constant : float = None
        self.radial_dose_function: Callable[[float], float] = None
        self.geometry_function: Callable[[float, float], float] = None
        self.anisotropy_function : Callable[[float, float], float] = None

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
            raise ValueError("Input BrachyPlan's BrachySimulation has no BrachySource.")

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
    #to do, parse inputs

