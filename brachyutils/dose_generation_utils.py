from abc import ABC, abstractmethod
from brachyutils.egsphant_utils import BrachyEgsphant
from brachyutils.dose_utils import BrachyDose
from brachyutils.simulation_utils import BrachySource
from geometry_utils import CatheterTable

from typing import Optional
from pathlib import Path

class DoseGenerator(ABC):
    def __init__(
        self,
        egsphant: BrachyEgsphant,
        brachysource: BrachySource,
        catheter_table: CatheterTable,
        ) -> None:
        r"""
        Purpose:
            - A generic class to wrap around all sorts of dose generators. Each generator should 
            support the attributes of this class and implements its abstract methods.
        Attributes:
            - egsphant: BrachyEgsphant
            - source: BrachySource
            - catheter_table: CatheterTable
            - dose: BrachyDose
        Inputs:
            - egsphant: BrachyEgsphant
            - brachysource: BrachySource
            - catheter_table: CatheterTable
        Functions:
            - generate_dose(): generates the dose distribution as well as its uncertaity per voxel.
        """
        self.egsphant = egsphant
        self.source: BrachySource = brachysource
        self.catheter_table: CatheterTable = catheter_table

        # this will be set by the generate_dose() method
        self.dose: BrachyDose = None
    
    @abstractmethod
    def validate_inputs(self):
        r"""
        Purpose:
            - Abstract method to validate the inputs of the dose generator.
            Each dose generator should implement this method.
        """
        pass
    
    @abstractmethod
    def generate_dose(self, pth_output: Optional[Path] = None):
        r"""
        Purpose:
            - Abstract method to generate the dose distribution.
            Each dose generator should implement this method.
        Inputs:
            - pth_output: Optional[Path]: If provided, the dose distribution will be saved to this path.     
        """
        if pth_output is not None:
            self.dose.write_brachydose_to_file(pth_output)

class DoseTG43(DoseGenerator):
    def __init__(
        self,
        egsphant: BrachyEgsphant,
        brachysource: BrachySource,
        catheter_table: CatheterTable,
        ) -> None:
        r"""
        Purpose:
            - A class to generate dose distribution using the TG43 formalism. 
            This class uses RapidBrachyTG43 to calculate the dose distribution.
        """
        super().__init__(egsphant, brachysource, catheter_table)
    
    def generate_dose(self, filename: Optional[Path] = None):
        r"""
        Purpose:
            - Generate the dose distribution using the TG43 formalism.
        Inputs:
            - filename: Optional[Path] = None
        """
        # do the dose calculation here
        # self.dose = BrachyDose(...)
        pass