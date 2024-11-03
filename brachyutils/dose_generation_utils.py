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
        def generate_dose(self, filename: Optional[Path] = None):
            pass