from __future__ import annotations
from typing import Protocol, List, TypeVar, Any, Callable, Dict, Union, Optional, Tuple
from pathlib import Path
import numpy as np

# Type variables for generics
BrachyPlanType = TypeVar('BrachyPlanType', bound='BrachyPlan')
DwellTimeOptimizerType = TypeVar('DwellTimeOptimizerType', bound='DwellTimeOptimizer')
BrachyStructureType = TypeVar('BrachyStructureType', bound='BrachyStructure')
BrachyDoseType = TypeVar('BrachyDoseType', bound='BrachyDose')

# Protocol classes (structural typing interfaces)
class DwellPosition(Protocol):
    """Protocol for DwellPosition class"""
    index: int
    position: List[float]

class Catheter(Protocol):
    """Protocol for Catheter class"""
    index: int
    dwells: List[DwellPosition]
    channel_total_time: float

class CatheterTable(Protocol):
    """Protocol for CatheterTable class"""
    catheter_list: List[Catheter]
    
    def __iter__(self) -> Any: ...

class BrachyDose(Protocol):
    """Protocol for BrachyDose class"""
    dose_image: Any
    origin: List[float]
    gridSizeInWorldUnit: List[float]

class Model(Protocol):
    """Protocol for optimization Model class"""
    def addVar(self, lb: float, ub: float, name: str, vtype: Any) -> Any: ...

class BrachyStructure(Protocol):
    """Protocol for BrachyStructure class"""
    name: str
    mask: Any
    target_volume: bool
    optimization_spacing_mm: Union[float, List[float]]

class DwellTimeVariable(Protocol):
    """Protocol for DwellTimeVariable class"""
    name: str
    coordinates: List[float]
    dose_rate_map: np.ndarray
    model_variable: Any

class Constraint(Protocol):
    """Protocol for Constraint class"""
    name: str
    expression: Callable

class BrachySource(Protocol):
    """Protocol for BrachySource class"""
    treatment_type: str
    air_kerma_per_history: float
    reference_air_kerma_rate: float

class Optimization_Config(Protocol):
    """Protocol for Optimization_Config class"""
    structure_name: str
    target: bool
    penalty_weight_linear: float
    penalty_weight_quadratic: float

class BrachyPlan(Protocol):
    """Protocol for BrachyPlan class"""
    catheter_table: Any
    dose_rate_tensor: Any
    combined_dose: BrachyDose
    structure_list: List[BrachyStructure]

class DwellTimeOptimizer(Protocol):
    """Protocol for DwellTimeOptimizer class"""
    plan: BrachyPlan
    model: Any
    roi_bounds: List[List[float]]
    
    def initialize_model(self, solver: str) -> Any: ...
    def set_dwellTimeVariables(
        self, plan: BrachyPlan, initial_dwell_time: float, 
        lower_bound: float, upper_bound: float
    ) -> List[DwellTimeVariable]: ...
    def get_optimization_roi_bounds(
        self, plan: BrachyPlan, dwellTimeVariables: List[DwellTimeVariable],
        roi_margin_mm: List[float]
    ) -> List[List[float]]: ...
    def set_penalty_function(
        self, plan: BrachyPlan, dwellTimeVariables: List[DwellTimeVariable],
        model: Any
    ) -> Callable: ...

# Export all types to make them accessible when importing from this module
__all__ = [
    'BrachyPlan', 'DwellTimeOptimizer', 'BrachyStructure', 'BrachyDose',
    'Catheter', 'CatheterTable', 'DwellPosition', 
    'DwellTimeVariable', 'Constraint', 'Model',
    'BrachySource', 'Optimization_Config',
    # Type variables
    'BrachyPlanType', 'DwellTimeOptimizerType', 
    'BrachyStructureType', 'BrachyDoseType'
]

