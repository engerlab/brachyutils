from __future__ import annotations
from typing import Protocol, List, TypeVar, Any, Callable, Dict, Union, Optional, Tuple
from pathlib import Path
import numpy as np
from abc import ABC

# Type variables for generics
BrachyPlanType = TypeVar('BrachyPlanType', bound='BrachyPlan')
DwellTimeOptimizerType = TypeVar('DwellTimeOptimizerType', bound='DwellTimeOptimizer')
BrachyStructureType = TypeVar('BrachyStructureType', bound='BrachyStructure')
BrachyDoseType = TypeVar('BrachyDoseType', bound='BrachyDose')

# Protocol classes (structural typing interfaces)
class DwellPosition(Protocol):
    """Protocol for DwellPosition class"""
    index: int
    angle: float
    position: List[float] | Dict[str, float]
    relativePos: float
    rotation: List[float] | Dict[str, float]
    time: float

class Catheter(Protocol):
    """Protocol for Catheter class"""
    index: int
    tip_position: Optional[List[float]]
    points: Optional[List[List[float]]]
    dwells: List[DwellPosition]
    afterloader_channel_number: Optional[int]
    channel_total_time: float
    step_size: float
    fit_function: Any
    insert_position: Optional[List[float]]

class CatheterTable(Protocol):
    """Protocol for CatheterTable class"""
    catheter_list: List[Catheter]
    step_size: float
    treatment_time: float
    channel_length: Optional[float]
    delivered_dwell_coordinates: Optional[Dict[str, List[List[float]]]]
    
    def __iter__(self) -> Any: ...

class BrachyDose(Protocol):
    """Protocol for BrachyDose class"""
    path: Optional[Path]
    dose_image: Any  # DoseImage from opentps
    uncertainty_image: Optional[Any]  # DoseImage from opentps
    voxel_edges: Optional[np.ndarray]
    interpolation_function: Optional[Any]
    unit_length: str
    xyz_format: bool
    
    def load_file_to_brachydose(self, pth_dose_file: Path, load_uncertainty: Optional[bool] = True) -> None: ...
    def create_interpolation_function(self) -> None: ...

class Model(Protocol):
    """Protocol for optimization Model class"""
    def addVar(self, lb: float, ub: float, name: str, vtype: Any) -> Any: ...

class BrachyStructure(Protocol):
    """Protocol for BrachyStructure class"""
    name: str
    mask: Any
    target_volume: bool
    optimization_spacing_mm: Union[float, List[float]]
    in_dvh: Optional[bool]
    dvh_metric_goals: Optional[Dict[str, float]]
    dvh_metrics_observed: Optional[Dict[str, float]]
    dvh_obj: Optional[Any]
    uvh: Optional[Any]
    uncertainty_mean: Optional[float]
    uncertainty_std: Optional[float]
    uncertainty_max: Optional[float]
    uncertainty_min: Optional[float]
    optimization_config: Optional[Any]

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
    spacing_mm: Union[float, List[float]]
    dose_voxel_goal: float
    penalty_weight_linear: float
    penalty_weight_quadratic: float
    penalty_weight_hotspot: float
    hotspot_threshold: float
    penalty_weight_uniformity: float
    mask_margin_mm: Union[float, List[float]]
    min_dose: float
    max_dose: float

class BrachyPlan(Protocol):
    """Protocol for BrachyPlan class"""
    phantom: Any  # BrachyPhantom
    dvh_metric_goals: Optional[dict]
    dvh_metrics_observed: Optional[dict]
    structure_list: List[BrachyStructure]
    phantom_origin: Optional[List[float]]
    organ_bounds: Optional[List[float]]
    catheter_table: Optional[Any]  # CatheterTable
    num_catheters: Optional[int]
    catheter_numbers: List[int]
    num_dwells: Optional[int]
    dwell_numbers: List[int]
    dwell_times: List[float]
    dwell_coordinates: List[List[float]]
    applicator_list: List[Any]  # List[BrachyApplicator]
    applicator_rotation_axis: np.ndarray
    applicator_rotation_origin: np.ndarray
    dose_rate_tensor: np.ndarray
    combined_dose: Optional[BrachyDose]
    uncertainty_tensor: np.ndarray
    simulation_setup: Optional[Any]  # BrachySimulation
    prescription_dose: Optional[float]

class DwellTimeOptimizer(Protocol):
    """Protocol for DwellTimeOptimizer class"""
    plan: Any  # BrachyPlan
    solver: Optional[str]
    dwellTimeVariables: Optional[List[Any]]  # List[BrachyDwellTime]
    model: Any
    roi_bounds: Optional[List[List[float]]]
    roi_margin_mm: Union[List[float], float]
    solution_found: bool
    solve_time: float
    
    def initialize_model(self, solver: str, pth_logfile: Optional[str] = None) -> Any: ...
    def set_dwellTimeVariables(
        self, plan: Any, initial_dwell_time: float = 0.0, 
        lower_bound: float = 0.0, upper_bound: float = 100
    ) -> List[Any]: ...
    def get_optimization_roi_bounds(
        self, plan: Any, dwellTimeVariables: List[Any],
        roi_margin_mm: List[float]
    ) -> List[List[float]]: ...
    def set_penalty_function(
        self, plan: Any, dwellTimeVariables: List[Any],
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

