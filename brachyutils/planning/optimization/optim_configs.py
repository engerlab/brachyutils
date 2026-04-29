from typing import List, Dict, Union, Annotated
from pydantic import (
    BaseModel, ConfigDict, model_validator,
    Field, TypeAdapter, ValidationError)
import numpy as np
from opentps.core.data.images import ROIMask

# Define the specific patterns
# 1. dwell_#_#_# (e.g., dwell_1_2_3)
PatternDwell = Annotated[str, Field(pattern=r'^dwell_\d+_\d+_\d+$')]

# 2. catheter_# (e.g., catheter_5)
PatternCatheter = Annotated[str, Field(pattern=r'^catheter_\d+$')]

# 3. sum_catheters (static string)
PatternSumCatheters = Annotated[str, Field(pattern=r'^sum_catheters$')]

# 4. sum_dwelltimes (static string)
PatternSumDwellTimes = Annotated[str, Field(pattern=r'^sum_dwelltimes$')]

# Create a Union type that accepts any of the above
ValidName = Union[PatternDwell, PatternCatheter, PatternSumCatheters, PatternSumDwellTimes]

def _validate_pattern(query: str, pattern: ValidName) -> bool:
    adapter = TypeAdapter(pattern)
    try:
        adapter.validate_python(query)
        return True
    except ValidationError:
        return False

def _is_binary_or_None(value):
    return value in [0, 1, None]

class Constraint_Config(BaseModel):
    """
    ### Purpose:
    - A class to represent the constraint information on other dwell time or catheter
    variables. The name of the config should match the name of the variable in the optimization model.
    - Each variable can have min, max or equality constraints. Set exactly the constraint you want and
    leave the others as None.
    - The name of the constraints on the number of catheters or the total dwell times should being with
    "sum_catheters" and "sum_dwelltimes".

    ### Attributes:
    - name:= The name of the model variable, which is a string in one of the following patterns:
        - dwell_#_#_#
        - catheter_#
        - sum_catheters
        - sum_dwelltime
    - minimum: int | float = None
    - maximum: int | float = None
    - equal: int | float = None
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        )

    name: ValidName
    minimum: int | float = None
    maximum: int | float = None
    equal: int | float = None

    @model_validator(mode="after")
    def sanity_check(self):
        if _validate_pattern(self.name, PatternCatheter):
            if (
                (not _is_binary_or_None(self.minimum))
                or (not _is_binary_or_None(self.maximum))
                or (not _is_binary_or_None(self.equal))):
                raise ValueError(f"minimum, maximum and equality constraints for {self.name} \
must be binary values (0 or 1)")
        elif _validate_pattern(self.name, PatternSumCatheters):
            if (
                (not (isinstance(self.minimum, int) or self.minimum is None))
                or (not isinstance(self.maximum, int) or self.maximum is None)
                or (not isinstance(self.equal, int) or self.equal is None)):
                raise ValueError(f"minimum, maximum and equality constraints for {self.name} \
must be integer values")

        if self.maximum is not None:
            if self.minimum:
                if self.minimum > self.maximum:
                    raise ValueError(f"maximum value cannot be less than \
minimum value for constraint {self.name}")
                if self.equal is not None and self.equal > self.maximum:
                    raise ValueError(f"equality value cannot be larger than \
maximum value for constraint {self.name}")
        if self.equal is not None:
            if self.minimum:
                if self.equal < self.minimum:
                    raise ValueError(f"equality value cannot be less than \
minimum value for constrant {self.name}")


class Optimization_Config(BaseModel):
    """
    ### Purpose:
    - This class holds the information regarding the optimization configuration per each structure.
    When loading the BrachyPlan the optimization config is created for each structure in the plan.structure_list.
    Some attributes are unique to target structures (CTV/PTV) and some are common to all structures.
    target attributes: 
        - penalty_weight_hotspot
        - hotspot_threshold
        - catheter_recommendaion
        - penalty_weight_variance_time
        - penalty_weight_uniformity
    ### Attributes:
    - structure_name: str := The name of the structure to which this optimization config applies.
    - is_target: bool := If true, we're looking at a target structure.
    - spacing_mm: List[float] | float := The spacing of the optimization grid in mm. 
    - dose_voxel_goal: float := The dose goal for every voxel in the structure in Gy.
    - penalty_weight_linear: float := Weight for linear penalty term in objective function. Default 1.
    - penalty_weight_quadratic: float := Weight for quadratic penalty term. Default 1.
    - penalty_weight_hotspot: float := Weight for hotspot penalty term. Default 0.
    - hotspot_threshold: float := If the average dose to the hot spot estimator volume goes above (target_dose * hotspot_threshold),
    penalty will be calculated for that hot spot estimator volume. Default 0.
    - penalty_weight_uniformity: float := Weight for dose uniformity penalty. Default 1.
    - mask_margin_mm: List[float] | float := Margin around structure for optimization in mm. Default 0.
    - min_dose: float := Minimum allowed dose in Gy. Default 0.
    - max_dose: float := Maximum allowed dose in Gy. Default 500.
    - constraint_num_catheters: int := The constraint on the number of catheters. Could specify the 
    minimum, maximum and the exact number of catheters desired in the plan.
    - catheter_recommendaion: bool := If True, catheter positions will be optimized as well. Default False.
    - dwell_coef_dict: Dict[str, np.array] := A dictionary mapping the name of the dwell position to the cropped, masked
    and flattend dose rate map corresponding to that dwell positition.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        )

    structure_name:str = None
    is_target:bool = False
    spacing_mm:float | List[float]= None
    dose_voxel_goal:float = None
    penalty_weight_linear:float = 0
    penalty_weight_quadratic:float = 0
    penalty_weight_hotspot:float = 0
    hotspot_threshold:float = 0
    penalty_weight_uniformity:float = 0
    penalty_weight_variance_time:float = 0
    mask_margin_mm:float | List[float]= 0
    min_dose:float = 0
    max_dose:float = 500
    catheter_recommendaion: bool = False
    constraint_num_catheters: Constraint_Config = None
    dwell_coef_dict:Dict[str, np.ndarray] = None
    mask:ROIMask = None
    # may be needed later
    # self.index_range_constraints: List[int] = None
    @model_validator(mode="after")
    def validate_target_only_fields(self):
        if not self.is_target:
            assert self.penalty_weight_hotspot == 0, "only target structure can have penalty_weight_hotspot"
            assert self.hotspot_threshold == 0, "only target structure can have hotspot_threshold"
            assert self.catheter_recommendaion == False, "only target structure can have catheter_recommendaion"
            assert self.penalty_weight_variance_time == 0, "only target structure can have penalty_weight_variance_time"
            assert self.penalty_weight_uniformity == 0, "only target structure can have penalty_weight_uniformity"
        return self