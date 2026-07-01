from typing import List, Dict, Annotated, Literal
from pydantic import (
    BaseModel, ConfigDict, model_validator, computed_field,
    Field, TypeAdapter, ValidationError)
import numpy as np
from opentps.core.data.images import ROIMask

# Define the specific patterns
# 1. dwell_#_#_# (e.g., 1_2_3)
PatternDwell = Annotated[str, Field(pattern=r"^dwell_[1-9]\d*_[1-9]\d*_(?:0|[1-9]\d*)$")]

# 2. catheter_# (e.g., 5)
PatternCatheter = Annotated[str, Field(pattern=r'^catheter_[1-9]\d*$')]

# 3. cluster name id: (depth, cluster.index+1)
PatterCluster = Annotated[str, Field(pattern=r"^\(\d+,\s*[1-9]\d*\)$")]

def _validate_pattern(query: str, pattern: PatternDwell | PatternCatheter) -> bool:
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
    - A class to define optimization constraints applied to dwell-time or catheter variables.
    - Each constraint is described by its `constraint_type`, the `variable_type` it applies to,
      optional bound values (`minimum`, `maximum`, `equal`), and the relevant variable identifiers.
    - Depending on the constraint type, the configuration may apply to a single variable, a group
      of variables, a catheter cluster, or a pair of catheter variables.
    - Validation ensures that the selected constraint type is compatible with the variable type and
      that the required supporting fields are provided.

    ### Attributes:
    - constraint_type: The type of constraint to apply. Must be one of:
        - `bound`: applies a min, max, or equality constraint to a single variable.
        - `sum`: applies a min, max, or equality constraint to the sum of multiple variables.
        - `uniqueness`: catheter-only constraint requiring a `segment_cluster_id`.
        - `continuity`: catheter-only constraint requiring a `segment_cluster_id`.
        - `num_catheters`: catheter-only constraint on the total number of catheters.
        - `collision`: catheter-only constraint between exactly two catheter variables.
    - variable_type: The type of optimization variable constrained, either:
        - `dwell`
        - `catheter`
    - minimum: Optional lower bound for the constraint.
    - maximum: Optional upper bound for the constraint.
    - equal: Optional equality target for the constraint.
    - variable_name_ids: Optional list of variable name identifiers associated with the constraint.
        - For `bound`, exactly 1 variable name id must be provided.
        - For `sum`, at least 1 variable name id must be provided.
        - For `collision`, exactly 2 catheter variable name ids must be provided.
        - For catheter constraints, all ids must match the catheter naming pattern, which is
            f"catheter_{catheter.index+1}".
        - For dwell constraints, all ids must match the dwell naming pattern, which is
            f"dwell_{catheter.index+1}_{dwell.index+1}_{dwell.angle}".
    - segment_cluster_id: Optional cluster identifier required for `uniqueness` and `continuity`
      constraints. The name pattern is f"({cluster.depth}, {cluster.index+1})"
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        )
    constraint_type: Literal[
        "bound", "sum", "uniqueness", "continuity", "num_catheters", "collision"
        ]
    variable_type: Literal["dwell", "catheter"]
    minimum: int | float = None
    maximum: int | float = None
    equal: int | float = None
    variable_name_ids: List[PatternDwell | PatternCatheter] = None
    segment_cluster_id: PatterCluster = None
    parent_catheter_name_ids: List[PatternCatheter] = None

    @model_validator(mode="after")
    def sanity_check(self):
        # assert that the type of the contraint and the variables match
        if self.constraint_type in ["uniqueness", "continuity", "num_catheters", "collision"]:
            if self.variable_type != "catheter":
                raise ValueError("This type of constraint is only comparible with variable type catheter")
            if self.constraint_type in ["uniqueness", "continuity"]:
                if self.segment_cluster_id is None:
                    raise ValueError(f"{self.constraint_type} constraint needs segment_cluster_id")
                if self.constraint_type == "continuity" and self.parent_catheter_name_ids is None:
                    raise ValueError(f"{self.constraint_type} needs parent catheter name ids")
            if self.constraint_type == "collision":
                if len(self.variable_name_ids) != 2:
                    raise ValueError(f"Collision is only possible between two catheter variables, \
but {len(self.variable_name_ids)} was provided.")
        if self.constraint_type == "bound":
            if len(self.variable_name_ids) != 1:
                raise ValueError(f"Only provide one variable name id when binding a variable's value.")
        if self.constraint_type == "sum":
            if len(self.variable_name_ids) < 1:
                raise ValueError("Provide at lease 1 variable name id for the sum constraint.")

        if (self.variable_type == "catheter"):
            if  self.constraint_type == "bound":
                if (
                    (not _is_binary_or_None(self.minimum))
                    or (not _is_binary_or_None(self.maximum))
                    or (not _is_binary_or_None(self.equal))):
                    raise ValueError(f"minimum, maximum and equality constraints for {self.name_id} \
    must be binary values (0 or 1)")
            if (self.constraint_type == "num_catheters"
                or self.constraint_type == "sum"):
                if (
                    (not (isinstance(self.minimum, int) or self.minimum is None))
                    or (not isinstance(self.maximum, int) or self.maximum is None)
                    or (not (isinstance(self.equal, int) or self.equal is None))):
                    raise ValueError(f"minimum, maximum and equality constraints for {self.name_id} \
    must be integer values")
            for name_id in self.variable_name_ids:
                if not _validate_pattern(name_id, PatternCatheter):
                    raise ValueError(f"The name Id {name_id} is invalid for the constraint {self.name_id}")

        elif self.variable_type == "dwell":
            for name_id in self.variable_name_ids:
                if not _validate_pattern(name_id, PatternDwell):
                    raise ValueError(f"The name Id {name_id} is invalid for the constraint {self.name_id}")
        else:
            raise ValueError(f"""Variable type can only be 'dwell' or 'cathter', \
but was given {self.variable_type} """)

        if self.maximum is not None:
            if self.minimum:
                if self.minimum > self.maximum:
                    raise ValueError(f"maximum value cannot be less than \
minimum value for constraint {self.name_id}")
                if self.equal is not None and self.equal > self.maximum:
                    raise ValueError(f"equality value cannot be larger than \
maximum value for constraint {self.name_id}")
        if self.equal is not None:
            if self.minimum:
                if self.equal < self.minimum:
                    raise ValueError(f"equality value cannot be less than \
minimum value for constrant {self.name_id}")
        return self

    @computed_field
    @property
    def name_id(self) -> str | List[str]:
        r"""
        ### Purpose:
        - The computed `name_id` is built automatically from the constraint
        configuration. 

        ### Output:
        - name_id := Its format depends on the constraint type:
            - `uniqueness`: `uniqueness_<segment_cluster_id>`
            - `continuity`: `continuity_<segment_cluster_id>`
            - `num_catheters`: `num_catheters`
            - `collision`: `collision_(<id1>, <id2>)`
            - `sum`: `sum_<variable_type>_[<variable_name_ids>]`
            - `bound`: `bound_<variable_type>_<variable_name_id>`
        """
        if self.constraint_type == "uniqueness":
            name_id = f"{self.constraint_type}_{self.segment_cluster_id}"
        elif self.constraint_type == "continuity":
            name_id = [f"{self.constraint_type}_{catheter_name_id}"
                       for catheter_name_id in self.variable_name_ids]
        elif self.constraint_type == "num_catheters":
            name_id = self.constraint_type
        elif self.constraint_type == "collision":
            name_id = f"{self.constraint_type}_({self.variable_name_ids[0]}, {self.variable_name_ids[1]})"
        elif self.constraint_type == "sum":
            name_id = f"{self.constraint_type}_{self.variable_type}_{self.variable_name_ids}"
        elif self.constraint_type == "bound":
            name_id = f"{self.constraint_type}_{self.variable_type}_{self.variable_name_ids[0]}"
            name_id = name_id+f"_min" if self.minimum is not None else name_id
            name_id = name_id+f"_max" if self.maximum is not None else name_id
            name_id = name_id+f"_eq" if self.equal is not None else name_id
        else:
            raise ValueError(f"Constraint type {self.constraint_type} is not valid for name_id generation.")
        return name_id

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