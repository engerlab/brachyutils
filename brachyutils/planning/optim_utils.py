# from abc import ABC, abstractmethod
from typing import List, Callable, Any
from brachyutils.planning.plan_utils import BrachyPlan, BrachyStructure
from pydantic import BaseModel
import numpy as np


class DwellTimeVariable(BaseModel):
    """
    ### Purpose:
    - A class to represent a DwellTimeVariable in the dwell time optimization problem.
    ### Attributes:
    - name:str := references the catheter_number and dwell position number in the format
    catheter_{catheter_number}_dwell_{dwell_position_number}
    - dwell_time:float := The initial dwell_time of the DwellTimeVariable.
    - lower_bound:float := The lower bound of the DwellTimeVariable.
    - upper_bound:float := The upper bound of the DwellTimeVariable.
    """

    name: str
    dwell_time: float = None
    lower_bound: float = None
    upper_bound: float = None
    coordinates: List[float] = None


class Constraint(BaseModel):
    """
    ### Purpose:
    - A class to represent a constraint in the optimization problem.
    ### Attributes:
    - name: The name of the constraint.
    - expression: The expression of the constraint.
    """

    name: str
    expression: Callable = None


class DwellTimeOptimizer(BaseModel):
    r"""
    ### Purpose:
    - An abstract dwell time optimizer class to specify the common components of a dwell time optimizer class that
    easily integrates to BrachyUtils.
    ### Attributes:
    - plan: The brachytherapy plan to be optimized. Note that the plan will be modified in place.
    - DwellTimeVariables: The set of the DwellTimeVariables to be optimized. In HDR brachy, dwell times and catheter positions
    - constraints: A set of relationships between the DwellTimeVariables that should not be violated.
    In HDR brachy, we want all dwell times to be positive and sometimes have upper or lower bounds.
    - penalty_function: A function that states how good a set of DwellTimeVariables are.
    - solver:str := The name
    - model: The object that incorporates all the attributes above to output the optimal dwell_time for
    each DwellTimeVariable
    ### Functions:
    - get_model_from_plan()
    - run()
    """

    model_config = {"arbitrary_types_allowed": True}
    plan: BrachyPlan
    solver: str = None
    DwellTimeVariables: List[DwellTimeVariable] = None
    constraints: List[Constraint] = None
    penalty_function: Callable = None
    model: Any = None

    def __init__(self, plan: BrachyPlan, solver=None):
        r"""
        ### Purpose:
        - A function to initialize the optimizer.
        ### Parameters:
        - plan: The brachytherapy plan to be optimized. Note that the plan will be modified in place.
        """
        super().__init__(plan=plan)
        self.plan = plan
        self.solver = solver
        self.DwellTimeVariables = self.get_DwellTimeVariables_from_plan(plan=self.plan)
        self.penalty_function = self.get_penalty_function_from_plan(
            plan=self.plan, dwellTimeVariables=self.DwellTimeVariables
        )
        self.constraints = self.get_constraints_from_plan(plan=self.plan)
        self.model = self.make_model(
            DwellTimeVariables=self.DwellTimeVariables,
            constraints=self.constraints,
            penalty_function=self.penalty_function,
        )

    def get_DwellTimeVariables_from_plan(
        self,
        plan: BrachyPlan,
        initial_dwell_time: float = 0.0,
        lower_bound: float = 0.0,
        upper_bound: float = 100,
    ) -> List[DwellTimeVariable]:
        r"""
        ### Purpose:
        - A function to get the DwellTimeVariables from the plan. The DwellTimeVariables are dwell times for each dwell positon
        inside the catehter table.
        ### Inputs:
        - plan: BrachyPlan := The plan should have a catheter table with at least one dwell position.
        - initial_dwell_time:float := The initial dwell_time of the DwellTimeVariable. Default is 0.
        - lower_bound:float := The lower bound of the DwellTimeVariable. Default is 0.
        - upper_bound:float := The upper bound of the DwellTimeVariable. Default is 100.
        ### Outputs:
        - DwellTimeVariable_list:List[DwellTimeVariable] := A list of DwellTimeVariables to be optimized. The DwellTimeVariables are the dwell times
        for each dwell position inside the catheter table.
        """
        DwellTimeVariable_list = []
        for catheter in plan.catheter_table:
            for dwell_position in catheter.dwells:
                DwellTimeVariable_list.append(
                    DwellTimeVariable(
                        name=f"catheter_{catheter.index}_dwell_{dwell_position.index}",
                        dwell_time=initial_dwell_time,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        coordinates=dwell_position.position,
                    )
                )
        return DwellTimeVariable_list

    def get_constraints_from_plan(self, plan: BrachyPlan) -> List[Constraint]:
        r"""
        ### Purpose:
        - A function to get the constraints from the plan. The constraints are the prescirbed dose to the voxels inside
        the target volume and the organs at risk. At minimum, the target volume should be defined in the plan.
        """
        constraint_list = []
        for structure in plan.structure_list:
            if structure.target_volume:
                pass
            else:
                pass

    def get_penalty_function_from_plan(
        self,
        plan: BrachyPlan,
        dwellTimeVariables: List[DwellTimeVariable],
        relavance_distance_mm: List[float] = [5.0, 5.0, 5.0],
    ) -> Callable:
        r"""
        ### Purpose:
        - A function to get the penalty function from the plan. The goal for the voxels inside the
        target volume is to reach the prescribed dose. the goal for the voxels in organs at risk is to
        reach zero. The dose rate maps are normalized by the prescribed dose by default. Only voxels
        that are close to the furthest dwell positions are considered.

        P = (1/prescribed_dose) * sum( p_linear_i(target) + p_quad_i(target) + p_hotspot_i(target))
            + sum( p_linear_i(oar) )

        where   p_linear_i(target) = dose_i - prescribed_dose_i if dose_i > prescribed_dose_i for i in all target volume voxels
                p_linear_i(oar) = dose_i for i in all oar voxels
                p_quad_i = (p_linear)^2
                p_hotspot_i = abs( mean(dose_i) - 2*prescribed_dose_i) if mean(dose_i) > 1.5*prescribed_dose_i
                dose_i := dose_rate_map_i * dwell_time_i

        ### Inputs:
        - plan: BrachyPlan := The plan should have a catheter table with at least one dwell position,
        a target volume defined, and the dose rate maps loaded.
        - relavance_distance_cm:List[float] := The distance from the furthest dwell position along each axis
        to consider voxels the dose rate maps. for each axis:
            inclusion_space = [
                closest_dwell_position -relavance_distance :
                furthest_dwell_position + relavance_distance
                ]

        ### Outputs:
        - penalty_function:Callable := A function that states how good a set of DwellTimeVariables are.
        The penalty function is a function of the dose rate maps and the prescribed dose.
        """
        # get the inclusion mask for the voxels to be included
        inclusion_boundaries = np.ones((3, 2))
        dwell_bounds = np.zeros((3, 2))
        for axis in [0, 1, 2]:
            dwell_bounds[axis, 0] = np.min(
                [dwelltime.coordinates[axis] for dwelltime in dwellTimeVariables]
            )
            dwell_bounds[axis, 1] = np.max(
                [dwelltime.coordinates[axis] for dwelltime in dwellTimeVariables]
            )
            inclusion_boundaries[axis, 0] = (
                dwell_bounds[axis, 0] - relavance_distance_mm[axis]
            )
            inclusion_boundaries[axis, 1] = (
                dwell_bounds[axis, 1] + relavance_distance_mm[axis]
            )
            # if the inclusion bound is outside the dose image, set it to the dose image bounds
            if (
                inclusion_boundaries[axis][0]
                < plan.combined_dose.dose_image.origin[axis]
            ):
                inclusion_boundaries[axis][0] = plan.combined_dose.dose_image.origin[axis]
            if (
                inclusion_boundaries[axis][1]
                > plan.combined_dose.dose_image.origin[axis]
                + plan.combined_dose.dose_image.gridSizeInWorldUnit[axis]
            ):
                inclusion_boundaries[axis][1] = (
                    plan.combined_dose.dose_image.origin[axis]
                    + plan.combined_dose.dose_image.gridSizeInWorldUnit[axis]
                )
                
        
        print("debug here")

    def make_model(
        self,
        DwellTimeVariables: List[DwellTimeVariable],
        constraints: List[Constraint],
        penalty_function: Callable,
    ) -> Any:
        r"""
        ### Purpose:
        - A function to make the model from the DwellTimeVariables, constraints, and penalty function.
        """
        pass

    def run(self):
        r"""
        ### Purpose:
        - A function to run the optimizer.
        """
        pass

    def get_optimized_plan_from_model(self) -> BrachyPlan:
        r"""
        ### Purpose:
        - A function to get the optimized plan from the model after the optimizaton is done.
        """
        pass
