# from abc import ABC, abstractmethod
from typing import List, Callable, Any
from brachyutils.planning.plan_utils import BrachyPlan, BrachyStructure
from pydantic import BaseModel

class Variable(BaseModel):
    """
    ### Purpose:
    - A class to represent a variable in the dwell time optimization problem.
    ### Attributes:
    - name:str := references the catheter_number and dwell position number in the format
    catheter_{catheter_number}_dwell_{dwell_position_number}
    - value:float := The initial value of the variable.
    - lower_bound:float := The lower bound of the variable.
    - upper_bound:float := The upper bound of the variable.
    """
    name: str
    value: float = None
    lower_bound: float = None
    upper_bound: float = None

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
    - variables: The set of the variables to be optimized. In HDR brachy, dwell times and catheter positions
    - constraints: A set of relationships between the variables that should not be violated.
    In HDR brachy, we want all dwell times to be positive and sometimes have upper or lower bounds.
    - penalty_function: A function that states how good a set of variables are.
    - solver:str := The name   
    - model: The object that incorporates all the attributes above to output the optimal value for
    each variable
    ### Functions:
    - get_model_from_plan()
    - run()
    """
    model_config = {
        "arbitrary_types_allowed": True
    }
    plan: BrachyPlan
    solver: str = None
    variables: List[Variable] = None
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
        self.variables = self.get_variables_from_plan(plan=self.plan)
        self.constraints = self.get_constraints_from_plan(plan=self.plan)
        self.penalty_function = self.get_penalty_function_from_plan(plan=self.plan)
        self.model = self.make_model(
            variables=self.variables,
            constraints=self.constraints,
            penalty_function=self.penalty_function,
            )

    def get_variables_from_plan(
        self,
        plan: BrachyPlan,
        initial_value:float=0.,
        lower_bound:float=0.,
        upper_bound:float=100) -> List[Variable]:
        r"""
        ### Purpose:
        - A function to get the variables from the plan. The variables are dwell times for each dwell positon
        inside the catehter table.
        ### Inputs:
        - plan: BrachyPlan := The plan should have a catheter table with at least one dwell position.
        - initial_value:float := The initial value of the variable. Default is 0.
        - lower_bound:float := The lower bound of the variable. Default is 0.
        - upper_bound:float := The upper bound of the variable. Default is 100.
        ### Outputs:
        - variable_list:List[Variable] := A list of variables to be optimized. The variables are the dwell times
        for each dwell position inside the catheter table.
        """
        variable_list = []
        for catheter in plan.catheter_table:
            for dwell_position in catheter.dwells:
                variable_list.append(Variable(
                    name=f"catheter_{catheter.index}_dwell_{dwell_position.index}",
                    value=initial_value,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound, 
                ))
        return variable_list

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
                
        
    
    def get_penalty_function_from_plan(self, plan: BrachyPlan) -> Callable:
        r"""
        ### Purpose:
        - A function to get the penalty function from the plan.
        """
        pass
    def make_model(
        self,
        variables: List[Variable],
        constraints: List[Constraint],
        penalty_function: Callable) -> Any:
        r"""
        ### Purpose:
        - A function to make the model from the variables, constraints, and penalty function.
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