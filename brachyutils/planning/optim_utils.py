from abc import ABC, abstractmethod
from typing import List, Callable, Dict, Any, Tuple
from brachyutils.planning.plan_utils import BrachyPlan, BrachyStructure
from pydantic import BaseModel, model_validator
import inspect

class Variable(BaseModel):
    """
    ### Purpose:
    - A class to represent a variable in the optimization problem.
    ### Attributes:
    - name:str := The name of the variable.
    - value:float := The value of the variable.
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

class Optimizer(BaseModel, ABC):
    r"""
    ### Purpose:
    - An abstract Optimizer class to specify the common components of an optimizer class that 
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
    plan: BrachyPlan
    variables: List[Variable] = None
    constraints: List[Constraint] = None
    penalty_function: Callable = None
    solver: str = None
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

    @abstractmethod
    def get_variables_from_plan(self, plan: BrachyPlan) -> List[Variable]:
        r"""
        ### Purpose:
        - A function to get the variables from the plan.
        """
        pass
    @abstractmethod
    def get_constraints_from_plan(self, plan: BrachyPlan) -> List[Constraint]:
        r"""
        ### Purpose:
        - A function to get the constraints from the plan.
        """
        pass
    @abstractmethod
    def get_penalty_function_from_plan(self, plan: BrachyPlan) -> Callable:
        r"""
        ### Purpose:
        - A function to get the penalty function from the plan.
        """
        pass
    @abstractmethod
    def make_model(self):
        r"""
        ### Purpose:
        - A function to make the model from the variables, constraints, and penalty function.
        """
        pass
    @abstractmethod
    def run(self):
        r"""
        ### Purpose:
        - A function to run the optimizer.
        """
        pass
    @abstractmethod
    def get_optimized_plan_from_model(self) -> BrachyPlan:
        r"""
        ### Purpose:
        - A function to get the optimized plan from the model after the optimizaton is done.
        """
        pass