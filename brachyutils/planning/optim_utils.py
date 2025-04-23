from ABC import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from brachyutils.planning.plan_utils import BrachyPlan, BrachyStructure


class Optimizer(ABC):
    r"""
    ### Purpose:
    - An abstract Optimizer class to specify the common components of an optimizer class that 
    easily integrates to BrachyUtils.
    ### Attributes:
    - variables: The set of the variables to be optimized. In HDR brachy, dwell times and catheter positions
    - constraints: A set of relationships between the variables that should not be violated.
    In HDR brachy, we want all dwell times to be positive and sometimes, a dwell time to be below
    a certain threshold.
    - penalty_function: A function that states how good a set of variables are.
    - solver: The object that 
    - model: The object that incorporates all the attributes above to output the optimal value for
    each variable 
    ### Functions:
    - run()
    """
    