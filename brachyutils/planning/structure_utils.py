from typing import List, Literal, Union, Dict
from opentps.core.data.images import ROIMask
from opentps.core.data import DVH
import numpy as np
import warnings
from brachyutils.dose.dose_utils import BrachyDose

class BrachyStructure:
    r"""
    ### Purpose:
    - this class holds the information regarding a structure inside a brachytherapy
    treatment plan.
    ### Attributes:
    #### Basic Attributes
    - name:str
    - mask: ROIMask
    - target_volume: bool
    #### DVH Attributes:
    - in_dvh: bool
    - dvh_metric_name: str (deprecated)
    - dvh_metric_clinical_goal: str (deprecated)
    - dvh_metric_goals: Dict[str, float]
    - dvh_metric_observed: float (deprecated)
    - dvh_metrics_observed: Dict[str, float]
    - dvh_obj: opentps.core.data.DVH
    #### Uncertainty Attributes:
    - uvh
    - uncertainty_mean
    - uncertainty_std
    - uncertainty_max
    - uncertainty_min
    #### Optimization Attributes:
    - name_in_gurobiModel
    - bound_coordinates_in_gurobiModel
    - penalty_weight_linear
    - penalty_weight_quadratic
    - penalty_weight_uniformity
    - dose_limit
    - max_dose
    - min_dose
    #### Simulation attributes:
    - density
    - density_mode
    - material
    ### Functions:
        - get_dvh_metric(combined_dose:BrachyDose)
        - to_dict(export_format:str)
    """

    def __init__(
        self,
        name: str = None,
        mask_contour: ROIMask = None,
        target_volume: bool = None,
        in_dvh: bool = None,
        dvh_metric_goals: Dict[str, float] = None,
        dvh_metric_name: str = None,
        dvh_metric_clinical_goal: float = None,

    ) -> None:
        r"""
        ### Purpose:
        - To initialize the BrachyStructure object.
        ### Inputs:
        - name:str := the name of the structure.
        - mask_contour:ROIMask := the mask contour of the structure.
        - target_volume:bool := flag to indicate whether the structure is a target volume or not.
        - in_dvh:bool := flag to indicate whether the structure is included in the dose volume histogram.
        - dvh_metric_goals:Dict[str, float] := a dictionary of DVH metrics and their clinical goals.
        - dvh_metric_name:str := (deprecated) the name of the DVH metric in the format of "D#cc|%(organName)",
        "V#Gy|%(organName)", where # represents the numerical threshold and "|" is or for example D95%(organName).
        - dvh_metric_clinical_goal:float := (deprecated) the clinical goal for the DVH metric.
        ### Outputs:
        - Void := will initialize the BrachyStructure object
        ### Dependencies:
        - opentps.core.data.ROIMask
        - opentps.core.data.DVH
        """
        self.name: str = None
        self.mask_contour: ROIMask = None
        self.target_volume: bool = None

        # dose volume histogram
        self.in_dvh: bool = None
        self.dvh_metric_name: str = None
        self.dvh_metric_clinical_goal: float = None
        self.dvh_metric_observed: float = None
        self.dvh_metrics_observed: Dict[str, float] = None
        # self.normalized_cummulative_dvh: np.array = None
        self.dvh_obj: DVH = None

        # uncertainty volume histogram
        self.uvh: np.array = None
        self.uncertainty_mean: float = None
        self.uncertainty_std: float = None
        self.uncertainty_max: float = None
        self.uncertainty_min: float = None

        # optimization attributes
        self.name_in_gurobiModel: str = None
        self.bound_coordinates_in_gurobiModel: list = None
        self.penalty_weight_linear: float = None
        self.penalty_weight_quadratic: float = None
        self.penalty_weight_uniformity: float = None
        self.dose_limit: float = None
        self.max_dose: float = 500
        self.min_dose: float = 0

        # simulation attributes
        self.density: float = None  # 0
        self.density_mode: str = None  # ""
        self.material: str = None  # "CT Material"

        self.name = name
        self.mask_contour = mask_contour
        self.target_volume = target_volume
        self.in_dvh = in_dvh

        #JK Mar 2025
        #To simplify passing multiple DVH constraints to evaluate in parallel, we will
        #just have users pass the dvh_metric_goals dictionary containing {metric name: goal}.
        #Plan utils will separate the dictionary by structure
        #Passing a single metric string and goal will be deprecated

        if dvh_metric_goals is not None:
            self.dvh_metric_goals = dvh_metric_goals
        elif dvh_metric_name is not None:
            warnings.warn("""Passing a single DVH metric name and goal is deprecated. 
                        Please pass a dictionary of multiple metrics and goals for the structure""", DeprecationWarning)
            self.dvh_metric_name = dvh_metric_name
            #for generality, I would like to maintain that the user doesn't need to
            #pass a goal for each metric, if they're not optimizing
            if dvh_metric_clinical_goal is not None:
                self.dvh_metric_clinical_goal = dvh_metric_clinical_goal
            else:
                self.dvh_metric_clinical_goal = None
            self.dvh_metric_goals = {self.dvh_metric_name: self.dvh_metric_clinical_goal}
        else:
            raise ValueError("""Please provide BrachyStructure with a single DVH metric name and goal,
                            or a dictionary of multiple metrics and goals""")
            
        assert np.all([self.name.lower() in dvh_metric_name.lower() for dvh_metric_name in self.dvh_metric_goals.keys()]),\
             "name should be in dvh metric name enclosed by paranthesis"

    def get_dvh_metric(self, combined_dose: BrachyDose, prescription_dose: float):
        r"""
        ### Purpose:
        - To calculate the DVH metric for the structure given the combined dose.
        The mask contour and DVH metrics should be set before calling this function.
        We expect the the dvh metric name to be in the format of "D#cc(organName)",
        "D#%(organName)", "V#Gy(organName)" or "V#%(organName)", where # is the threshold
        value. for example "D95%(organName)".
        ### Inputs:
        - combined_dose := the combined dose object for the patient.
        ### Outputs:
        - Void := will update the BrachyStructure.dvh_metrics_observed dictionary and
        BrachyStructure.dvh_obj attributes. Will also update the last calculated value
        to BrachyStructure.dvh_metric_observed for backward compatibility (deprecated).
        """
        assert self.mask_contour is not None, "mask is not loaded"
        assert self.dvh_metric_name is not None or self.dvh_metric_goals is not None, "dvh metric name is not set"
        #assert (
        #    self.dvh_metric_clinical_goal is not None
        #), "dvh metric clinical goal is not set"
        assert isinstance(
            combined_dose, BrachyDose
        ), "combined dose is not a BrachyDose object"
        self.dvh_obj = DVH(self.mask_contour, combined_dose.dose_image, prescription=prescription_dose)
        self.dvh_metrics_observed = {}

        for dvh_metric_name in self.dvh_metric_goals.keys():
            metric_string = dvh_metric_name.split("(")[0]

            if "D" in metric_string:
                if "%" in metric_string:
                    threshold = float(metric_string.split("%")[0].split("D")[-1])
                elif "cc" in metric_string:
                    threshold = float(metric_string.split("cc")[0].split("D")[-1])
                else:
                    raise ValueError(
                        "invalid name for DVH metric name. \
                        The metrics starting with 'D' should have percent sign (%) or cc.\
                        for example 'D95%(organ name)' or 'D2cc(organ name)'"
                    )
                self.dvh_metrics_observed[dvh_metric_name] = self.dvh_obj.computeDcc(threshold)
                
            elif "V" in metric_string:
                if "%" in metric_string:
                    threshold = float(metric_string.split("%")[0].split("V")[-1])
                elif "Gy" in metric_string:
                    threshold = float(metric_string.split("Gy")[0].split("V")[-1])
                else:
                    raise ValueError(
                        "invalid name for DVH metric name. \
                        The metrics starting with 'V' should have percent sign (%) or Gy.\
                        for example 'V95%(organ name)' or 'V2Gy(organ name)'"
                    )
                self.dvh_metrics_observed[dvh_metric_name] = self.dvh_obj.computeVx(threshold)
 
            else:
                raise ValueError(
                    "invalid name for DVH metric name. \
                    The metric should should start with D followed by cc or %, or V followed by Gy or %."
                )
            warnings.warn("""BrachyStructure attribute dvh_metric_observed is deprecated.
                            Please use dvh_metrics_observed dictionary instead.""", DeprecationWarning)
            self.dvh_metric_observed = list(self.dvh_metrics_observed.values())[0]

        return self.dvh_metrics_observed

    def to_dict(self, export_format: str):
        r"""
        ### Purpose:
        - To export the BrachyStructure object into a dictionary of a certain format.
        ### Inputs:
        - export_format := the export_format of the exported plan. an example is:
            - "RapidBrachy":{
                "density": 0,
                "density_mode": "",
                "dose_limit": 0,
                "dvhConstraints": "",
                "in_dvh": true,
                "linear_weight": 1,
                "material": "CT Material",
                "max_dose": 500,
                "min_dose": 0,
                "name": "BODY",
                "quadratic_weight": 1,
                "type": "" or "Target volume" or "Organ at risk",
                "uniformity_weight": 1}

            - "WebApp": Not implemented yet
        """
        if export_format == "WebApp":
            raise NotImplementedError("export to WebApp is not implemented yet")
        elif export_format == "RapidBrachy":
            return {
                "density": self.density,
                "density_mode": self.density_mode,
                "dose_limit": self.dose_limit,
                "dvhConstraints": "",
                "in_dvh": self.in_dvh,
                "linear_weight": self.penalty_weight_linear,
                "material": self.material,
                "max_dose": self.max_dose,
                "min_dose": self.min_dose,
                "name": self.name,
                "quadratic_weight": self.penalty_weight_quadratic,
                "type": "Target volume" if self.target_volume else "Organ at risk",
                "uniformity_weight": self.penalty_weight_uniformity,
            }

    def info(self):
        print(self.to_dict("RapidBrachy"))

