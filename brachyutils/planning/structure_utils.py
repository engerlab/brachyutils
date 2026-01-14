from typing import List, Literal, Union, Dict
from opentps.core.data.images import ROIMask
from opentps.core.data import DVH, ROIContour
import numpy as np
import warnings
from brachyutils.dose.dose_utils import BrachyDose
from brachyutils.types import Optimization_Config
class BrachyStructure:
    r"""
    ### Purpose:
    - this class holds the information regarding a structure inside a brachytherapy
    treatment plan.
    ### Attributes:
    #### Basic Attributes
    - name:str
    - mask: ROIMask
    - is_target: bool
    #### DVH Attributes:
    - in_dvh: bool
    - dvh_metric_goals: Dict[str, float]
    - dvh_metrics_observed: Dict[str, float]
    - dvh_obj: opentps.core.data.DVH
    #### Uncertainty Attributes:
    - uvh
    - uncertainty_mean
    - uncertainty_std
    - uncertainty_max
    - uncertainty_min
    #### Optimization Attributes:
    - optimization_config:Optimization_Config := the optimization config object for the structure. see optim_utils.py  
    ### Functions:
        - get_dvh_metric(combined_dose:BrachyDose)
        - to_dict(export_format:str)
    """

    def __init__(
        self,
        name: str = None,
        mask: ROIMask | ROIContour = None,
        is_target: bool = None,
        in_dvh: bool = None,
        dvh_metric_goals: Dict[str, float] = None,
        optimization_config: Optimization_Config = None,
    ) -> None:
        r"""
        ### Purpose:
        - To initialize the BrachyStructure object.
        ### Inputs:
        - name:str := the name of the structure.
        - mask:ROIMask | ROIContour := the mask or contour of the structure.
        - is_target:bool := flag to indicate whether the structure is a target volume or not.
        - in_dvh:bool := flag to indicate whether the structure is included in the dose volume histogram.
        - dvh_metric_goals:Dict[str, float] := a dictionary of DVH metrics and their clinical goals.
        V_{#Gy|%}(organName), where # represents the numerical threshold and "|" is or. For example D95%(organName).
        - optimization_config:Optimization_Config := the optimization config object for the structure. see optim_utils.py 
        ### Outputs:
        - Void := will initialize the BrachyStructure object
        ### Dependencies:
        - opentps.core.data.ROIMask
        - opentps.core.data.DVH
        """
        self.name = name
        self.mask = mask
        self.is_target = is_target

        # dose volume histogram
        self.in_dvh:bool = in_dvh
        self.dvh_metric_goals: Dict[str, float] = None
        # self.dvh_metric_name: str = None
        # self.dvh_metric_clinical_goal: float = None
        self.dvh_metrics_observed: Dict[str, float] = None
        self.dvh_obj: DVH = None

        # uncertainty volume histogram
        self.uvh: np.array = None
        self.uncertainty_mean: float = None
        self.uncertainty_std: float = None
        self.uncertainty_max: float = None
        self.uncertainty_min: float = None

        # optimization attributes
        self.optimization_config: Optimization_Config = None
        # # simulation attributes
        # self.density: float = None  # 0
        # self.density_mode: str = None  # ""
        # self.material: str = None  # "CT Material"
        self.in_dvh = in_dvh
        if self.in_dvh:
            if dvh_metric_goals is None:
                raise ValueError(
                    """Please provide BrachyStructure with a dictionary of multiple metrics and goals"""
                    )
            self.dvh_metric_goals = dvh_metric_goals
            assert np.all([self.name.lower() in dvh_metric_name.lower() for dvh_metric_name in self.dvh_metric_goals.keys()]),\
             "name should be in dvh metric name enclosed by paranthesis"
        if optimization_config is not None:
            self.set_optimization_config(optimization_config)

    def get_dvh_metric(
        self,
        combined_dose: BrachyDose,
        prescription_dose: float = None,
        return_percentage: bool = False,
        body_contour: ROIContour = None,
        ) -> Dict[str, float]:
        r"""
        ### Purpose:
        - To calculate the DVH metric for the structure given the combined dose.
        The mask contour and DVH metrics should be set before calling this function.
        We expect the the dvh metric name to be in the format of "D#cc(organName)",
        "D#%(organName)", "V#Gy(organName)" or "V#%(organName)", HI(organName) or CI(organName),
        where # is the threshold value, for example "D95%(organName)". HI is for homogeniety index
        and CI is for conformity index implemented by OpenTPS.
        ### Inputs:
        - combined_dose := the combined dose object for the patient.
        - prescription_dose := the prescribed dose to the target volume (PTV or CTV).
        - return_percentage := if true, the value of the dvh metric is normalized to
        the prescription dose for Dcc or D% and to the volume of the organName for VGy or V%.
        - body_contour := the body contour is needed for conformity index calculation.
        If the body contour is not provided, the conformity index will not be calculated.
        ### Outputs:
        - Void := will update the BrachyStructure.dvh_metrics_observed dictionary and
        BrachyStructure.dvh_obj attributes. Will also update the last calculated value
        to BrachyStructure.dvh_metric_observed for backward compatibility (deprecated).
        """
        assert self.mask is not None, "mask is not loaded"
        assert any(self.dvh_metric_goals), "dvh metric goals are not set"
        #assert (
        #    self.dvh_metric_clinical_goal is not None
        #), "dvh metric clinical goal is not set"
        assert isinstance(
            combined_dose, BrachyDose
        ), "combined dose is not a BrachyDose object"
        self.dvh_obj = DVH(
            self.mask,
            combined_dose.dose_image,
            prescription=prescription_dose,
            # maxDVH=combined_dose.dose_image.imageArray.max(), XXX if the max dose is veyr large, it'll break the histogram
            )
        self.dvh_metrics_observed = {}

        for dvh_metric_name in self.dvh_metric_goals.keys():
            metric_string = dvh_metric_name.split("(")[0]

            if metric_string.startswith("D"):
                if "%" in metric_string:
                    threshold = float(metric_string.split("%")[0].split("D")[-1])
                    self.dvh_metrics_observed[dvh_metric_name] = self.dvh_obj.computeDx(threshold, return_percentage)
                elif "cc" in metric_string:
                    threshold = float(metric_string.split("cc")[0].split("D")[-1])
                    self.dvh_metrics_observed[dvh_metric_name] = self.dvh_obj.computeDcc(threshold, return_percentage)
                else:
                    raise ValueError(
                        "invalid name for DVH metric name. \
                        The metrics starting with 'D' should have percent sign (%) or cc.\
                        for example 'D95%(organ name)' or 'D2cc(organ name)'"
                    )

            elif metric_string.startswith("V"):
                if "%" in metric_string:
                    threshold = float(metric_string.split("%")[0].split("V")[-1])
                    self.dvh_metrics_observed[dvh_metric_name] = self.dvh_obj.computeVx(threshold, return_percentage)
                elif "Gy" in metric_string:
                    threshold = float(metric_string.split("Gy")[0].split("V")[-1])
                    self.dvh_metrics_observed[dvh_metric_name] = self.dvh_obj.computeVg(threshold, return_percentage)
                else:
                    raise ValueError(
                        "invalid name for DVH metric name. \
                        The metrics starting with 'V' should have percent sign (%) or Gy.\
                        for example 'V95%(organ name)' or 'V2Gy(organ name)'"
                    ) 
            elif metric_string.startswith("HI"):
                self.dvh_metrics_observed[dvh_metric_name] = self.dvh_obj.homogeneityIndex()
            elif metric_string.startswith("CI"):
                if body_contour is None:
                    raise ValueError("body_contour should be defined to compute the conformity index")
                elif isinstance(body_contour, ROIMask):
                    assert np.allclose(body_contour.gridSize, combined_dose.dose_image.gridSize), \
                    f"body contour grid size does not match dose grid size, {body_contour.gridSize} vs {combined_dose.dose_image.gridSize}"
                else:
                    # body contour is ROIContour, it's good to go
                    pass
                self.dvh_metrics_observed[dvh_metric_name] = self.dvh_obj.conformityIndex(body_contour)
            else:
                raise ValueError(
                    "invalid name for DVH metric name. \
                    The metric should should start with D followed by cc or %, or V followed by Gy or %.\
                    or HI for homogeniety index or CI for conformityIndex"
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
                "dose_voxel_goal": 0,
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
                "dose_voxel_goal": self.dose_voxel_goal,
                "dvhConstraints": "",
                "in_dvh": self.in_dvh,
                "linear_weight": self.penalty_weight_linear,
                "material": self.material,
                "max_dose": self.max_dose,
                "min_dose": self.min_dose,
                "name": self.name,
                "quadratic_weight": self.penalty_weight_quadratic,
                "type": "Target volume" if self.is_target else "Organ at risk",
                "uniformity_weight": self.penalty_weight_uniformity,
            }

    def info(self):
        print(self.to_dict("RapidBrachy"))

    def set_optimization_config(
        self,
        optimzation_config: Optimization_Config=None,
        **kwargs
        ) -> None:
        r"""
        ### Purpose:
        - To prepare the BrachyStructure object for optimization.
        ### Inputs:

        - Void := will update the BrachyStructure object with the optimization id and
        will set the penalty weights to 1.0.
        """
        from brachyutils.planning.optimization.optim_utils import Optimization_Config
        if optimzation_config is not None:
            self.optimization_config = optimzation_config
        else:
            if kwargs is None:
                raise ValueError("Please provide optimization config")
            self.optimization_config = Optimization_Config(kwargs)
