from typing import List, Union
import numpy as np
from pydantic import (BaseModel, ConfigDict, computed_field, model_validator,
                      SkipValidation)
import SimpleITK as sitk
from brachyutils.brachy_types import BrachyDose
from opentps.core.data.images import ROIMask
from opentps.core.processing.imageProcessing.sitkImageProcessing import imageToSITK


class DwellPosition(BaseModel):
    r"""
    ### Purpose:
    - This class holds the information regarding a dwell position.

    ### Attributes:
    - index: int := the index of the dwell position along the catheter
    - angle := angle of the IMBT shield
    - position:dict: np.array := dwell position in the patient coordinate system [x, y, z]
    - relativePos: float := distance along the catheter from the reference point. increments of x mm
    - rotation: np.array := rotation of the dwell position in the patient coordinate system [x, y, z]
    - time: float := dwell time for this dwell position
    - weight: float := ratio of this dwell time over the sum of all dwell times in all catheters.
    - catheter_index: int := the index of the catheter this dwell position belongs to.

    ### Methods:
    - validate_dwell_position()
    - name_id
    - weight()
    - to_dict()
    - get_position()
    - isin_mask()
    - set_time()
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: int
    angle: int = 0
    position: List[float] | np.ndarray
    relativePos: float
    rotation: List[float] | np.ndarray = None
    time: float = 0.0
    _time_diff: float = 0.0
    catheter_index: int = None
    gen_dose_rate: bool = True
    dose_rate: SkipValidation[BrachyDose] = None

    @model_validator(mode="after")
    def validate_dwell_position(self):
        self.position = np.array(self.position)
        if self.rotation is not None:
            self.rotation = np.array(self.rotation)
        # when we instantiate a dwell position, we set the time difference to be the same as the time,
        # so that during combined dose calculation, we calculate dose difference based on 
        # the time diff only. Time diff is only set to zero after every dose calculation.
        self._time_diff = self.time
        return self

    @computed_field
    def name_id(self) -> str:
        return f"{self.catheter_index+1}_{self.index+1}_{self.angle}"

    def weight(self, total_time: float) -> float:
        r"""
        ### Purpose:
        - To calculate the weight of the dwell position relative to a total time.
        The total time could come from the catheter or the treatment plan.

        ### Inputs:
        - self := the DwellPosition object.
        - total_time:float=None := the total time of the catheter or the treatment plan.
        if this is not provided, the weight of the dwell position will be returned.

        ### Outputs:
        - float := the weight of the dwell position.
        """
        return self.time / total_time

    def to_dict(self, total_time:float=None) -> dict:
        r"""
        ### Purpose:
        - To convert the dwell position to a dictionary.

        ### Inputs:
        - self := the DwellPosition object.

        ### Outputs:
        - dict := the dictionary containing the dwell position.
        """
        if total_time is None:
            total_time = self.time
        return {
            "index": int(self.index),
            "name_id": self.name_id,
            "catheter_index": self.catheter_index,
            "angle": float(self.angle),
            "position": list(self.position),
            "relativePos": float(self.relativePos),
            "rotation": list(self.rotation),
            "time": float(self.time),
            "weight": float(self.weight(total_time)),
        }

    def get_position(self) -> List[float]:
        r"""
        ### Purpose:
        - To get the position of the dwell position.

        ### Inputs:
        - self := the DwellPosition object.

        ### Outputs:
        - List[float] := the position of the dwell position.
        """
        return self.position

    def isin_mask(self, mask:Union[ROIMask, sitk.Image]) -> bool:
        r"""
        ### Purpose:
        - To check if the dwell position is inside a given mask.

        ### Inputs:
        - self := the DwellPosition object.
        - mask:Union[ROIMask, sitk.Image] := the mask to check if the dwell position is inside.

        ### Outputs:
        - bool := True if the dwell position is inside the mask, False otherwise.
        """
        if isinstance(mask, ROIMask):
            mask = imageToSITK(mask)
        index = mask.TransformPhysicalPointToIndex(self.position)
        in_mask = mask.GetPixel(index) > 0
        return in_mask
    
    def set_time(self, time:float) -> None:
        r"""
        ### Purpose:
        - To set the time of the dwell position and calculate the time difference from the previous time.

        ### Inputs:
        - self := the DwellPosition object.
        - time:float := the new time to set for the dwell position.

        ### Outputs:
        - None
        """
        self._time_diff = time - self.time
        self.time = time

    def __setattr__(self, name, value):
        # override the __setattr__ method to update the time difference when the time attribute is updated.
        if name == "time":
            self.set_time(value)
        else:
            super().__setattr__(name, value)