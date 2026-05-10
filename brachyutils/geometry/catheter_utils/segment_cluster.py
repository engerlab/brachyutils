from brachyutils.geometry.catheter_utils import Catheter
from pydantic import BaseModel, ConfigDict, computed_field, Field, field_validator
from typing import Dict, List
import numpy as np

class CatheterSegment(Catheter):
    cluster_name_id: str

    @computed_field
    @property
    def name_id(self) -> str:
        return f"{self.cluster_name_id}_{super().name_id}"

class SegmentCluster(BaseModel):
    r"""
    ### Purpose:
    - A class to represent a cluster of segments for each catheter insertion position.
    This is useful for case where we are optimizing multiple catheter trajectories
    for the same insertion point.
    
    ### Attributes:
    - depth: int := the depth of the segment cluster. Clusters can be lead to other clusters like a tree structure.
    The depth of the root cluster is 0, the depth of its children is 1, and so on.
    - segment_dict: Dict[int, Catheter] := a dictionary of segments inside the cluster.
    The key is the catheter index, and the value is the Catheter object.
    - cluster_dict: Dict[int, SegmentCluster] := a dictionary of child clusters inside the cluster.
    The key is the cluster index, and the value is the SegmentCluster object.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,)
    index: int
    depth: int = Field(default=0)
    segment_dict: Dict[int, Catheter] = Field(default_factory=dict)
    cluster_dict: Dict[int, "SegmentCluster"] = Field(default_factory=None)

    @field_validator('depth')
    @classmethod
    def validate_depth(cls, v):
        if v < 0:
            raise ValueError("Depth must be a non-negative integer")
        return v

    @computed_field
    @property
    def insertion_position(self) -> np.ndarray:
        return self.segment_dict[0].insertion_position

    @computed_field
    @property
    def name_id(self) -> str:
        return f"{self.depth}_{self.index+1}"