from brachyutils.geometry.catheter_utils import Catheter
from pydantic import BaseModel, ConfigDict, computed_field, Field, field_validator
from typing import Dict, List
import numpy as np
import trimesh


from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
from brachyutils.geometry.catheter_utils.config_cathgen import Config_Angled_CathGen

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
        return f"({self.depth},{self.index+1})"

class CatheterBox(BaseModel):
    r"""
    ### Purpose:
    - A class to represent the bounding box where candidate catheter trajectories
    are defined.

    ### Attributes:
    - num_physical_catheters: int := the number of physical catheters to be inserted.
    - structure_dict: Dict[str, Trimesh] := a dictionary of structures to be considered
    for catheter trajectory optimization.
    - rotation_angle_deg: float := the rotation angle of the catheter box around the right left (X) axis (degrees).
    This value should be less than 15 degrees.
    - insertion_point_spacing_mm: float := the spacing between adjacent catheter insertion
    points on the bottom plane (mm).
    - num_decision_planes: int := the number of decision planes to be defined in the 
    catheter box. All insertion points and landing points must be on the decision planes.
    At least there are 2 planes: inferior plane and superior plane.
    - config_angle: Dict[str, Config_Angled_CathGen] | Config_Angled_CathGen | None: The angle configuartion
    for each insertion point. If a single Config_Angled_CathGen is provided, it will be 
    applied to all insertion points. If None, the default Config_Angled_CathGen() 
    will be applied to all insertion points.
    """
    # TODO: For future consider the following concepts
    # - insertion_point_margin_mm: float := the margin from the edge of the bottom plane to the first 
    # insertion point (mm).

    num_physical_catheters: int = Field(default=1, description="the number of physical catheters to be inserted.")
    structure_dict: Dict[str, trimesh.Trimesh] = Field(default_factory=dict, description="a dictionary of structures to be considered for catheter trajectory optimization.")
    rotation_angle_deg: float = Field(default=0, description="the rotation angle of the catheter box around the right left (X) axis (degrees).")
    insertion_point_spacing_mm: float = Field(default=10, description="the spacing between adjacent catheter insertion points on the bottom plane (mm).")
    num_decision_planes: int = Field(default=2, description="the number of decision planes to be defined in the catheter box.")
    config_angle: Dict[str, Config_Angled_CathGen] | Config_Angled_CathGen | None = None
    _segment_cluster_dict: Dict[str, SegmentCluster] = None
    _cached_catheter_table: CatheterTable = None

    @computed_field
    @property
    def catheter_table(self) -> CatheterTable:
        r"""
        ### Purpose:
        - To generate a catheter table object from all the segments in the catheter box.
        This will be used for dose rate generation and dose optimization.
        """
        if self._cached_catheter_table is not None:
            return self._cached_catheter_table
        else:
            pass

    def get_colliding_segments(self) -> Dict[str, List[str]]:
        r"""
        ### Purpose:
        - To generate a dictionary of colliding segments for each segment in the catheter box.
        The key is the segment name_id, and the value is a list of name_ids of segments that collide with it.
        """
        pass

    def get_segments_at_depth(self, depth: int) -> Dict[str, CatheterSegment]:
        r"""
        ### Purpose:
        - To get all the segments at a specific depth in the catheter box.
        The depth of the root cluster is 0, the depth of its children is 1, and so on.
        """
        pass

    def get_parent_segments(self, segment_name_id: str) -> Dict[str, CatheterSegment]:
        r"""
        ### Purpose:
        - To get all the parent segments of a specific segment in the catheter box.
        The parent segments are the segments that are on the same chain of segments
        leading to the root cluster.
        """
        pass

    def get_geometric_constraints(self) -> Dict:
        r"""
        ### Purpose:
        - To generate the geometric constraints for catheter trajectory optimization.
        The geometric constraints include the following:
        - Uniqueness constraint: only one segment from each insertion point can be selected.
        $$
            \sum_k c_k = 1\\
            \forall c_k \textrm{ with the same insertion point }
        $$
        
        - Continuity Constraint: If a segment from an inner decision plane is selected, all its
        parents on the segment chain must be selected.
        $$
            e_F (\sum_k^{F-1} c_k) = (F-1) c_{F} \\
            \forall c_k \textrm{ on the same chain of segments with length of F}, \quad e_F \in \{0,1\}
        $$
        
        - Collision Constraint: If a segment is selected, all segments that collide with it must not be selected.
        $$
            \sum_k c_k = 1\\
            \forall c_k \textrm{ in a collision cluster}\\
        $$
        
        - Catheter Number Constraint: The total number of selected segments with depth of 0 must 
        be less than equal to the number of physical catheters to be inserted.
        $$
            \sum_k c_k \leq num_physical_catheters\\
            \forall c_k \textrm{ with depth of 0}
        $$
        """
        pass
