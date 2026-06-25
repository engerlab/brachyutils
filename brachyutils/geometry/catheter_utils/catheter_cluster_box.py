from pathlib import Path
from pydantic import BaseModel, ConfigDict, computed_field, Field, field_validator
from typing import Dict, List
import numpy as np
import trimesh
from collections import defaultdict

from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable, Catheter
from brachyutils.geometry.catheter_utils.config_cathgen import Config_Angled_CathGen, Decision_Plane
from brachyutils.geometry.catheter_utils.catheter_cluster_box_utils import (
    generate_candidate_segments,
    decision_planes_to_ply,
    segment_lines_to_ply,
    )

class Segment(BaseModel):
    r"""
    ### Purpose:
    - Represents a single catheter trajectory segment between a departure point and a landing point.
    - Used to group candidate catheter paths within a shared insertion point cluster.

    ### Attributes:
    - index: int := the index of the segment within its cluster.
    - cluster_name_id: str := identifier for the cluster that this segment belongs to.
    - line: List[List[float]] := the departure and landing points of the segment in patient coordinates.
      The value is expected to be a list containing two 3D points.
    """
    
    index:int = Field(..., description="The index of the segment in its cluster")
    cluster_name_id: str = Field(..., description="The cluster that this segment stems from")
    line:List[List[float]] = Field(..., description="The departure and landing point of the \
segment in patients coordinates.")
    # these attributes will be set and used later when interacting with the optimization model.
    catheter_name_id:int = None
    # dwell_index_list:List[int] = None

    @computed_field
    @property
    def name_id(self) -> str:
        return f"{self.cluster_name_id}_{self.name_id+1}"

class SegmentCluster(BaseModel):
    r"""
    ### Purpose:
    - A class to represent a cluster of segments for each catheter insertion position.
    This is useful for case where we are optimizing multiple catheter trajectories
    for the same insertion point.
    
    ### Attributes:
    - depth: int := the depth of the segment cluster. Clusters can be lead to other clusters like a tree structure.
    The depth of the root cluster is 0, the depth of its children is 1, and so on.
    - segment_dict: Dict[int, Segment] := a dictionary of segments inside the cluster.
    The key is the catheter index, and the value is the Segment object.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,)
    index: int
    depth: int = Field(default=0)
    segment_dict: Dict[int, Segment] = Field(default=None)

    @field_validator('depth')
    @classmethod
    def validate_depth(cls, v):
        if v < 0:
            raise ValueError("Depth must be a non-negative integer")
        return v

    @computed_field
    @property
    def insert_position(self) -> np.ndarray:
        return self.segment_dict[0].line[0]

    @computed_field
    @property
    def name_id(self) -> str:
        return f"({self.depth},{self.index+1})"

class ClusterBox(BaseModel):
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
    - oar_collision_margin_mm: float := the collision margin between catheter segments and
    organs at risk (OARs) (mm).
    - segment_collision_margin_mm: float := the collision margin between catheter segments (mm).
    - target_structure_names:  List[str] := "The list of the names of the target structures;
    Usually CTV or PTV.")  
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,)

    # # user defined attributes
    num_physical_catheters: int = Field(default=1, description="the number of physical catheters to be inserted.")
    structure_dict: Dict[str, trimesh.Trimesh] = Field(default_factory=dict, description="a dictionary of \
structures to be considered for catheter trajectory optimization.")
    rotation_angle_deg: float = Field(default=0, description="the rotation angle of the catheter box \
around the right left (X) axis (degrees).")
    insertion_point_spacing_mm: float = Field(default=10, description="the spacing between adjacent \
catheter insertion points on the bottom plane (mm).")
    num_decision_planes: int = Field(default=2, description="the number of decision planes to be \
defined in the catheter box.")
    config_angle: Dict[str, Config_Angled_CathGen] | Config_Angled_CathGen | None = Field(
        default=None,
        description="The angle configuartion for each insertion point. \
If a single Config_Angled_CathGen is provided, it will be applied to all \
insertion points. If None, the default Config_Angled_CathGen() will be applied \
to all insertion points.")
    oar_collision_margin_mm: float = Field(default=0, description="the collision margin between \
catheter segments and organs at risk (OARs) (mm).")
    segment_collision_margin_mm: float = Field(default=0, description="the collision margin between \
catheter segments (mm).")
    target_structure_names: List[str] = Field(..., description="The list of the names of the target \
structures; Usually CTV or PTV.")
    box_margin_mm: float = Field(default=0, description="The margin between the box boundaries and the OARs")
    
    # # internal attributes
    segment_cluster_dict: Dict[str, SegmentCluster] = Field(default=None)
    _cached_catheter_table: CatheterTable = None

    _plane_dict: Dict[str, Decision_Plane]

    def model_post_init(self, __context):
        r"""
        ### Purpose:
        - To validate and generate the segment clusters for the catheter box 
        after the object is initialized.
        
        ### Steps:
        1. Generate the segment clusters for the catheter box based on the user defined attributes.
        2. Ensure the number of physical catheters is less than the number of insertion points.
        """
        # # Initialize the catheter box 
        # # based on the structure dict, box rotation angle, insertion point spacing 
        _, self._plane_dict = generate_candidate_segments(
            mesh_dict=self.structure_dict,
            insertion_point_spacing_mm=self.insertion_point_spacing_mm,
            oar_danger_dist_mm=self.oar_collision_margin_mm,
            target_structure_names=self.target_structure_names,
            config_angled_cathgen=self.config_angle,
            bb_rotation_angle_deg=self.rotation_angle_deg,
            bb_num_planes=self.num_decision_planes,
            bb_margin_mm = self.box_margin_mm,
        )
        # # get the segment clusters and segments from plane dict.
        self.segment_cluster_dict = get_segment_cluster_from_planes(plane_dict=self._plane_dict)
        return self

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
            all_catheters = []
            for idx, segment in enumerate(self.all_segments):
                catheter = Catheter(
                    index=idx,
                    digitization_points=segment.line
                )
                segment.catheter_name_id = catheter.name_id
                all_catheters.append(catheter)

            self._cached_catheter_table = CatheterTable(
                catheters_dict=all_catheters
            )
            return self._cached_catheter_table

    @computed_field
    def all_segments(self) -> List[Segment]:
        r"""
        ###  Purpose:
        - To return a list with all the catheter segments gathered from all the clusters.
        """
        all_point_pairs = []
        for cluster in self.segment_cluster_dict.values():
            for segment in cluster.segment_dict.values():
                all_point_pairs.append(segment)
        return all_point_pairs

    def get_colliding_segments(self) -> Dict[str, List[str]]:
        r"""
        ### Purpose:
        - To generate a dictionary of colliding segments for each segment in the catheter box.
        The key is the segment name_id, and the value is a list of name_ids of segments that collide with it.
        """
        pass

    def get_segments_at_depth(self, depth: int) -> Dict[str, Segment]:
        r"""
        ### Purpose:
        - To get all the segments at a specific depth in the catheter box.
        The depth of the root cluster is 0, the depth of its children is 1, and so on.
        """
        pass

    def get_parent_segments(self, segment_name_id: str) -> Dict[str, Segment]:
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

    def to_ply(self, out_ply_dir:str | Path):
        r"""
        Write the planes and the segment to .PLY files for visualization
        """
        # write the planes to ply
        decision_planes_to_ply(
            out_ply_dir=out_ply_dir,
            decision_plane_dict=self._plane_dict
        )
        all_segments_lines = np.array(
            [seg.line for seg in self.all_segments]
        )
        segment_lines_to_ply(
            out_ply_dir=out_ply_dir,
            point_pairs=all_segments_lines
        )

def get_segment_cluster_from_planes(
    plane_dict:Dict[int, Decision_Plane]) -> Dict[str, SegmentCluster]:
    r"""
    ### Purpose:
    - This function creates segment clusters from plane dictionaries that have been filled
    with segments.
    """
    cluster_dict = defaultdict(SegmentCluster)
    for plane in plane_dict.values():
        if plane.depth == len(plane_dict) - 1:
            break 
        all_insert_points = [p0 for p0, _ in plane.segment_lines]
        idx = sorted(np.unique(all_insert_points, axis=0, return_index=True)[1])
        idx = idx + [len(all_insert_points)]
        for i, j in enumerate(idx):
            if j == idx[-1]:
                break
            cluster = SegmentCluster(
                index=i,
                depth=plane.depth,
            )
            segment_dict = defaultdict()
            # Avoid clusters that have no valid segments
            # segments can be removed due to collision with 
            # oars or falling out of a landing plane.
            segments_in_a_cluster = plane.segment_lines[j:idx[i+1]]
            if len(segments_in_a_cluster) == 0:
                continue
            for k, line in enumerate(segments_in_a_cluster):
                segment_dict[k] = Segment(
                    cluster_name_id=cluster.name_id,
                    index=k,
                    line=line
                    )
            cluster.segment_dict = segment_dict
            cluster_dict[cluster.name_id] = cluster
    return cluster_dict
