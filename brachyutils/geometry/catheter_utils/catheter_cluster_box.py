from pathlib import Path
from pydantic import BaseModel, ConfigDict, computed_field, Field, field_validator
from typing import Dict, List, Union
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
    The format is (SegmentCluster.depth, SegmentCluster.index+1)
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
        r"""
        The unique identifier for this Segment. The format is
        ({SegmentCluster.depth}, {SegmentCluster.index+1})_{Segment.index+1}
        """
        
        return f"{self.cluster_name_id}_{self.index+1}"

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

    @computed_field
    @property
    def catheter_name_ids(self) -> List[str]:
        cath_name_ids = []
        for segment in self.segment_dict.values():
            cath_name_ids.append(segment.catheter_name_id)
        return cath_name_ids

    def __iter__(self):
        for segment in self.segment_dict.values():
            yield segment

    def __len__(self):
        return len(self.segment_dict)

    def __getitem__(self, indices: int | slice | str) -> Segment | Dict[int, Segment]:
        r"""
        ### Purpose:
        - To get a subset of the segments in this cluster.
        
        ### Inputs:
        - `indicies`: int | slice | str := Depending on the input type, return the following. 
            - `int`: Get a single segment by its index in this cluster.
            - `slice`: Get a set of segments by their range of indicies.
            - `str`: Get a single segment by its name_id. The segment name_id is of the format:
            ({SegmentCluster.depth}, {SegmentCluster.index+1})_{Segment.index+1}
        """
        if isinstance(indices, int):
            return self.segment_dict[indices]
        elif isinstance(indices, str):
            cluster_name_id, segment_index_plus_1 = indices.split("_")
            if cluster_name_id != self.name_id:
                raise ValueError(f"Wrong cluster is being queried. This is cluster {self.name_id}\
but {cluster_name_id} was requested.")
            return self.segment_dict[int(segment_index_plus_1)-1]
        elif isinstance(indices, slice):
            segments_sub_dict = defaultdict(Segment)
            indices = list(range(*indices.indices(len(self.segment_dict))))
            for i in indices:
                segments_sub_dict[i] = list(self.segment_dict.values())[i]
            return segments_sub_dict
        else:
            raise ValueError("indicies format is not correct.")

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
    cluster_dict: Dict[str, SegmentCluster] = Field(default=None)
    # index of a segment in the box, when we get all segments
    # the keys are the depth, cluster index, segment index
    _cached_segment_dict: Dict[int, Segment] = None
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
        self.cluster_dict = get_clusters_from_planes(plane_dict=self._plane_dict)
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
            for idx, segment in enumerate(self.all_segments_list):
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
    def all_segments_dict(self) -> Dict[int, Segment]:
        r"""
        ###  Purpose:
        - To return a dictionary with all the catheter segments gathered from all the clusters.
        The keys are [depth][cluster index][segment index]
        """
        if self._cached_segment_dict is not None:
            return self._cached_segment_dict
        else:
            self._cached_segment_dict = defaultdict(dict)
            for cluster in self.cluster_dict.values():
                self._cached_segment_dict[cluster.depth][cluster.index] = cluster.segment_dict 
            return self._cached_segment_dict

    @ computed_field
    def all_segments_list(self) -> List[Segment]:
        outer = self.all_segments_dict
        flat = []
        for level1 in outer.values():
            for level2 in level1.values():
                for val in level2.values():
                    flat.append(val)
        return flat

    @computed_field
    def num_segments(self) -> int:
        return len(self.all_segments_list)

    def __iter__(self):
        for cluster in self.cluster_dict.values():
            yield cluster

    def __len__(self):
        return len(self.cluster_dict)

    def __getitem__(
        self,
        indices: int | slice | str) -> Union[
            Segment, SegmentCluster, Dict[str, Segment], Dict[int, SegmentCluster]
        ]:
        r"""
        ### Purpose:
        - To get the right clusters or segments based on the indices provided.

        ### Inputs:
        - indices: int | slice | str := The indices could be in many forms:
            1. [str] : Will return the cluster if the string matches "({cluster.depth},{cluster.index+1})".
            If the format matches "({cluster.depth},{cluster.index+1})_{segment.index+1}", it will return
            the corresponding segment.
            2. [int]: This will get you all the clusters with cluster.depth == indices
            3. [int][int]: This will get you all the segments at cluster.depth, cluster.index == indices
            4. [int][int][int]: This will get you a single segment at
            cluster.depth, cluster.index, segment.index == indices
            5. [slice]: This will get you all the clusters with cluster.depth in indices
            6. [slice][slice]: This will get you all the clusters with cluster.depth, cluster.index in indices
            7. [slice][slice][slice]: This will get you all the segments with
            cluster.depth, cluster.index, segment.index in indices
        """

        # ------------------------------------------------------------------ #
        # 1. String → SegmentCluster or Segment                               #
        # ------------------------------------------------------------------ #
        if isinstance(indices, str):
            # "(depth,index+1)_segment+1"  →  Segment
            if "_" in indices:
                cluster_key, seg_part = indices.rsplit("_", 1)
                cluster_key = cluster_key.strip()
                if cluster_key not in self.cluster_dict:
                    raise KeyError(
                        f"No cluster with key '{cluster_key}' in ClusterBox."
                    )
                seg_idx = int(seg_part) - 1   # 1-based → 0-based
                return self.cluster_dict[cluster_key].segment_dict[seg_idx]

            # "(depth,index+1)"  →  SegmentCluster
            key = indices.strip()
            if key not in self.cluster_dict:
                raise KeyError(f"No cluster with key '{key}' in ClusterBox.")
            return self.cluster_dict[key]

        # ------------------------------------------------------------------ #
        # 2. Integer → depth level                                            #
        # ------------------------------------------------------------------ #
        if isinstance(indices, int):
            depth = indices
            # Returns a _DepthView proxy so [int][int] and [int][int][int] chain works
            return _DepthView(
                segments_dict=self.all_segments_dict,
                depth=depth
            )

        # ------------------------------------------------------------------ #
        # 3. Slice → across depths                                            #
        # ------------------------------------------------------------------ #
        if isinstance(indices, slice):
            all_depths = sorted(self.all_segments_dict.keys())
            selected_depths = all_depths[indices]
            return _SliceView(
                segments_dict=self.all_segments_dict,
                selected_depths=selected_depths
            )

        raise TypeError(
            f"Unsupported index type '{type(indices).__name__}'. "
            f"Expected int, slice, or str."
        )

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
        
    def get_parent_segments(self, segment_name_id: str) -> Dict[str, Segment]:
        r"""
        ### Purpose:
        - To get all the parent segments of a specific segment in the catheter box.
        The parent segments are the segments that are on the same chain of segments
        leading to the root cluster.
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
            [seg.line for seg in self.all_segments_list]
        )
        segment_lines_to_ply(
            out_ply_dir=out_ply_dir,
            point_pairs=all_segments_lines
        )

def get_clusters_from_planes(
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


class _DepthView:
    r"""
    Returned by ClusterBox[int_depth].
    Wraps all_segments_dict[depth] and supports:
        [int]   → Dict[int, Segment]   (all segments of that cluster)
        [int][int] → Segment
        [slice] → Dict[int, Dict[int, Segment]]
        [slice][slice] → Dict[int, Segment]  (flat)
    """

    def __init__(self, segments_dict: dict, depth: int):
        if depth not in segments_dict:
            raise KeyError(f"No clusters found at depth {depth}.")
        self._data = segments_dict[depth]   # {cluster_idx: {seg_idx: Segment}}

    # Make it behave like a dict when the caller doesn't chain further
    def __repr__(self):
        return repr(self._data)

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def keys(self):
        return self._data.keys()

    def __getitem__(self, indices: int | slice) -> Union[
        Segment, Dict[int, Segment], "_ClusterIndexView"
    ]:
        # [int]   → Dict[int, Segment]  i.e. all segments of cluster `indices`
        if isinstance(indices, int):
            cluster_idx = indices
            if cluster_idx not in self._data:
                raise KeyError(
                    f"No cluster at index {cluster_idx} for this depth."
                )
            return _ClusterIndexView(self._data[cluster_idx])

        # [slice] → filtered {cluster_idx: {seg_idx: Segment}}
        if isinstance(indices, slice):
            all_cluster_keys = sorted(self._data.keys())
            selected_keys = all_cluster_keys[indices]
            subset = {k: self._data[k] for k in selected_keys}
            return _ClusterSliceView(subset)

        raise TypeError(
            f"Unsupported index type '{type(indices).__name__}' "
            f"at cluster level. Expected int or slice."
        )


class _ClusterIndexView:
    r"""
    Returned by ClusterBox[depth][cluster_idx].
    Wraps {seg_idx: Segment} and supports a further [int] or [slice].
    """

    def __init__(self, seg_dict: dict):
        self._data = seg_dict   # {seg_idx: Segment}

    def __repr__(self):
        return repr(self._data)

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def keys(self):
        return self._data.keys()

    def __getitem__(self, indices: int | slice) -> Union[Segment, Dict[int, Segment]]:
        # [int]   → single Segment
        if isinstance(indices, int):
            if indices not in self._data:
                raise KeyError(f"No segment at index {indices}.")
            return self._data[indices]

        # [slice] → {seg_idx: Segment}
        if isinstance(indices, slice):
            all_seg_keys = sorted(self._data.keys())
            selected_keys = all_seg_keys[indices]
            return {k: self._data[k] for k in selected_keys}

        raise TypeError(
            f"Unsupported index type '{type(indices).__name__}' "
            f"at segment level. Expected int or slice."
        )


class _ClusterSliceView:
    r"""
    Returned by ClusterBox[depth][cluster_slice].
    Wraps a subset of {cluster_idx: {seg_idx: Segment}} and supports
    a further [slice] to filter segments.
    """

    def __init__(self, data: dict):
        self._data = data   # {cluster_idx: {seg_idx: Segment}}

    def __repr__(self):
        return repr(self._data)

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def keys(self):
        return self._data.keys()

    def __getitem__(self, indices: slice) -> Dict[int, Segment]:
        # [slice] → flat {seg_idx: Segment} across all selected clusters
        if isinstance(indices, slice):
            flat = {}
            for seg_dict in self._data.values():
                all_seg_keys = sorted(seg_dict.keys())
                selected_keys = all_seg_keys[indices]
                flat.update({k: seg_dict[k] for k in selected_keys})
            return flat

        raise TypeError(
            f"Unsupported index type '{type(indices).__name__}' "
            f"at segment level. Expected slice."
        )


class _SliceView:
    r"""
    Returned by ClusterBox[depth_slice].
    Wraps a subset of depths and supports:
        [slice]        → _ClusterSliceView across all selected clusters
        [slice][slice] → flat {seg_idx: Segment}
    """

    def __init__(self, segments_dict: dict, selected_depths: list):
        self._data = {
            d: segments_dict[d]
            for d in selected_depths
            if d in segments_dict
        }   # {depth: {cluster_idx: {seg_idx: Segment}}}

    def __repr__(self):
        return repr(self._data)

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def keys(self):
        return self._data.keys()

    def __getitem__(self, indices: slice) -> "_ClusterSliceView":
        if isinstance(indices, slice):
            # Merge all cluster dicts from all selected depths
            merged_clusters = {}
            for cluster_dict in self._data.values():
                all_keys = sorted(cluster_dict.keys())
                selected_keys = all_keys[indices]
                merged_clusters.update(
                    {k: cluster_dict[k] for k in selected_keys}
                )
            return _ClusterSliceView(merged_clusters)

        raise TypeError(
            f"Unsupported index type '{type(indices).__name__}' "
            f"at cluster level. Expected slice."
        )