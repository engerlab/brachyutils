from typing import Dict
from collections import defaultdict

from brachyutils.geometry.catheter_utils.catheter_cluster_box import ClusterBox
from brachyutils.planning.optimization.optim_configs import Constraint_Config

def get_geometric_constraints(cluster_box:ClusterBox) -> Dict:
    r"""
    ### Purpose:
    - To generate the geometric constraints for catheter trajectory optimization from a cluster box.
    The geometric constraints include the following:
    1. Uniqueness constraint: only one segment from each insertion point can be selected.
    $$
        \sum_k c_k = 1\\
        \forall c_k \textrm{ with the same insertion point }
    $$

    2. Catheter Number Constraint: The total number of selected segments with depth of 0 must 
    be less than equal to the number of physical catheters to be inserted.
    $$
        \sum_k c_k \leq num_physical_catheters\\
        \forall c_k \textrm{ with depth of 0}
    $$

    3. Continuity Constraint: If a segment from an inner decision plane is selected, all its
    parents on the segment chain must be selected.
    $$
        e_F (\sum_k^{F-1} c_k) = (F-1) c_{F} \\
        \forall c_k \textrm{ on the same chain of segments with length of F}, \quad e_F \in \{0,1\}
    $$

    4. Collision Constraint: If a segment is selected, all segments that collide with it must not be selected.
    $$
        \sum_k c_k = 1\\
        \forall c_k \textrm{ in a collision cluster}\\
    $$
    
    ### Inputs:
    - cluster_box: ClusterBox := This object contains all the candidate catheter segments, organized
    by their cluster where they stem from.
    
    ### Outputs:
    - all_constraints : Dict[str, dict] := A dictionary containing the many geometric constraints above.
    """
    catheter_table = cluster_box.catheter_table

    # # Uniqueness constraints
    # uniqness_constraints = defaultdict()
    # for cluster in cluster_box.segment_cluster_dict.values():
    #     constr = Constraint_Config(
    #         constraint_type="uniqueness",
    #         variable_type="catheter",
    #         maximum=1,
    #         segment_cluster_id=cluster.name_id,
    #         variable_name_ids=cluster.catheter_name_ids
    #     )
    #     uniqness_constraints[constr.name_id] = constr

    # # Catheter Num Constraints
    catheter_num_constraint = {}
    segments_at_depth_0 = []
    for i, segment in cluster_box.all_segments_dict[0][0].items():
        segments_at_depth_0.append(segment.catheter_name_id)

    constr = Constraint_Config(
        constraint_type="num_catheters",
        variable_type="catheter",
        maximum=cluster_box.num_physical_catheters,
        variable_name_ids=segments_at_depth_0
    )
