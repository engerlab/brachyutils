from typing import Dict
from collections import defaultdict

from brachyutils.geometry.catheter_utils.catheter_cluster_box import ClusterBox
from brachyutils.planning.optimization.optim_configs import Constraint_Config, Optimization_Config
from brachyutils.planning.plan_utils import BrachyPlan
from brachyutils.dose.tg43_dose_calculator import BrachyUtilsTG43 
from brachyutils.dose.dose_generation_utils import RapidBrachyTG43 
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import CatheterTableOptim_Gurobi 
from brachyutils.geometry.catheter_utils.config_cathgen import (
    Config_Angled_CathGen, Config_ClusterBox
)

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
    All segments sharing the same insertion point would have an identical continuity constraint.

    4. Collision Constraint: If a segment is selected, all segments that collide with it must not be selected.
    $$
        \sum_k c_k = 1\\
        \forall c_k \textrm{ in a collision cluster}\\
    $$
    The collision constraint is on a pair of colliding segments.

    ### Inputs:
    - cluster_box: ClusterBox := This object contains all the candidate catheter segments, organized
    by their cluster where they stem from.
    
    ### Outputs:
    - all_constraints : Dict[str, dict] := A dictionary containing the many geometric constraints above.
    """
    catheter_table = cluster_box.catheter_table

    # # Uniqueness constraints
    uniqness_constraints = defaultdict(Constraint_Config)
    for cluster in cluster_box.cluster_dict.values():
        constr = Constraint_Config(
            constraint_type="uniqueness",
            variable_type="catheter",
            maximum=1,
            segment_cluster_id=cluster.name_id,
            variable_name_ids=[f"catheter_{name_id}" for name_id in cluster.catheter_name_ids]
        )
        uniqness_constraints[constr.name_id] = constr

    # # Catheter Num Constraints
    catheter_num_constraint = {}
    segments_at_depth_0 = []
    for segment in cluster_box[0].values():
        segments_at_depth_0.append(f"catheter_{segment.catheter_name_id}")

    constr = Constraint_Config(
        constraint_type="num_catheters",
        variable_type="catheter",
        maximum=cluster_box.num_physical_catheters,
        variable_name_ids=segments_at_depth_0
    )
    catheter_num_constraint[constr.name_id] = constr

    # # Continuity Constraint
    continuity_constraints = defaultdict(Constraint_Config)
    for cluster in cluster_box:
        if cluster.depth == 0:
            continue
        parents = cluster_box.get_parent_segments(cluster_name_id=cluster.name_id)
        constr = Constraint_Config(
            constraint_type="continuity",
            variable_type="catheter",
            segment_cluster_id=cluster.name_id,
            equal=len(parents),
            variable_name_ids=[f"catheter_{name_id}" for name_id in cluster.catheter_name_ids],
            parent_catheter_name_ids=[f"catheter_{parent.catheter_name_id}" for parent in parents],
        )
        continuity_constraints[f"{constr.constraint_type}_{constr.segment_cluster_id}"] = constr

    # # Collision Constraint
    collision_constraints = defaultdict(Constraint_Config)
    colliding_segments = cluster_box.get_colliding_segments()
    for col_seg in colliding_segments:
        constr = Constraint_Config(
            constraint_type="collision",
            variable_type="catheter",
            equal=1,
            variable_name_ids=[
                f"catheter_{col_seg[0].catheter_name_id}", f"catheter_{col_seg[1].catheter_name_id}"]
        )
        collision_constraints[constr.name_id] = constr

    return {
        "uniqueness": uniqness_constraints,
        "num_catheters": catheter_num_constraint,
        "continuity": continuity_constraints,
        "collision": collision_constraints
    }

class ClusterBoxOptim:
    r"""
    ### Purpose:
    - To find the optimal catheter trajectories, dwell times and penalty weights for an
    initial BrachyPlan.
    """
    def __init__(
        self,
        plan:BrachyPlan,
        optimization_config_list: list[Optimization_Config],
        cluster_box_config: Config_ClusterBox,
        dose_generator: RapidBrachyTG43 | BrachyUtilsTG43,
        ):
        self.plan: BrachyPlan = None
        self.cluster_box: ClusterBox = None
        self.dose_generator: RapidBrachyTG43 | BrachyUtilsTG43 = None
        self.catheter_table_optimizer: CatheterTableOptim_Gurobi = None

        self.structure_names_list = [config.structure_name for config in optimization_config_list]
        self.target_structure_names = [
            config.structure_name for config in optimization_config_list
            if config.is_target
            ]
        self.plan = self.validate_plan_initialization(plan=plan)
        self.cluster_box = self.get_cluster_box_from_plan(
            plan= self.plan,
            structure_names_list= self.structure_names_list,
            target_structure_names= self.target_structure_names,
            cluster_box_config= cluster_box_config)

    def validate_plan_initialization(self, plan:BrachyPlan) -> BrachyPlan:
        if plan.phantom is None:
            raise ValueError("The plan.phantom is None. Please load a phantom into the plan before proceeding.")
        if plan.catheter_table is not None:
            raise ValueError("The plan.catheter_table is not None. Please set it to None before proceeding.")
        return plan

    def get_cluster_box_from_plan(
        self,
        plan:BrachyPlan,
        structure_names_list:list[str],
        target_structure_names:list[str],
        cluster_box_config: Config_ClusterBox
        ) -> ClusterBox:
        r"""
        ### Purpose:
        - To generate a cluster box from a BrachyPlan object. The cluster box will be used to generate
        the geometric constraints for catheter trajectory optimization. It also provides a catheter table.
        
        ### Inputs:
        - plan: BrachyPlan := The initial brachytherapy plan without a catheter table
        - structure_names_list: list[str] := The list of structure names to be used for
        generating the cluster box. These structures will be used to generate the OARs and target structures.
        - target_structure_names: list[str] := The list of target structure names to be used for generating the cluster box. These structures will be used to generate the target structures.
        - cluster_box_config: Config_ClusterBox := The configuration for the cluster box generation.
        
        ### Outputs:
        - cluster_box: ClusterBox := The generated cluster box object.
        """
        
        mesh_dict = plan.phantom.get_structure_mask(
            query_structure_names=structure_names_list,
            mask_type="mesh",
            strict_name_match=False,
        )
        cluster_box = ClusterBox(
            structure_dict=mesh_dict,
            target_structure_names=target_structure_names,
            num_physical_catheters=cluster_box_config.num_physical_catheters, 
            rotation_angle_deg=cluster_box_config.rotation_angle_deg, 
            insertion_point_spacing_mm=cluster_box_config.insertion_point_spacing_mm, 
            num_decision_planes=cluster_box_config.num_decision_planes, 
            config_angle=cluster_box_config.config_angle,
            oar_collision_margin_mm=cluster_box_config.oar_collision_margin_mm, 
            segment_collision_margin_mm=cluster_box_config.segment_collision_margin_mm, 
            box_margin_mm=cluster_box_config.box_margin_mm, 
            cluster_dict=cluster_box_config.cluster_dict,
            )
        return cluster_box