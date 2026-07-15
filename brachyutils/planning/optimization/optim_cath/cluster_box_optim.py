from copy import deepcopy
from typing import Dict, Literal
from collections import defaultdict
from pathlib import Path
from brachyutils.geometry.catheter_utils.catheter_cluster_box import ClusterBox
from brachyutils.geometry.catheter_utils.dwell_position import DwellPosition
from brachyutils.planning.optimization.optim_configs import Constraint_Config
from brachyutils.planning.plan_utils import BrachyPlan
from brachyutils.dose.dose_generation_utils import RapidBrachyTG43 
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import CatheterTableOptim_Gurobi 
from brachyutils.geometry.catheter_utils.config_cathgen import Config_ClusterBox

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
        config_cluster_box: Config_ClusterBox,
        ):
        self.plan: BrachyPlan = None
        self.cluster_box: ClusterBox = None
        self.optimization_object: CatheterTableOptim_Gurobi = None
        self.geometric_constraint_dict = None

        self.plan = self.validate_plan_initialization(plan=plan)
        self.original_phantom = deepcopy(self.plan.phantom)

        self.cluster_box = self.get_cluster_box_from_plan(
            plan= self.plan,
            config_cluster_box= config_cluster_box)
        self.generate_dose_rate_for_cluster(
            plan=self.plan,
            cluster_box=self.cluster_box,)
        self.optimization_object = self.build_optimization_object(plan=self.plan)
        self.set_geometric_constraints(
            cluster_box=self.cluster_box,
            optim_obj=self.optimization_object)

    def validate_plan_initialization(self, plan:BrachyPlan) -> BrachyPlan:
        if plan.phantom is None:
            raise ValueError("The plan.phantom is None. Please load a phantom into the plan before proceeding.")
        if plan.catheter_table is not None:
            raise ValueError("The plan.catheter_table is not None. Please set it to None before proceeding.")
        if plan.prescription_dose is None:
            raise ValueError("The plan.prescription_dose is None. Please set it to a value before proceeding.")
        if plan.structure_list is None or len(plan.structure_list) == 0:
            raise ValueError("The plan.structure_list is None or empty. Please load structures into the plan before proceeding.")
        if plan.optimization_config_dict is None or len(plan.optimization_config_dict) == 0:
            raise ValueError("The plan.optimization_config_dict is None or empty. Please load optimization configurations into the plan before proceeding.")

        return plan

    def get_cluster_box_from_plan(
        self,
        plan:BrachyPlan,
        config_cluster_box: Config_ClusterBox
        ) -> ClusterBox:
        r"""
        ### Purpose:
        - To generate a cluster box from a BrachyPlan object. The cluster box will be used to generate
        the geometric constraints for catheter trajectory optimization. It also provides a catheter table,
        and the plan phantom will be cropped to the bounding box of the cluster box.

        ### Inputs:
        - plan: BrachyPlan := The initial brachytherapy plan without a catheter table
        - config_cluster_box: Config_ClusterBox := The configuration for the cluster box generation.
        
        ### Outputs:
        - cluster_box: ClusterBox := The generated cluster box object.
        """
        structure_names_list = list(plan.optimization_config_dict.keys())
        target_structure_names = plan.target_structure_names
        mesh_dict = plan.phantom.get_structure_mask(
            query_structure_list=structure_names_list,
            mask_type="mesh",
            strict_name_match=False,
        )

        # # crop and resample the phantom to the bounding box of the target structures.
        plan.phantom.crop_by_contour(
            contour_name=structure_names_list,
            strict_name_match=False,
            marginInMM=config_cluster_box.box_margin_mm
        )
        plan.phantom.resample_to(
            spacing=plan.optimization_config_dict.get(target_structure_names[0]).spacing_mm,
        )

        cluster_box = ClusterBox(
            structure_dict=mesh_dict,
            target_structure_names=target_structure_names,
            num_physical_catheters=config_cluster_box.num_physical_catheters, 
            rotation_angle_deg=config_cluster_box.rotation_angle_deg, 
            insertion_point_spacing_mm=config_cluster_box.insertion_point_spacing_mm, 
            num_decision_planes=config_cluster_box.num_decision_planes, 
            config_angle=config_cluster_box.config_angle,
            oar_collision_margin_mm=config_cluster_box.oar_collision_margin_mm, 
            segment_collision_margin_mm=config_cluster_box.segment_collision_margin_mm, 
            box_margin_mm=config_cluster_box.box_margin_mm, 
            )
        return cluster_box

    def generate_dose_rate_for_cluster(
        self,
        plan: BrachyPlan,
        cluster_box: ClusterBox,
        ) -> BrachyPlan:
        r"""
        ### Purpose:
        - To generate the dose rate maps for the dwell positions in catheter segments in the cluster box.
        - By default we use BrachyUtilsTG43 for dose generation. Note that the phantom in the plan has already
        been cropped to the bounding box of the cluster box, so the dose generation will be faster.
        - The catheter table of the plan will be updated with the dose rates.

        ### Inputs:
        - plan: BrachyPlan := The initial brachytherapy plan without a catheter table
        - cluster_box: ClusterBox := The cluster box containing the catheter segments for which dose rates
        will be generated. 
        
        ### Outputs:
        - None:= The catheter table in the plan will be updated with the new dose rates.
        """
        from time import time
        plan.set_catheter_table(
            catheter_table=cluster_box.catheter_table,
            dwells_near_ptv=True,
            )
        t0 = time()
        dose_generator = RapidBrachyTG43(
            dir_plan_export="temp_data/tg43/cluster_box"+plan.phantom.pth_image.name)
        dose_generator.run_dose_generation(
            plan=plan,
            generate_dose_rate_maps=True,)
        t1=time()
        print("Time for RapidBRachyTG43 was: ", t1-t0)

    def build_optimization_object(
        self,
        plan: BrachyPlan
        ):
        r"""
        ### Purpose:
        - To build the initial optimization model.
        """
        print("Building optimization model for the cluster")
        optim_obj = CatheterTableOptim_Gurobi(
            plan=plan,
            multi_processing=True
        )
        return optim_obj

    def set_geometric_constraints(
        self,
        cluster_box:ClusterBox,
        optim_obj:CatheterTableOptim_Gurobi,
        ):
        r"""
        ### Purpose:
        - To set the geometric catheter constraints to the model.
        These constraints are generated using get_geometric_constraints() 
        """
        
        self.geometric_constraint_dict = get_geometric_constraints(cluster_box=cluster_box)
        for constraint_type, constraint_dict in self.geometric_constraint_dict.items():
            print(f"setting {constraint_type} constraints")
            optim_obj.set_constraints(constraint_config_dict=constraint_dict)

    def get_optimized_plan_from_model(
        self,
        solve_strategy:Literal["simultaneous", "cascaded"] = "simultaneous",
        dwell_reach_mm: float = None
        ) -> BrachyPlan:
        r"""
        ### Puropse:
        - To solve the optimization and get the final catheter table.
        The solve could have two strategy, simultaneous catheter-dwell-time optimization
        or cascaded catheter-dwell-time optimization

        ### Inputs:
        - `solve_strategy` := If simultaenous, both the catheter variables and the dwell times are
        optmized simultaneiously. If cascaded, the catheter variables are first optimized with respect
        to the normalized combined dose rate map with dwell times constrained to 1 second, then the
        catheter variables are bound to their optimal solution and dwell times are optimized.
        - `dwell_reach_mm` := The voxel goal for the combined dose rate is set to the dose rate value at
        this distance on the transverse plane cutting through the center of the source normalized by the 
        target combined dose.
        ### Outputs:
        `outplan`: BrachyPlan := The plan with the optimized catheter positions and dwell times.
        """
        if solve_strategy == "simultaneous":
            if dwell_reach_mm is not None:
                raise ValueError("`dwell_reach_mm` is only applicable to cascaded solving")
            # probably should assert that dwell times are not bound to 1 and the 
            # voxel dose goal is not normalized.
            outplan = self.optimization_object.get_optimized_plan_from_model()
        if solve_strategy == "cascaded":
            if dwell_reach_mm is None:
                raise ValueError("please provide the `dwell_responsibility_mm` value.")
            voxel_goal_combined_dose = self.plan.structure_dict.get(
                self.plan.target_structure_names[0]).optimization_config.dose_voxel_goal 
            dose_voxel_goal = get_voxel_dose_rate_goal(
                    dwell_reach_mm = dwell_reach_mm,
                    dose_voxel_goal = dose_voxel_goal,
                    dwell_position = self.plan.catheter_table.all_dwells[0],
                )
        return outplan

    def export_to(
        self,
        out_dir: str | Path,
        dose_normalization_constant:float = 1):
        r"""
        ### Purpose:
        - Writes the contents of self to a directory. The contents are the cluster box that is 
        written to many .ply files, as well as the combined dose rate map normalized to a user
        specified value.

        ### Inputs:
        - out_dir:= directory where self will be exported.
        - dose_normalization_constant:= the combined dose rate map is normalized by this value.
        """
        out_dir = Path(out_dir)
        self.cluster_box.to_ply(
            out_ply_dir=out_dir)
        saved_dwelltimes = [deepcopy(dwell.time) for dwell in self.plan.catheter_table.all_dwells]
        # # export the normalized combined dose rate
        self.plan.catheter_table.reset_dwelltimes_to(reset_value=1)
        combined_dose_rate = self.plan.combined_dose
        combined_dose_rate.dose_image.imageArray = (
            combined_dose_rate.dose_image.imageArray / dose_normalization_constant)
        combined_dose_rate.write_brachydose_to_file(
            pth_dose_file=out_dir/"combined_dose_rate.seq.nrrd")
        for dwell, time in zip(
            self.plan.catheter_table.all_dwells,
            saved_dwelltimes):
            dwell.time = time        
        self.cluster_box.catheter_table.write_to_slicer_markup(
            pth_mrk_json=out_dir/"catheter_table.mrk.json")

def run_catheter_recommendation():
    r"""
    ### Purpose:
    - To recommend catheters based on two strategies: bulk or bulk-then-sequential
    This could be a stand alone method controlling the ClusterBoxOptim.
    """
    pass

def get_voxel_dose_rate_goal(
    dwell_reach_mm:float,
    dose_voxel_goal:float,
    dwell_position:DwellPosition,):
    r"""
    ### Purpose:
    - To obtain the voxel goal for the combined dose rate map based on the dwell reach and
    the voxel dose goal.

    ### Inputs:
    - `dwell_reach_mm` := 
    - `dose_voxel_goal` := 
    - `dwell_position` := 
    """
    # TODO Priority 1: complete this!
    pass