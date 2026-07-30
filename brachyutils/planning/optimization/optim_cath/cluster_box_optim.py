from copy import deepcopy
from typing import Dict, Literal, Annotated, List
from collections import defaultdict
from pathlib import Path
from brachyutils.geometry.catheter_utils.catheter_cluster_box import ClusterBox
from brachyutils.geometry.catheter_utils import CatheterTable, Catheter
from brachyutils.planning.optimization.optim_configs import Constraint_Config
from brachyutils.planning.plan_utils import BrachyPlan
from brachyutils.dose.dose_generation_utils import RapidBrachyTG43 
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import CatheterTableOptim_Gurobi 
from brachyutils.geometry.catheter_utils.config_cathgen import Config_ClusterBox
import random
import numpy as np
import pandas as pd
from time import time
import json

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

    4 Future:
    - Consider bounding a cluster to 1 to enforce using a specific insertion point with
    undetermined insertion angle.
    
    ### Inputs:
    - cluster_box: ClusterBox := This object contains all the candidate catheter segments, organized
    by their cluster where they stem from.
    
    ### Outputs:
    - all_constraints : Dict[str, dict] := A dictionary containing the many geometric constraints above.
    The items of this dictionary are: "uniqueness", "num_catheters", "continuity", "collision"
    The values are  Dict[constraint.name, constraint]
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
            variable_name_ids=[f"{name_id}" for name_id in cluster.catheter_name_ids]
        )
        uniqness_constraints[constr.name_id] = constr

    # # Catheter Num Constraints
    catheter_num_constraint = {}
    segments_at_depth_0 = []
    for segment in cluster_box[0].values():
        segments_at_depth_0.append(f"{segment.catheter_name_id}")

    if isinstance(cluster_box.num_physical_catheters, int):
        eq_num_catheters = cluster_box.num_physical_catheters
        min_num_catheters = None
        max_num_catheters = None
    elif isinstance(cluster_box.num_physical_catheters, list):
        eq_num_catheters = None
        min_num_catheters = cluster_box.num_physical_catheters[0]
        max_num_catheters = cluster_box.num_physical_catheters[1]
        
    constr = Constraint_Config(
        constraint_type="num_catheters",
        variable_type="catheter",
        equal=eq_num_catheters,
        maximum=max_num_catheters,
        minimum=min_num_catheters,
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
            variable_name_ids=cluster.catheter_name_ids,
            parent_catheter_name_ids=[parent.catheter_name_id for parent in parents],
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
                f"{col_seg[0].catheter_name_id}", f"{col_seg[1].catheter_name_id}"]
        )
        collision_constraints[constr.name_id] = constr

    return {
        "uniqueness": uniqness_constraints,
        "num_catheters": catheter_num_constraint,
        "continuity": continuity_constraints,
        "collision": collision_constraints
    }

def get_angle_constraints(
    cluster_box: ClusterBox,
    x_angle: float = None,
    y_angle: float = None,
    ):
    r"""
    ### Purpose:
    - To generate constraints that tells the model to chose catheters with a user-defined
    angle only. For example, among all the catheters in a cluster only pick the ones that
    are straight (x_angle = 0, y_angle = 0) or not. All the catheters that do not have the
    desired angles are bound to zero.

    ### Inputs:
    - cluster_box := The catheter cluster box object containing all the angle information.
    - x_angle := Describes the rotation anlge arround the y axis.
    - y_angle := Describes the rotation anlge arround the x axis.

    ### Output:
    - `angle_constraint_dict` := 
    """
    angle_constraint_dict = defaultdict(Constraint_Config)
    segments_to_keep = []
    for segment in cluster_box.all_segments_list:
        pick_this_segment = False
        if ((x_angle is not None)
            and y_angle is not None):
            if (np.isclose(segment.x_angle, x_angle, atol=0.01)
                and np.isclose(segment.y_angle, y_angle, atol=0.01)):
                pick_this_segment = True
        else:
            if x_angle is not None:
                if np.isclose(segment.x_angle, x_angle, atol=0.01):
                    pick_this_segment = True
            if y_angle is not None:
                if np.isclose(segment.y_angle, y_angle, atol=0.01):
                    pick_this_segment = True
        if pick_this_segment:
            segments_to_keep.append(segment.catheter_name_id)

    for segment in cluster_box.all_segments_list:
        if segment.catheter_name_id not in segments_to_keep:
            bind_angle = Constraint_Config(
                constraint_type="bound",
                variable_type="catheter",
                variable_name_ids=[segment.catheter_name_id],
                equal=0,)
            angle_constraint_dict[bind_angle.name_id] = bind_angle
    return angle_constraint_dict

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
        r"""
        ### Purpose:
        - To initialize a cluster box optimization object.
        
        ### Inputs:
        - plan := The brachyplan object with loaded `phantom`, `prescription_dose`,
        `structure_list`, and `optimization_config_dict`. Note that the `catheter_table`
        of this plan should be None.
        - config_cluster_box := The config object for the cluster box generation. see
        Config_ClusterBox class for details.
        """
        
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
        if plan.dvh_metric_goals is None or len(plan.dvh_metric_goals) == 0:
            raise ValueError("The plan dvh metric goals is empty. please load the dvh metric goals.")
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
            config_catheter_rotation=config_cluster_box.config_catheter_rotation,
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
            dwells_near_ptv=False, # set to true for reducing number of dwells
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
        constraint_dict: Dict[str, Constraint_Config] = None,
        ):
        r"""
        ### Purpose:
        - To set the geometric catheter constraints to the model.
        These constraints are generated using get_geometric_constraints() 
        """
        if self.geometric_constraint_dict is None:
            self.geometric_constraint_dict = get_geometric_constraints(cluster_box=cluster_box)            
            for constraint_type, geo_constraint_dict in self.geometric_constraint_dict.items():
                print(f"setting {constraint_type} constraints")
                optim_obj.set_constraints(constraint_config_dict=geo_constraint_dict)

        if constraint_dict is not None:
            optim_obj.set_constraints(constraint_config_dict=constraint_dict)
            for name, constraint in constraint_dict.items():
                print(f"added constraint to cluster box optim {name}")
                self.geometric_constraint_dict[constraint.constraint_type][constraint.name_id] = constraint

    def get_optimized_plan_from_model(
        self,
        solve_strategy:Literal["simultaneous", "cascaded"] = "simultaneous",
        dwell_responsibility_radius_mm: float = None
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
        - `dwell_responsibility_radius_mm` := The target combined dose rate is set to the dose rate value at
        this distance on the transverse plane cutting through the center of the source normalized by the 
        target combined dose.  
        ### Outputs:
        `outplan`: BrachyPlan := The plan with the optimized catheter positions and dwell times.
        """
        if solve_strategy == "simultaneous":
            if dwell_responsibility_radius_mm is not None:
                raise ValueError("`dwell_responsibility_radius_mm` is only applicable to cascaded solving")
            # probably should assert that dwell times are not bound to 1 and the 
            # voxel dose goal is not normalized.
            outplan = self.optimization_object.get_optimized_plan_from_model()
        if solve_strategy == "cascaded":
            if dwell_responsibility_radius_mm is None:
                raise ValueError("please provide the `dwell_responsibility_mm` value.")
            raise NotImplementedError("Yeah this hasn't worked so far. The binary \
problem is not solvable. if you figure it out, big ups!")
            # target_dose_rate_goal = get_target_dose_rate_goal(
            #     dwell_responsibility_radius_mm,
            #     voxel_dose_goal = self.plan.stru,
            #     dwell_position = self.plan.catheter_table.all_dwells[0],
            # )
        return outplan

    def update_num_physical_catheters(
        self,
        num_physical_catheters:int|List[int]):
        r"""
        ### Purpose:
        - To update the number of physical catheters for the cluster box and the
        related constraint.
        """
        self.cluster_box.num_physical_catheters = num_physical_catheters
        if isinstance(num_physical_catheters, int):
            self.geometric_constraint_dict.get(
                "num_catheters").get("num_catheters").equal = self.cluster_box.num_physical_catheters
        else:
            self.geometric_constraint_dict.get(
                "num_catheters").get("num_catheters").minimum = self.cluster_box.num_physical_catheters[0]
            self.geometric_constraint_dict.get(
                "num_catheters").get("num_catheters").maximum = self.cluster_box.num_physical_catheters[1]

        self.set_geometric_constraints(
            cluster_box=self.cluster_box,
            optim_obj=self.optimization_object,
            constraint_dict=self.geometric_constraint_dict["num_catheters"]
            )

    def get_physical_catheter_table(
        self,
        catheter_table: CatheterTable,) -> CatheterTable:
        r"""
        ### Purpose:
        - To join the optimized catheter segments into phaysical catheters.
        """
        # # Find all the catheters that are on the same chain
        new_catheter_table = defaultdict(Catheter)
        indx_physical_catheter = 0
        for cluster in self.cluster_box.cluster_dict.values():
            if cluster.depth != 0:
                continue
            # # gatheter all the catheter segments recursively
            caths_on_chain = self.cluster_box.get_catheters_on_chain(
                cluster.insert_position,
                catheter_table)
            if len(caths_on_chain) == 0:
                continue
            # # Join them in a single chain (dwell positions and digitization)
            # # make sure catheters are ordered from tip to base
            caths_on_chain.reverse()
            new_dwells = []
            indx_dwell = 0
            digitization_points = []
            for cath in caths_on_chain:
                for dwell in cath.dwells:
                    dwell.index = indx_dwell
                    new_dwells.append(dwell)
                    indx_dwell +=1
                digitization_points.extend(cath.digitization_points)
            physical_catheter = Catheter(
                index=indx_physical_catheter,
                dwells=new_dwells,
                digitization_points=digitization_points,)
            new_catheter_table[physical_catheter.name_id] = physical_catheter
            indx_physical_catheter +=1

        physical_catheter_table = CatheterTable(
            catheters_dict=new_catheter_table
        )
        return physical_catheter_table

    def export_to(
        self,
        out_dir: str | Path,
        export_combined_dose_rate:bool = False,
        dose_normalization_constant:float = 1):
        r"""
        ### Purpose:
        - Writes the contents of self to a directory. The contents are the cluster box that is 
        written to many .ply files, as well as the combined dose rate map normalized to a user
        specified value.

        ### Inputs:
        - out_dir:= directory where self will be exported.
        - export_combined_dose_rate := If ture, it'll write out the combined dose rate. 
        - dose_normalization_constant := the combined dose rate map is normalized by this value.
        """
        out_dir = Path(out_dir)
        self.cluster_box.to_ply(
            out_ply_dir=out_dir)
        self.cluster_box.catheter_table.write_to_json(
            pth_json=out_dir/"candidate_catheter_table.json")
        self.cluster_box.catheter_table.write_to_slicer_markup(
            pth_mrk_json=out_dir/"candidate_catheter_table.mrk.json")

        physical_catheter_table = self.get_physical_catheter_table(self.plan.catheter_table)
        physical_catheter_table.write_to_json(
            pth_json=out_dir/"catheter_table.json")
        physical_catheter_table.write_to_slicer_markup(
            pth_mrk_json=out_dir/"catheter_table.mrk.json")

        self.plan.catheter_table.combined_dose.write_brachydose_to_file(
            pth_dose_file=out_dir/"combined.seq.nrrd"
        )
        self.original_phantom.export_to(dir_nrrd_out=out_dir)

        if export_combined_dose_rate:
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

def run_experiment_sequential(
    cbox_optim:ClusterBoxOptim,
    max_num_physical_catheters: int,
    step_num_physical_catheters: int,
    initial_num_physical_catheters: int,
    prob_catheter_deviation: Annotated[float, "0.0 to 1.0"] = 0,
    prepandicular_catheters:bool = False,
    multi_objective_optimizer: None = None,
    list_hyper_parameters: List[str] = None,
    dir_output: str | Path = None,
    ):
    r"""
    ### Purpose:
    - To run experiments showing the effectiveness and robustness of the sequential
    catheter recommendation approach. The steps are as follows:
    1. Start with `initial_num_physical_catheters`.
    2. Run cluster box optimization.
    3. Get the optimal catheter trajectories (c*).
    4. Disturb each catheter in c* with acording to `prob_catheter_deviation`.
    5. Run `multi_objective_optimization` and get acceptance rate.
    6. Generate bounding constraint dictionary c=c*.
    7. Update the number of physical catheters by adding `step_num_physical_catheters`.
    8. If the new number of physical catheters <= max_num_physical_catheters,
        repeat Step 2.

    ### Inputs:
    - `cbox_optim` := A cluster box optimization objec that is already initialized. 
    - `max_num_physical_catheters`:= The maximum number of physical catheters to be inseretd.
    - `step_num_physical_catheters`:= At each iteration, we add a few more catheters to see
    their impact on acceptance rate.
    - `initial_num_physical_catheters`:= The initial optimization with have a bulk number
    of cathetes. 
    - `prob_catheter_deviation`:= After each optimization step, we can disturb
    the unconstrained catheters with a certain probability to assess the robustness 
    of the pipeline to catheter deviation.
    - `prepandicular_catheters` := If true, an additional constraint is added per segment
    cluster that tells the model to pick only the straight catheters or not. Note that
    during robustness analysis, the disturbance may deviate from prepandicular path.
    - `multi_objective_optimizer`:= The multi objective optimization class that'll
    give us an acceptance rate, the observed dvh metrics and their penalty weights
    as well as the timing data. 
    - `dir_output`:= directory where the outputs will be written to
 
    ### Outputs:
    The following infomration written to `dir_output`:
    - `out_df`:= A dataframe with containing the info of each number of catheters and
    multiple coloumns with the following information written to dir_output/"results.csv"
        - `num_physical_catheters` per iteration
        - `acceptance_rate` per iteration
        - optimal_hyper_params from MOO per iteration.
        - observed_dvh_metrics from BrachyPlan per iteration.
    - `cbox_optim` post optimization is exported to 
    `dir_output`/f"cbox_{num_physical_catheters}_catheters". For the export content see 
    `ClusterBoxOptim.export_to()`
    - `constraint_dict` per iteration
    - `expriment_info`:
        - max_num_physical_catheters
        - step_num_physical_catheters
        - initial_num_physical_catheters
        - prob_catheter_deviation
        - multi_objective_optimizer
        - range_hyper_parameters
    """
    # TODO: Deep debug this shit!
    experiment_info = {
            "max_num_physical_catheters": max_num_physical_catheters,
            "step_num_physical_catheters": step_num_physical_catheters,
            "initial_num_physical_catheters": initial_num_physical_catheters,
            "prob_catheter_deviation":prob_catheter_deviation,
        # + ["multi_objective_optimizer", range_hyper_parameters] TODO figure out after MOO
    }
    with open(dir_output/"experiment_info.json", "w") as out_file:
        json.dump(experiment_info, out_file, indent=4)   

    all_dvh_metric_names = []
    for structure in cbox_optim.plan.dvh_metric_goals:
        all_dvh_metric_names = all_dvh_metric_names + (
            cbox_optim.plan.dvh_metric_goals.get(
                structure).get("dvh_metric_names")
        )

    out_df = pd.DataFrame(columns=(
        ["num_physical_catheters","acceptance_rate",
         "time_optim_catheters", "time_optim_moo"]
        + all_dvh_metric_names)
        # + all_hyper_parameter_names TO Be Added Next Week!
        )

    c_equal_1_constraints = defaultdict(Constraint_Config)
    for num_phys_catheters in range(
        initial_num_physical_catheters,
        max_num_physical_catheters+step_num_physical_catheters,
        step_num_physical_catheters):

        # # First set the new number of physical catheters 
        cbox_optim.update_num_physical_catheters(
            num_physical_catheters=num_phys_catheters)

        # # add straight catheter constraint if needed
        if prepandicular_catheters:
            prepandicular_constraints = get_angle_constraints(
                cluster_box=cbox_optim.cluster_box,
                x_angle=0,
                y_angle=0,)
            # If a catheter has been previously bound to 1 (already inserted)
            # remove its angle constraint
            donot_disturbe_constrs = []
            for angle_constr_name in prepandicular_constraints:
                c_equal_1 = c_equal_1_constraints.get(angle_constr_name)
                if c_equal_1 is not None:
                    donot_disturbe_constrs.append(angle_constr_name)
            for constr_name in donot_disturbe_constrs:
                prepandicular_constraints.pop(constr_name)

            cbox_optim.optimization_object.set_constraints(prepandicular_constraints)
        # # Get the optimal catheters (c*)
        t0_cath = time()
        optimized_plan = cbox_optim.get_optimized_plan_from_model() 
        t1_cath = time()
        if prepandicular_catheters:
            # # Free the model for future constraints
            cbox_optim.optimization_object.remove_constraints(prepandicular_constraints)

        # # Now disturbe the catheters
        donot_disturbe_catheters = []
        for c_equal_1 in c_equal_1_constraints.values():
            donot_disturbe_catheters.extend(c_equal_1.variable_name_ids)
        disturbed_catheter_table = disturbe_catheter_table(
            catheter_table=optimized_plan.catheter_table,
            prob_catheter_deviation=prob_catheter_deviation,
            cluster_box=cbox_optim.cluster_box,
            donot_disturbe_catheters=donot_disturbe_catheters,
            )

        # # get ready for MOO by binding the model to c*
        # # c* has two parts those that =1 and =0.
        c_equal_0_constraints = defaultdict(Constraint_Config)
        for catheter in disturbed_catheter_table:
            if catheter.channel_total_time == 0:
                bind_to_0 = Constraint_Config(
                    constraint_type="bound",
                    variable_type="catheter",
                    equal=0,
                    variable_name_ids=[catheter.name_id],
                )
                c_equal_0_constraints[
                    bind_to_0.name_id] = bind_to_0
            else:
                bind_to_1 = Constraint_Config(
                    constraint_type="bound",
                    variable_type="catheter",
                    equal=1,
                    variable_name_ids=[catheter.name_id],
                )
                c_equal_1_constraints[
                    bind_to_1.name_id] = bind_to_1

        # # Make sure there are no colliding pairs in the disturbed
        # # catheter table! resolve by setting one to zero randomly.
        collision_pairs = [
            constr.variable_name_ids for constr in
            cbox_optim.geometric_constraint_dict["collision"].values()]
        c_eq_1_catheters = []
        for constr in c_equal_1_constraints.values():
            c_eq_1_catheters.extend(constr.variable_name_ids)
        conflict_found = []            
        for col_pair in collision_pairs:
            if (col_pair[0] in c_eq_1_catheters
                and col_pair[1] in c_eq_1_catheters):
                conflict_found.append(col_pair)
                set_to_zero = random.choice(col_pair)
                c_equal_1_constraints.pop(f"bound_catheter_{set_to_zero}_eq")
                bind_to_0 = Constraint_Config(
                    constraint_type="bound",
                    variable_type="catheter",
                    equal=0,
                    variable_name_ids=[set_to_zero],
                )
                c_equal_0_constraints[
                    bind_to_0.name_id] = bind_to_0
        # # free the catheter number to avoid conflicts with
        # # collision free scrambled catheter table binding
        cbox_optim.optimization_object.remove_constraints(
            cbox_optim.geometric_constraint_dict["num_catheters"]
        )

        # # now add the new bounds to the optimization object
        cbox_optim.optimization_object.set_constraints(
            constraint_config_dict=c_equal_1_constraints
        )
        cbox_optim.optimization_object.set_constraints(
            constraint_config_dict=c_equal_0_constraints
        )

        # # Pass it to MOO and get acceptance rate as well as
        # other metrics The final solution is the one with
        # lowest dose to urethra.
        # TODO: to be implemented!
        t0_moo = time()
        cbox_optim.get_optimized_plan_from_model()
        t1_moo = time()

        # # Free the catheters that were bound 0 for next iteration!
        cbox_optim.optimization_object.remove_constraints(
            constraint_config_dict=c_equal_0_constraints
        )
        physical_catheters_used = _count_physical_catheters_used(
            catheter_table=disturbed_catheter_table,
            cluster_box=cbox_optim.cluster_box
        )

        out_df.loc[len(out_df)] = {
        "num_physical_catheters": physical_catheters_used,
        "acceptance_rate": 0,
        "time_optim_catheters": t1_cath - t0_cath,
        "time_optim_moo": t1_moo - t0_moo,
        } | cbox_optim.plan.get_dvh_metrics()
        if dir_output is not None:
            out_df.to_csv(dir_output/f"results.csv")
            print(f"Wrote sequential trail for {physical_catheters_used} Catheters")
            cbox_optim.export_to(out_dir=dir_output/f"cbox_{physical_catheters_used}_catheters")

def disturbe_catheter_table(
    catheter_table: CatheterTable,
    prob_catheter_deviation: float,
    cluster_box: ClusterBox,
    donot_disturbe_catheters:List[str],
    ):
    r"""
    ### Purpose:
    - To disturbe the trajectory of a physical catheter.
    The probability of a physical catheter not deviating (1 - `prob_catheter_deviation`)
    is equal to the probability of a segment not deviating to the power of number of
    segments.
    $$
    (1 - `prob_catheter_deviation`) = (1-p)^n
    $$
    where   p:= probability of segment deviation.
            n:= number of segments on a chain.
    So in this case, we solve for p. Note that disturbance at a lower depth has
    much larger consequences than disturbance at higher depths. If a segment at higher
    depth is disturbed, all its children are set to zero and random children for 
    neighboring segment are chosen. Accordingly, p often being much
    smaller than `prob_catheter_deviation` reflects this depth dependent impact.
    
    This method recursively travels down the cluster children to activate, deactivate
    or disturbe them using catheter.digitization_points[1]
    """
    if prob_catheter_deviation == 0:
        return catheter_table
    # # calculate p, the probability of each segment.
    n_segs_on_chain = cluster_box.num_decision_planes-1
    p = 1 - (1 - prob_catheter_deviation)**(1/n_segs_on_chain) 

    for cluster in cluster_box.cluster_dict.values():
        if cluster.depth > 0:
            continue
        _disturbe_this_cluster(
            prob_segment_disturbance=p,
            insert_position=cluster.insert_position,
            catheter_table=catheter_table,
            cluster_box=cluster_box,
            donot_disturbe_catheters=donot_disturbe_catheters,
        )
    return catheter_table

def _disturbe_this_cluster(
    prob_segment_disturbance:float,
    insert_position: np.typing.ArrayLike,
    catheter_table: CatheterTable,
    cluster_box:ClusterBox,
    donot_disturbe_catheters:List[str],
    ):
    r"""
    ### Purpose:
    - Given an insert point, get the cluster, get the catheter that is active
    in this cluster, turn that catheter of, pick a random neighbour and activate it
    as well as the cluster of that neighbour.
    
    - exit condition: if none of the catheters in this cluster are active
    or the cluster has no children
    """
    cluster = cluster_box.get_cluster_by_insert_position(insert_position)
    if cluster is None:
        return
    catheter_2_turn_off = None
    for catheter in catheter_table.get_catheters_by_ids(
        cluster.catheter_name_ids):
        if catheter.name_id in donot_disturbe_catheters:
            continue
        if catheter.channel_total_time == 0:
            continue
        else:
            yes_disturb = random.random() < prob_segment_disturbance
            if yes_disturb:
                catheter_2_turn_off = catheter.name_id
                catheter.reset_dwelltimes_to(0.0)
                _deactivate_this_cluster(
                    catheter.digitization_points[1],
                    catheter_table,
                    cluster_box)
            else:
                # run the same process on the child cluster (recussion alert)
                _disturbe_this_cluster(
                    prob_segment_disturbance,
                    catheter.digitization_points[1],
                    catheter_table,
                    cluster_box,
                    donot_disturbe_catheters,
                )
    if catheter_2_turn_off is None:
        return
    # now pick a random neighbour that is not that catheter
    # and activate it!
    copied_list = deepcopy(cluster.catheter_name_ids)
    copied_list.remove(catheter_2_turn_off)
    if len(copied_list) == 0:
        # if there are no other candidates available, just keep this one.
        random_neighbour_id = catheter_2_turn_off
    else:
        random_neighbour_id = random.choice(copied_list)
    catheter = catheter_table.get_catheters_by_ids([random_neighbour_id]).pop()
    catheter.reset_dwelltimes_to(1)
    _activate_this_cluster(
        catheter.digitization_points[1],
        catheter_table,
        cluster_box)

def _deactivate_this_cluster(
    insert_position: np.typing.ArrayLike,
    catheter_table: CatheterTable,
    cluster_box:ClusterBox
    ):
    r"""
    ### Purpose:
    - Given an insert point, get the cluster that stems from it
    find the catheter that is active in that cluster, note its
    tip position. reset_dwell_times() for all the catheters in that
    cluster. recursively call this function on the cluster that stems
    from the tip position.
    
    - exit condition: if cluster box did not return a cluster based on
    that insert point, return!
    
    ### Inputs:
    - insert_position := 3D coordinates of the insertion point
    - catheter_table := 
    - cluster_box :=
    """
    cluster = cluster_box.get_cluster_by_insert_position(insert_position)
    if cluster is None:
        return
    for catheter in catheter_table.get_catheters_by_ids(
        cluster.catheter_name_ids):
        next_insertion_point = catheter.digitization_points[1]
        if catheter.channel_total_time != 0:
            catheter.reset_dwelltimes_to(0.0)
    _deactivate_this_cluster(
        insert_position=next_insertion_point,
        catheter_table=catheter_table,
        cluster_box=cluster_box
    )

def _activate_this_cluster(
    insert_position: np.typing.ArrayLike,
    catheter_table: CatheterTable,
    cluster_box:ClusterBox
    ):
    r"""
    ### Purpose:
    - Given an insertion point, get the cluster with this insertion
    point from the box. randomly select one of its catheter and set
    the dwell times of that catheter to 1. Then recursively pass the 
    tip of that catheter to the same function for activation.
    
    - exit condition: if cluster box did not return a cluster based on
    that insert point, return!
    """
    cluster = cluster_box.get_cluster_by_insert_position(insert_position)
    if cluster is None:
        return
    num_dwells = 0
    num_sampling = 0
    while num_dwells == 0 and num_sampling <= len(cluster.catheter_name_ids):
        random_catheter_name_id = random.choice(
            cluster.catheter_name_ids
        )
        catheter = catheter_table.get_catheters_by_ids([random_catheter_name_id])[0]
        num_dwells = len(catheter.dwells)
        num_sampling += 1

    catheter.reset_dwelltimes_to(1.0)
    _activate_this_cluster(catheter.digitization_points[1], catheter_table, cluster_box)

def _count_physical_catheters_used(
    catheter_table:CatheterTable,
    cluster_box: ClusterBox,) -> int:
    r"""
    Counts the number of physical catheters used in the table based
    on the clusters with depth = 0 that had an active catheter
    """
    catheter_count = 0
    for cluster in cluster_box.cluster_dict.values():
        if cluster.depth > 0:
            continue
        catheters = catheter_table.get_catheters_by_ids(cluster.catheter_name_ids)
        for catheter in catheters:
            if catheter.channel_total_time > 0:
                catheter_count += 1
    return catheter_count
