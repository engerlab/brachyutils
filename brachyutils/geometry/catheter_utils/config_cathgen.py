from pydantic import BaseModel, Field
from numpy.typing import ArrayLike
from typing import List, Tuple, Dict

class Config_Catheter_Rotation(BaseModel):
    """
    ### Purpose:
    - Configuration for angled catheter generation. This will be used in angled_catheter_pairs().

    ### Attributes:
    - `x_angle_max`  : half-wdith angle away from normal rotating around x axis (degrees);
    sweeps -x_angle_max to +x_angle_max. 0 means parallel to normal, 90 means perpendicular to normal.
    - `x_angle_step` : x angle increment (degrees)
    - `y_angle_max`   : half-width angle away from normal rotating around y axis (degrees);
    sweeps -y_angle_max to +y_angle_max.
    - `y_angle_step`  : ap angle increment (degrees)
    """
    x_angle_max: float = Field(default=10, description="  half-wdith angle away from normal rotating around x axis (degrees);\
sweeps -x_angle_max to +x_angle_max. 0 means parallel to normal, 90 means perpendicular to normal.")
    x_angle_step: float = Field(default=10, description="right-left angle increment (degrees)")
    y_angle_max: float = Field(default=10, description="half-width angle away from normal rotating around y axis (degrees); \
sweeps -y_angle_max to +y_angle_max.")
    y_angle_step: float = Field(default=10, description="anterior-posterior (ap) angle increment (degrees)")

class Decision_Plane(BaseModel):
    r"""
    ### Purpose:
    - An object to store decision plane attributes. This will be returned by obb_planes()
    
    ### Attributes:
    - `depth`: int := The order of this decision plane. starts from zero.
    - `origin`: ArrayLike:= The center of the decision plane in world coordinates. len = 3.
    - `normal`: ArrayLike:= The normal of the decision plane in world coordinates. len = 3.
    - `transform`: ArrayLike:= The transform from matrix between world axis to the plane's local axis.
    size 4 x 4
    - `extents`: ArrayLike:= The full length of the sides of the plane.
    """
    model_config={"arbitrary_types_allowed":True}
    depth: int = Field(..., description="The order of this decision plane. starts from zero.")
    origin: ArrayLike = Field(..., description="The center of the decision plane in \
world coordinates. len = 3.")
    normal: ArrayLike = Field(..., description="The normal of the decision plane in world coordinates. len = 3.")
    transform: ArrayLike = Field(..., description="The transform from matrix between world axis to the plane's local axis.\
size 4 x 4")
    extents: ArrayLike = Field(..., description="The full length of the sides of the plane.")
    segment_lines: List[Tuple] = Field(default=None, description="The list of the segments lines that departure \
from this plane.")

class Config_ClusterBox(BaseModel):
    r"""
    ### Purpose:
    - Configuration for the cluster box generation.
    
    ### Attributes:
    - `num_physical_catheters`: int := the number of physical catheters to be inserted.
    - `rotation_angle_deg`: float := the rotation angle of the catheter box around the right left (X) axis (degrees).
    - `insertion_point_spacing_mm`: float := the spacing between adjacent catheter insertion points on the bottom plane (mm).
    - `num_decision_planes`: int := the number of decision planes to be defined in the catheter box.
    - `config_angle`: Dict[str, Config_Catheter_Rotation] | Config_Catheter_Rotation | None := The angle configuartion for each insertion point. If a single Config_Catheter_Rotation is provided, it will be applied to all insertion points. If None, the default Config_Catheter_Rotation() will be applied to all insertion points.
    - `oar_collision_margin_mm`: float := the collision margin between catheter segments and organs at risk (OARs) (mm).
    - `segment_collision_margin_mm`: float := the collision margin between catheter segments (mm). Measured as center of the catheter segments.
    - `box_margin_mm`: float := The margin between the box boundaries and the OARs
    """
    num_physical_catheters: int = Field(default=1, description="the number of physical catheters to be inserted.")
    rotation_angle_deg: float = Field(default=0, description="the rotation angle of the catheter box \
around the right left (X) axis (degrees).")
    insertion_point_spacing_mm: float = Field(default=10, description="the spacing between adjacent \
catheter insertion points on the bottom plane (mm).")
    num_decision_planes: int = Field(default=2, description="the number of decision planes to be \
defined in the catheter box.")
    config_angle: Dict[str, Config_Catheter_Rotation] | Config_Catheter_Rotation | None = Field(
        default=None,
        description="The angle configuartion for each insertion point. \
If a single Config_Catheter_Rotation is provided, it will be applied to all \
insertion points. If None, the default Config_Catheter_Rotation() will be applied \
to all insertion points.")
    oar_collision_margin_mm: float = Field(default=5, description="the collision margin between \
catheter segments and organs at risk (OARs) (mm).")
    segment_collision_margin_mm: float = Field(default=5, description="the collision margin between \
catheter segments (mm). Measured as center of the catheter segments.")
    box_margin_mm: float = Field(default=0, description="The margin between the box boundaries and the OARs")
