from pydantic import BaseModel, Field

class Config_Angled_CathGen(BaseModel):
    """
    ### Purpose:
    - Configuration for angled catheter generation. This will be used in angled_catheter_pairs().

    ### Attributes:    
    - x_angle_max  : half-wdith angle away from normal rotating around x axis (degrees);
    sweeps -x_angle_max to +x_angle_max. 0 means parallel to normal, 90 means perpendicular to normal.
    - x_angle_step : x angle increment (degrees)
    - y_angle_max   : half-width angle away from normal rotating around y axis (degrees);
    sweeps -y_angle_max to +y_angle_max.
    - y_angle_step  : ap angle increment (degrees)
    """
    x_angle_max: float = Field(default=4, description="  half-wdith angle away from normal rotating around x axis (degrees);\
sweeps -x_angle_max to +x_angle_max. 0 means parallel to normal, 90 means perpendicular to normal.")
    x_angle_step: float = Field(default=2, description="right-left angle increment (degrees)")
    y_angle_max: float = Field(default=4, description="half-width angle away from normal rotating around y axis (degrees); \
sweeps -y_angle_max to +y_angle_max.")
    y_angle_step: float = Field(default=2, description="anterior-posterior (ap) angle increment (degrees)")
