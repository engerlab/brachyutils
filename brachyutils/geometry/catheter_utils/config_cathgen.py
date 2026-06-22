from pydantic import BaseModel, Field

class Config_Angled_CathGen(BaseModel):
    """
    ### Purpose:
    - Configuration for angled catheter generation. This will be used in angled_catheter_pairs().

    ### Attributes:    
    - x_angle_max  : maximum angle away from normal rotating on the right-left (rl) axis (x) (degrees);
    sweeps -x_angle_max to +x_angle_max. 0 means parallel to normal, 90 means perpendicular to normal.
    - x_angle_step : rl angle increment (degrees)
    - y_angle_max   : half-width of anterior-posterior (ap) sweep (degrees); sweeps -y_angle_max to +y_angle_max.
    - y_angle_step  : ap angle increment (degrees)
    """
    x_angle_max: float = Field(default=4, description="maximum right-left angle away from normal (degrees); \
        sweeps from -x_angle_max to +x_angle_max. 0 means parallel to normal, 90 means perpendicular to normal.")
    x_angle_step: float = Field(default=2, description="right-left angle increment (degrees)")
    y_angle_max: float = Field(default=4, description="half-width of anterior-posterior (ap) sweep (degrees); \
sweeps -y_angle_max to +y_angle_max. 0 means no sweep and the catheter is pointing towards \
the center of the bottom plane.")
    y_angle_step: float = Field(default=2, description="anterior-posterior (ap) angle increment (degrees)")
