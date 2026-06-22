from pydantic import BaseModel, Field

class Config_Angled_CathGen(BaseModel):
    """
    ### Purpose:
    - Configuration for angled catheter generation. This will be used in angled_catheter_pairs().

    ### Attributes:    
    - rl_max  : maximum angle away from normal rotating on the right-left (rl) axis (x) (degrees); sweeps from -rl_max to +rl_max.
    -rl_max to +rl_max. 0 means parallel to normal, 90 means perpendicular to normal.
    - rl_step : rl angle increment (degrees)
    - ap_max   : half-width of anterior-posterior (ap) sweep (degrees); sweeps -ap_max to +ap_max.
    - ap_step  : ap angle increment (degrees)
    """
    rl_max: float = Field(default=10, description="maximum right-left angle away from normal (degrees); \
        sweeps from -rl_max to +rl_max. 0 means parallel to normal, 90 means perpendicular to normal.")
    rl_step: float = Field(default=5, description="right-left angle increment (degrees)")
    ap_max: float = Field(default=10, description="half-width of anterior-posterior (ap) sweep (degrees); \
sweeps -ap_max to +ap_max. 0 means no sweep and the catheter is pointing towards \
the center of the bottom plane.")
    ap_step: float = Field(default=5, description="anterior-posterior (ap) angle increment (degrees)")
