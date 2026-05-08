from pydantic import BaseModel, Field

class Config_Angled_CathGen(BaseModel):
    """
    ### Purpose:
    - Configuration for angled catheter generation. This will be used in angled_catheter_pairs().

    ### Attributes:    
    - alt_max  : maximum altitude angle away from normal (degrees). 0 means parallel to normal,
    90 means perpendicular to normal.
    - alt_step : altitude angle increment (degrees)
    - az_max   : half-width of azimuthal sweep (degrees); sweeps -az_max to +az_max. 0 means
    no sweep and the catheter is pointing towards the center of the bottom plane.
    - az_step  : azimuthal angle increment (degrees)
    """
    alt_max: float = Field(default=10, description="maximum altitude angle away from normal (degrees). \
0 means parallel to normal, 90 means perpendicular to normal.")
    alt_step: float = Field(default=5, description="altitude angle increment (degrees)")
    az_max: float = Field(default=10, description="half-width of azimuthal sweep (degrees); \
sweeps -az_max to +az_max. 0 means no sweep and the catheter is pointing towards \
the center of the bottom plane.")
    az_step: float = Field(default=5, description="azimuthal angle increment (degrees)")
