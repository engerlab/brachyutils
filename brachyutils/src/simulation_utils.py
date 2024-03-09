class BrachySimulation:
    r"""
    Purpose:
        - This class holds the information needed for simulating a brachytherapy plan using the 
        RapidBrachyMC software.
    """
    def __init__(self):
        self.treatment_type: str = "HDR"
        self.source_geometry: str = "MicroSelectronV2"
        self.core_material: str = "G4_Ir"
        self.mass_number: int = 192
        self.atomic_number: int = 77
        self.pth_plan: str = None
        self.pth_phantom: str = None
        self.air_kerma_per_history: float = 1.149000e-11
        self.reference_air_kerma: float = 4.278729e04
        self.number_histories: int = None
        self.total_time: float = None
        self.dose_format: str = "3ddose"
        self.number_of_threads: int = None
        self.control_verbose: int = 0
        self.run_verbose: int = 0
        self.tracking_verbose: int = 0
        self.PrintProgress: int = None
        self.beam_on: int = None

    def run_simulation(self):
        r"""
        Purpose:
            - to use RapidBrachyMC to simulate the brachytherapy plan.
        """
        raise NotImplementedError

    def to_string(self):
        r"""
        Purpose:
            - to convert the object to a string.
        """
        raise NotImplementedError
