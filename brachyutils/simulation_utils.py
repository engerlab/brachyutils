class BrachySimulation:
    r"""
    Purpose:
        - This class holds the information needed for simulating a brachytherapy plan using the
        RapidBrachyMC software.
    """

    def __init__(self, simulation_dict: dict = None):
        self.treatment_type: str = "HDR" #HDR or LDR
        self.source_geometry: str = "MicroSelectronV2"
        self.core_material: str = "G4_Ir"
        self.mass_number: int = 192
        self.atomic_number: int = 77
        self.pth_plan: str = None
        self.pth_phantom: str = None
        self.air_kerma_per_history: float = 1.149000e-11
        self.reference_air_kerma: float = 4.278729e04
        self.number_histories: int = int(1e7)
        self.total_time: float = None
        self.dose_format: str = "nrrd"
        self.number_of_threads: int = 16
        self.control_verbose: int = 0
        self.run_verbose: int = 0
        self.tracking_verbose: int = 0
        self.print_progress: int = self.number_histories / 100

        if simulation_dict is not None:
            self.treatment_type = (
                simulation_dict["treatment_type"]
                if "treatment_type" in simulation_dict
                else self.treatment_type
            )
            self.source_geometry = (
                simulation_dict["source_geometry"]
                if "source_geometry" in simulation_dict
                else self.source_geometry
            )
            self.core_material = (
                simulation_dict["core_material"]
                if "core_material" in simulation_dict
                else self.core_material
            )
            self.mass_number = (
                simulation_dict["mass_number"]
                if "mass_number" in simulation_dict
                else self.mass_number
            )
            self.atomic_number = (
                simulation_dict["atomic_number"]
                if "atomic_number" in simulation_dict
                else self.atomic_number
            )
            self.pth_plan = (
                simulation_dict["pth_plan"]
                if "pth_plan" in simulation_dict
                else self.pth_plan
            )
            self.pth_phantom = (
                simulation_dict["pth_phantom"]
                if "pth_phantom" in simulation_dict
                else self.pth_phantom
            )
            self.air_kerma_per_history = (
                simulation_dict["air_kerma_per_history"]
                if "air_kerma_per_history" in simulation_dict
                else self.air_kerma_per_history
            )
            self.reference_air_kerma = (
                simulation_dict["reference_air_kerma"]
                if "reference_air_kerma" in simulation_dict
                else self.reference_air_kerma
            )
            self.number_histories = (
                simulation_dict["number_histories"]
                if "number_histories" in simulation_dict
                else self.number_histories
            )
            self.total_time = (
                simulation_dict["total_time"]
                if "total_time" in simulation_dict
                else self.total_time
            )
            self.dose_format = (
                simulation_dict["dose_format"]
                if "dose_format" in simulation_dict
                else self.dose_format
            )
            self.number_of_threads = (
                simulation_dict["number_of_threads"]
                if "number_of_threads" in simulation_dict
                else self.number_of_threads
            )
            self.control_verbose = (
                simulation_dict["control_verbose"]
                if "control_verbose" in simulation_dict
                else self.control_verbose
            )
            self.run_verbose = (
                simulation_dict["run_verbose"]
                if "run_verbose" in simulation_dict
                else self.run_verbose
            )
            self.tracking_verbose = (
                simulation_dict["tracking_verbose"]
                if "tracking_verbose" in simulation_dict
                else self.tracking_verbose
            )
            self.print_progress = (
                simulation_dict["print_progress"]
                if "print_progress" in simulation_dict
                else self.print_progress
            )

    def validate(self, verbose = False):
        r"""
        Purpose:
            - to validate the simulation object.
        Returns:
            - True if the fields are valid for export, False otherwise.
        """
        required_types = {
            self.treatment_type: str,
            self.source_geometry: str,
            self.core_material: str,
            self.mass_number: int,
            self.atomic_number: int,
            self.pth_plan: str,
            self.pth_phantom: str,
            self.air_kerma_per_history: float,
            self.reference_air_kerma: float,
            self.number_histories: int,
            self.total_time: float,
            self.dose_format: str,
            self.number_of_threads: int,
            self.control_verbose: int,
            self.run_verbose: int,
            self.tracking_verbose: int,
            self.print_progress: int
            }
        for key, value in required_types.items():
            if not isinstance(key, value):
                if verbose:
                    print(f"BrachySimulation: field {key} is not of type {value}")
                return False


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
        Input:
            - self: BrachySimulation
        Output:
            - a string containing the information of the simulation.
        Dependencies:
            - None
        """
        self.validate()
        return f"""/source/treatmentType {self.treatment_type}
            /source/switch {self.source_geometry}
            /source/coreMaterial {self.core_material}
            /source/core/A {self.mass_number}
            /source/core/Z {self.atomic_number}
            /sim/plan {self.pth_plan}
            /world/phantom {self.pth_phantom}
            /parallel_world/ak_per_history {self.air_kerma_per_history}
            /parallel_world/ref_ak {self.reference_air_kerma}
            /parallel_world/total_time {self.total_time}
            /dose/format {self.dose_format}
            /run/numberOfThreads {self.number_of_threads}
            /run/initialize
            /control/verbose {self.control_verbose}
            /run/verbose {self.run_verbose}
            /tracking/verbose {self.tracking_verbose}
            /run/printProgress {self.PrintProgress}
            /sim/beamOn {self.number_histories}
            """
