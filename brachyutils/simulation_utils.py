class BrachySource:
    r"""
    Purpose:
        - This class holds the information needed for simulating a brachytherapy source using the
        RapidBrachyMC software.
    """

    def __init__(
        self,
        treatment_type:str = "HDR",
        source_geometry: str = "MicroSelectronV2",
        core_material: str = "G4_Ir",
        mass_number: int = 192,
        atomic_number: int = 77,
        air_kerma_per_history: float = 1.149000e-11,
        reference_air_kerma: float = 4.278729e04,
        source_dict: dict = None,
        ) -> None:
        r"""
        Purpose:
            - A class to hold the information of a brachytherapy source.
        Inputs:
            - treatment_type: str
            - source_geometry: str
            - core_material: str
            - mass_number: int
            - atomic_number: int
            - air_kerma_per_history: float
            - reference_air_kerma: float
            - source_dict: dict
        Attributes:
            - treatment_type: str
            - source_geometry: str
            - core_material: str
            - mass_number: int
            - atomic_number: int
            - air_kerma_per_history: float
            - reference_air_kerma: float
        Functions:
            - validate(): checks if the fields are valid for export.
            - to_dict(): converts the object to a dictionary.
        """
        
        assert (
            (treatment_type is not None)
            and (source_geometry is not None)
            and (core_material is not None)
            and (mass_number is not None)
            and (atomic_number is not None)
            and (air_kerma_per_history is not None)
            and (reference_air_kerma is not None)
        ) != (
            source_dict is not None
        ), "Either provide treatment_type, source_geometry, core_material, mass_number,\
            atomic_number, air_kerma_per_history, reference_air_kerma or provide source_dict. Not both."
        
        if source_dict is not None:
            treatment_type = source_dict.get("treatment_type", None)
            source_geometry = source_dict.get("source_geometry", None)
            core_material = source_dict.get("core_material", None)
            mass_number = source_dict.get("mass_number", None)
            atomic_number = source_dict.get("atomic_number", None)
            air_kerma_per_history = source_dict.get("air_kerma_per_history", None)
            reference_air_kerma = source_dict.get("reference_air_kerma", None)
        
        self.treatment_type: str = treatment_type 
        self.source_geometry: str = source_geometry
        self.core_material: str = core_material
        self.mass_number: int = mass_number 
        self.atomic_number: int = atomic_number 
        self.air_kerma_per_history: float = air_kerma_per_history 
        self.reference_air_kerma: float = reference_air_kerma 
        
        self.validate()

    def validate(self, verbose = False):
        r"""
        Purpose:
            - to validate the source object.
        Returns:
            - True if the fields are valid for export, False otherwise.
        """
        required_types = {
            self.treatment_type: str,
            self.source_geometry: str,
            self.core_material: str,
            self.mass_number: int,
            self.atomic_number: int,
            self.air_kerma_per_history: float,
            self.reference_air_kerma: float
            }
        for key, value in required_types.items():
            if not isinstance(key, value):
                try:
                    key = value(key)
                    continue
                except ValueError:
                    pass
                if verbose:
                    print(f"BrachySource: field {key} is not of type {value}")
                return False
        return True
    
    def to_dict(self):
        r"""
        Purpose:
            - to convert the object to a dictionary.
        Input:
            - self: BrachySource
        Output:
            - a dictionary containing the information of the source.
        Dependencies:
            - None
        """
        return {
            "treatment_type": self.treatment_type,
            "source_geometry": self.source_geometry,
            "core_material": self.core_material,
            "mass_number": self.mass_number,
            "atomic_number": self.atomic_number,
            "air_kerma_per_history": self.air_kerma_per_history,
            "reference_air_kerma": self.reference_air_kerma
        }

    def to_string(self):
        r"""
        Purpose:
            - to convert the object to a string.
        Input:
            - self: BrachySource
        Output:
            - a string containing the information of the source with the proper macro commands.
        Dependencies:
            - None
        """
        return (
            f"/treatmentType {self.treatment_type}\n" +
            f"/source/switch {self.source_geometry}\n" +
            f"/source/coreMaterial {self.core_material}\n" +
            f"/source/core/A {self.mass_number}\n" +
            f"/source/core/Z {self.atomic_number}\n"
            f"/parallel_world/ak_per_history {self.air_kerma_per_history}\n" +
            f"/parallel_world/ref_ak {self.reference_air_kerma}\n"
            )

class BrachySimulation:
    r"""
    Purpose:
        - This class holds the information needed for simulating a brachytherapy plan using the
        RapidBrachyMC software.
    """

    def __init__(
        self,
        brachy_source: BrachySource,
        world_material: str = "Air",
        number_histories: int = 2e9,
        total_time: float = None,
        dose_format: str = "nrrd",
        number_of_threads: int = 16,
        control_verbose: int = 0,
        run_verbose: int = 0,
        tracking_verbose: int = 0,
        print_progress: int = 1e6,
        simulation_dict: dict = None
        ) -> None:
        r"""
        Purpose:
            - A class to hold the information of a brachytherapy simulation.
        Inputs:
            - brachy_source: BrachySource
            - world_material: str
            - number_histories: int
            - total_time: float
            - dose_format: str
            - number_of_threads: int
            - control_verbose: int
            - run_verbose: int
            - tracking_verbose: int
            - print_progress: int
            - simulation_dict: dict
        Attributes:
            - brachy_source: BrachySource
            - world_material: str
            - number_histories: int
            - total_time: float
            - dose_format: str
            - number_of_threads: int
            - control_verbose: int
            - run_verbose: int
            - tracking_verbose: int
            - print_progress: int
        Functions:
            - validate(): checks if the fields are valid for export.
            - to_string(): converts the object to a string.
        """
        
        assert (
            (brachy_source is not None)
            and (world_material is not None)
            and (number_histories is not None)
            and (total_time is not None)
            and (dose_format is not None)
            and (number_of_threads is not None)
            and (control_verbose is not None)
            and (run_verbose is not None)
            and (tracking_verbose is not None)
            and (print_progress is not None)
        ) != (
            simulation_dict is not None
        ), "Either provide , brachy_source, world_material, number_histories, total_time,\
            dose_format, number_of_threads, control_verbose, run_verbose, tracking_verbose, and\
            print_progress or provide source_dict. Not both."
        
        if simulation_dict is not None:
            brachy_source = BrachySource(simulation_dict.get("source_dict", None))
            world_material = simulation_dict.get("world_material", None)
            number_histories = simulation_dict.get("number_histories", None)
            total_time = simulation_dict.get("total_time", None)
            dose_format = simulation_dict.get("dose_format", None)
            number_of_threads = simulation_dict.get("number_of_threads", None)
            control_verbose = simulation_dict.get("control_verbose", None)
            run_verbose = simulation_dict.get("run_verbose", None)
            tracking_verbose = simulation_dict.get("tracking_verbose", None)
            print_progress = simulation_dict.get("print_progress", None)

        self.brachy_source: BrachySource = brachy_source
        self.world_material: str = world_material
        self.number_histories: int = number_histories
        self.total_time: float = total_time
        self.dose_format: str = dose_format
        self.number_of_threads: int = number_of_threads
        self.control_verbose: int = control_verbose
        self.run_verbose: int = run_verbose
        self.tracking_verbose: int = tracking_verbose
        self.print_progress: int = print_progress

        self.validate()

    def validate(self, verbose = False):
        r"""
        Purpose:
            - to validate the simulation object.
        Returns:
            - True if the fields are valid for export, False otherwise.
        """
        required_types = {
            self.world_material: str,
            self.pth_plan: str,
            self.pth_phantom: str,
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
                try:
                    key = value(key)
                    continue
                except ValueError:
                    pass
                if verbose:
                    print(f"BrachySimulation: field {key} is not of type {value}")
                return False
        return True

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
        return (
            f"/treatmentType {self.brachy_source.treatment_type}\n" +
            f"/source/switch {self.brachy_source.source_geometry}\n" +
            f"/source/coreMaterial {self.brachy_source.core_material}\n" +
            f"/source/core/A {self.brachy_source.mass_number}\n" +
            f"/source/core/Z {self.brachy_source.atomic_number}\n" +
            f"/sim/plan {self.pth_plan}\n" +
            f"/world/phantom {self.pth_phantom}\n" +
            f"/world/material {self.world_material}\n" +
            f"/parallel_world/ak_per_history {self.brachy_source.air_kerma_per_history}\n" +
            f"/parallel_world/ref_ak {self.brachy_source.reference_air_kerma}\n" +
            f"/parallel_world/total_time {self.total_time}\n" +
            f"/dose/format {self.dose_format}\n" +
            f"/run/numberOfThreads {self.number_of_threads}\n" +
            f"/run/initialize\n" +
            f"/control/verbose {self.control_verbose}\n" +
            f"/run/verbose {self.run_verbose}\n" +
            f"/tracking/verbose {self.tracking_verbose}\n" +
            f"/run/printProgress {self.print_progress}\n" +
            f"/sim/beamOn {self.number_histories}" 
            )

    def to_dict(self):
        r"""
        Purpose:
            - to convert the object to a dictionary.
        Input:
            - self: BrachySimulation
        Output:
            - a dictionary containing the information of the simulation.
        Dependencies:
            - None
        """
        return {
            "source_dict": self.brachy_source.to_dict(),
            "world_material": self.world_material,
            "number_histories": self.number_histories,
            "total_time": self.total_time,
            "dose_format": self.dose_format,
            "number_of_threads": self.number_of_threads,
            "control_verbose": self.control_verbose,
            "run_verbose": self.run_verbose,
            "tracking_verbose": self.tracking_verbose,
            "print_progress": self.print_progress
        }
