import json
from pathlib import Path
from typing import Union
import pydicom
from collections import defaultdict

class BrachySource:
    def __init__(
        self,
        treatment_type: str = "HDR",
        source_geometry: str = "MicroSelectronV2",
        core_material: str = "G4_Ir",
        mass_number: int = 192,
        atomic_number: int = 77,
        air_kerma_per_history: float = 1.149000e-11,
        reference_air_kerma: float = None,  # 4.278729e04,
        source_dict: Union[dict, Path, str] = None,
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
            - source_dict: dict | Path | str: either a dictionary containing the source information, or a path to a json file.
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
            if isinstance(source_dict, (Path, str)):
                if not Path(source_dict).exists():
                    raise ValueError(f"Path {source_dict} does not exist.")
                if Path(source_dict).suffix == ".json":
                    with open(source_dict, "r") as f:
                        source_dict = json.load(f)
                elif Path(source_dict).suffix == ".dcm":
                    source_dict = self.load_from_dicom(source_dict)
                else:
                    raise ValueError(f"File {source_dict} is not a json nor a dicom file.")
            elif isinstance(source_dict, dict):
                source_dict = source_dict
            else:
                raise ValueError(
                    f"source_dict should be either a dictionary, a path to a json file, or a path to a dicom file. Got {source_dict}"
                )

            treatment_type = source_dict.get("treatment_type", "HDR")
            source_geometry = source_dict.get("source_geometry", "MicroSelectronV2")
            core_material = source_dict.get("core_material", "G4_Ir")
            mass_number = source_dict.get("mass_number", 192)
            atomic_number = source_dict.get("atomic_number", 77)
            air_kerma_per_history = source_dict.get(
                "air_kerma_per_history", 1.149000e-11
            )
            reference_air_kerma = source_dict.get("reference_air_kerma", 4.278729e04)

        self.treatment_type: str = treatment_type
        self.source_geometry: str = source_geometry
        self.core_material: str = core_material
        self.mass_number: int = mass_number
        self.atomic_number: int = atomic_number
        self.air_kerma_per_history: float = air_kerma_per_history
        self.reference_air_kerma: float = reference_air_kerma

        self.validate()

    def validate(self, verbose=False):
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
            self.reference_air_kerma: float,
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
            "reference_air_kerma": self.reference_air_kerma,
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
            f"/treatmentType {self.treatment_type}\n"
            + f"/source/switch {self.source_geometry}\n"
            + f"/source/coreMaterial {self.core_material}\n"
            + f"/source/core/A {self.mass_number}\n"
            + f"/source/core/Z {self.atomic_number}\n"
            + f"/parallel_world/ak_per_history {self.air_kerma_per_history}\n"
            + f"/parallel_world/ref_ak {self.reference_air_kerma}\n"
        )

    def to_json(self, output_path: Union[str, Path]):
        r"""
        Purpose:
            - to convert the object to a json string.
        Input:
            - self: BrachySource
            - output_path: Union[str, Path]
        Output:
            - a json string containing the information of the source.
        Dependencies:
            - json
        """
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f)

    def load_from_dicom(self, pth_dicom: Union[str, Path]) -> dict:
        r"""
        Purpose:
            - to load the simulation object from a dicom directory.
        Input:
            - self: BrachySource
            - pth_dicom: Union[str, Path]
        Output:
            - None
        Dependencies:
            - None
        """
        # Ensure path exists and is directory
        pth_dicom = Path(pth_dicom)
        if not pth_dicom.exists():
            raise FileNotFoundError(f"Path {pth_dicom} does not exist.")
        # Find and load the plan file
        plan_dcm = pydicom.dcmread(str(pth_dicom))

        source_dict = defaultdict(str)
        source_dict["treatment_type"] = plan_dcm.get("BrachyTreatmentType", "HDR")
        source_dict["source_geometry"] = plan_dcm.get("SourceModelName", "MicroSelectronV2")
        source_dict["core_material"] = plan_dcm.get("SourceEncapsulationMaterial", "G4_Ir")
        source_dict["mass_number"] = 192 if source_dict["core_material"] == "G4_Ir" else 0 
        source_dict["atomic_number"] = 77 if source_dict["core_material"] == "G4_Ir" else 0
        source_dict["air_kerma_per_history"] = 1.149000e-11 if source_dict["core_material"] == "G4_Ir" else 0
        source_dict["reference_air_kerma"] = plan_dcm.get("SourceStrength", 4.278729e04)
        return source_dict
class BrachySimulation:
    default_source = BrachySource(reference_air_kerma=4.278729e04)

    def __init__(
        self,
        brachy_source: BrachySource = default_source,
        world_material: str = "Air",
        number_histories: int = 1e6,
        total_time: float = None,
        dose_format: str = "nrrd",
        number_of_threads: int = 12,
        control_verbose: int = 0,
        run_verbose: int = 0,
        tracking_verbose: int = 0,
        print_progress: int = 1e4,
        pth_plan: str = "combined.plan",
        pth_phantom: str = "ct.egsphant",
        simulation_dict: Union[dict, Path, str] = None,
    ) -> None:
        r"""
        Purpose:
            - A class to hold the information of a brachytherapy simulation. The
            simulations are done using the RapidBrachyMC software.
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
            - pth_plan: str
            - pth_phantom: str
            - simulation_dict: dict | Path | str: either a dictionary containing the simulation information, or a path to a json file.
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
            - pth_plan: str
            - pth_phantom: str
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
            and (pth_plan is not None)
            and (pth_phantom is not None)
        ) != (
            simulation_dict is not None
        ), "Either provide , brachy_source, world_material, number_histories, total_time,\
            dose_format, number_of_threads, control_verbose, run_verbose, tracking_verbose,\
            print_progress, pth_plan and pth_phantom or provide source_dict. Not both."

        if simulation_dict is not None:
            if isinstance(simulation_dict, (Path, str)):
                assert Path(
                    simulation_dict
                ).exists(), f"Path {simulation_dict} does not exist."
                assert (
                    Path(simulation_dict).suffix == ".json"
                ), f"Path {simulation_dict} is not a json file."

                with open(simulation_dict, "r") as f:
                    simulation_dict = json.load(f)

            brachy_source = BrachySource(
                source_dict=simulation_dict.get(
                    "source_dict", BrachySimulation.default_source.to_dict()
                )
            )
            world_material = simulation_dict.get("world_material", "Air")
            number_histories = simulation_dict.get("number_histories", 1e6)
            total_time = simulation_dict.get("total_time", None)
            dose_format = simulation_dict.get("dose_format", "nrrd")
            number_of_threads = simulation_dict.get("number_of_threads", 12)
            control_verbose = simulation_dict.get("control_verbose", 0)
            run_verbose = simulation_dict.get("run_verbose", 0)
            tracking_verbose = simulation_dict.get("tracking_verbose", 0)
            print_progress = int(simulation_dict.get("print_progress", 1e4))
            pth_plan = simulation_dict.get("pth_plan", None)
            pth_phantom = simulation_dict.get("pth_phantom", None)

        self.brachy_source: BrachySource = brachy_source if isinstance(
            brachy_source, BrachySource
        ) else BrachySource(source_dict=brachy_source)
        self.world_material: str = world_material
        self.number_histories: int = number_histories
        self.total_time: float = float(total_time)
        self.dose_format: str = dose_format
        self.number_of_threads: int = number_of_threads
        self.control_verbose: int = control_verbose
        self.run_verbose: int = run_verbose
        self.tracking_verbose: int = tracking_verbose
        self.print_progress: int = print_progress
        self.pth_plan: str = pth_plan
        self.pth_phantom: str = pth_phantom

        self.validate()

    def validate(self, verbose=False):
        r"""
        Purpose:
            - to validate the simulation object.
        Returns:
            - True if the fields are valid for export, False otherwise.
        """
        required_types = {
            self.brachy_source: BrachySource,
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
            self.print_progress: int,
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
            f"/source/treatmentType {self.brachy_source.treatment_type}\n"
            + f"/source/switch {self.brachy_source.source_geometry}\n"
            + f"/source/coreMaterial {self.brachy_source.core_material}\n"
            + f"/source/core/A {self.brachy_source.mass_number}\n"
            + f"/source/core/Z {self.brachy_source.atomic_number}\n"
            + f"/sim/plan {self.pth_plan}\n"
            + f"/world/phantom {self.pth_phantom}\n"
            + f"/world/material {self.world_material}\n"
            + f"/parallel_world/ak_per_history {self.brachy_source.air_kerma_per_history}\n"
            + f"/parallel_world/ref_ak {self.brachy_source.reference_air_kerma}\n"
            + f"/parallel_world/total_time {self.total_time}\n"
            + f"/dose/format {self.dose_format}\n"
            + f"/run/numberOfThreads {self.number_of_threads}\n"
            + "/run/initialize\n"
            + f"/control/verbose {self.control_verbose}\n"
            + f"/run/verbose {self.run_verbose}\n"
            + f"/tracking/verbose {self.tracking_verbose}\n"
            + f"/run/printProgress {self.print_progress}\n"
            + f"/sim/beamOn {self.number_histories}"
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
            "print_progress": self.print_progress,
        }

    def to_json(self, output_path: Union[str, Path]):
        r"""
        Purpose:
            - to convert the object to a json string.
        Input:
            - self: BrachySimulation
            - output_path: Union[str, Path]
        Output:
            - a json string containing the information of the simulation.
        Dependencies:
            - json
        """
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f)