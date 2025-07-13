import json
from pathlib import Path
from typing import Union, Literal
import pydicom
from collections import defaultdict
from pydantic import BaseModel, model_validator
class BrachySource(BaseModel):
    r"""
    ### Purpose:
    - A class to hold the information of a brachytherapy source.
    ### Attributes:
    - treatment_type: str
    - source_geometry: str
    - core_material: str
    - mass_number: int
    - atomic_number: int
    - air_kerma_per_history: float
    - reference_air_kerma_rate: float
    - source_dict: dict | Path | str: either a dictionary containing the source information, or a path to a json or a dicom plan file.
    ### Functions:
    - validate(): checks if the fields are valid for export.
    - to_dict(): converts the object to a dictionary.
    """

    treatment_type: Literal["HDR", "PLDR", "TLDR"] = "HDR"
    source_geometry: str = "MicroSelectronV2"
    core_material: str = "G4_Ir"
    mass_number: int = 192
    atomic_number: int = 77
    air_kerma_per_history: float = 1.149000e-11
    reference_air_kerma_rate: float = None  # 4.278729e04,
    # source_dict: Union[dict, Path, str] = None

    def __init__(self, pth_source:Path|str=None, **data):
        r"""
        ### Purpose:
        - Initialize the BrachySource object.
        - If pth_source is provided, it will be processed to extract the source information.
        """
        if pth_source is not None:
            pth_source = Path(pth_source)
            if not pth_source.exists():
                raise ValueError(f"Path {self.source_dict} does not exist.")
            if pth_source.suffix == ".json":
                with open(self.source_dict, "r") as f:
                    source_data = json.load(f)
            elif pth_source.suffix == ".dcm":
                source_data = self.load_from_dicom(pth_dicom=pth_source)
            else:
                raise ValueError(
                    f"File {self.source_dict} is not a json nor a dicom file."
                )
            super().__init__(**source_data)
        else:
            super().__init__(**data)

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
            "reference_air_kerma_rate": self.reference_air_kerma_rate,
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
            + f"/parallel_world/AKS {self.reference_air_kerma_rate}\n"
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
            
    @classmethod
    def load_from_dicom(cls, pth_dicom: Union[str, Path]) -> dict:
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
        # FIXME make a constant dictonary of the sources that RapidBrachy works with and 
        # pick that source depending on the source geometry. 
        # Fill in reference air kerma from dicom though.
        source_dict = defaultdict(str)
        source_dict["treatment_type"] = plan_dcm.get("BrachyTreatmentType", "HDR")
        source_dict["source_geometry"] = plan_dcm.TreatmentMachineSequence[0].ManufacturerModelName
        if "microselectron-hdr v2" in source_dict["source_geometry"].lower():
            source_dict["source_geometry"] = "MicroSelectronV2"
        # source_dict["source_geometry"] = plan_dcm.get("SourceModelName", "MicroSelectronV2")
        source_dict["core_material"] = plan_dcm.SourceSequence[0].SourceIsotopeName
        if source_dict["core_material"] == "Ir-192":
            source_dict["core_material"] = "G4_Ir"
            source_dict["mass_number"] = 192
            source_dict["atomic_number"] = 77
            source_dict["air_kerma_per_history"] = 1.149000e-11
        source_dict["reference_air_kerma_rate"] = plan_dcm.SourceSequence[0].ReferenceAirKermaRate
        return source_dict
class BrachySimulation(BaseModel):
    r"""
    ### Purpose:
    - A class to hold the information of a brachytherapy simulation. The
    simulations are done using the RapidBrachyMC software.
    ### Attributes:
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
    ### Functions:
    - validate(): checks if the fields are valid for export.
    - to_string(): converts the object to a string.
    """
    brachy_source: BrachySource = BrachySource()
    world_material: str = "Air"
    number_histories: int = 1e6
    total_time: float = None
    dose_format: str = "nrrd"
    number_of_threads: int = 12
    control_verbose: int = 0
    run_verbose: int = 0
    tracking_verbose: int = 0
    print_progress: int = 1e4
    pth_plan: str = "combined.plan"
    pth_phantom: str = "ct.egsphant"
    simulation_dict: Union[dict, Path, str] = None
    
    @model_validator(mode="before")
    def finish_initialization(cls, all_inputs):
        r"""
        ### Purpose:
        Handle initialization from either simulation_dict or direct parameters.
        Process brachy_source in a consistent way.
        """
        def process_simulation_dict(sim_dict):
            if isinstance(sim_dict, (Path, str)):
                if not Path(sim_dict).exists():
                    raise ValueError(f"Path {sim_dict} does not exist.")
                if Path(sim_dict).suffix != ".json":
                    raise ValueError(f"File {sim_dict} is not a json file.")
                with open(sim_dict, "r") as f:
                    return json.load(f)
            elif isinstance(sim_dict, dict):
                return sim_dict
            raise ValueError(
                f"simulation_dict should be either a dictionary or a path to a json file. Got {sim_dict}"
            )

        def create_brachy_source(source_input):
            if source_input is None:
                return BrachySource()
            if isinstance(source_input, (dict, Path, str)):
                return BrachySource(source_dict=source_input)
            if isinstance(source_input, BrachySource):
                return source_input
            raise ValueError(
                f"brachy_source should be either a dictionary, a path to a json file, or a BrachySource object. Got {source_input}"
            )

        if all_inputs.get("simulation_dict") is not None:
            sim_dict = process_simulation_dict(all_inputs["simulation_dict"])
            sim_dict["brachy_source"] = create_brachy_source(sim_dict.get("brachy_source"))
            return sim_dict
        else:
            all_inputs["brachy_source"] = create_brachy_source(all_inputs.get("brachy_source"))
            return all_inputs

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
        # self.validate()
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
            + f"/parallel_world/AKS {self.brachy_source.reference_air_kerma_rate}\n"
            + f"/parallel_world/total_time {self.total_time}\n"
            + f"/dose/format {self.dose_format}\n"
            + f"/run/numberOfThreads {self.number_of_threads}\n"
            + "/run/initialize\n"
            + f"/control/verbose {self.control_verbose}\n"
            + f"/run/verbose {self.run_verbose}\n"
            + f"/tracking/verbose {self.tracking_verbose}\n"
            + f"/run/printProgress {int(self.print_progress)}\n"
            + f"/sim/beamOn {int(self.number_histories)}"
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