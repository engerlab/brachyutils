import json
from pathlib import Path
from typing import Union, Literal, List
import pydicom
from collections import defaultdict
from pydantic import BaseModel, ConfigDict, model_validator, Field
from brachyutils.geometry.applicator_utils import BrachyApplicator
from brachyutils.planning.source_utils import BrachySource

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
    ### Functions:
    - validate(): checks if the fields are valid for export.
    - to_string(): converts the object to a string.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    brachy_source: BrachySource | Path | str = BrachySource()
    world_material: str = "Air"
    number_histories: int = int(1e7)
    total_time: float = None
    dose_format: str = "nrrd"
    number_of_threads: int = 8
    control_verbose: int = 0
    run_verbose: int = 0
    tracking_verbose: int = 0
    print_progress: int = int(1e5)
    pth_plan: str = "combined.plan"
    pth_phantom: str = "phantom.seq.nrrd"
    pth_body_stl: str = None
    body_material: str = "Water"
    applicator_list : List[BrachyApplicator] = []

    # Exclude from dumps, default to None
    pth_simulation_setup: Path | str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def process_setup_and_source(self) -> "BrachySimulation":
        # 1. Load data from JSON if the path is provided
        if self.pth_simulation_setup is not None:
            pth = Path(self.pth_simulation_setup)
            if not pth.exists():
                raise ValueError(f"Path {pth} does not exist.")
            
            if pth.suffix == ".json":
                with open(pth, "r") as f:
                    data = json.load(f)
                
                # Dynamically set attributes from the JSON
                for key, value in data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            else:
                raise ValueError(f"File {pth} is not a json file.")

        # 2. Coerce brachy_source into a BrachySource object
        if isinstance(self.brachy_source, (Path, str)):
            # Uses the pth_source field from our refactored BrachySource class
            self.brachy_source = BrachySource(pth_source=self.brachy_source)
        elif isinstance(self.brachy_source, dict):
            self.brachy_source = BrachySource(**self.brachy_source)
        elif not isinstance(self.brachy_source, BrachySource):
            raise ValueError(
                f"brachy_source should be a dictionary, a path, or a BrachySource object. Got {type(self.brachy_source)}"
            )

        return self

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
        outstring = (
            f"/source/treatmentType {self.brachy_source.treatment_type}\n"
            + f"/source/switch {self.brachy_source.source_geometry}\n"
            + f"/source/core/material {self.brachy_source.core_material}\n"
            + f"/source/core/A {self.brachy_source.mass_number}\n"
            + f"/source/core/Z {self.brachy_source.atomic_number}\n"
            + f"/sim/plan {self.pth_plan}\n"
            + f"/world/phantom {self.pth_phantom}\n"
        )
        if self.pth_body_stl is not None:
            outstring = (
                outstring
                + f"/world/body_mask {self.pth_body_stl}\n"
                + f"/world/body_mask_material {self.body_material}\n"
            )

        if self.applicator_list:
            for applicator in self.applicator_list:
                outstring = (
                    outstring
                    + f"/applicator/path {applicator.name}.stl\n"
                    + f"/applicator/material {applicator.material}\n"
                    + f"/applicator/density {applicator.density}\n"
                    + f"/applicator/done\n"
                )
        outstring = (
            outstring
            + f"/parallel_world/AKS {self.brachy_source.air_kerma_strength}\n"
            + f"/parallel_world/ak_per_history {self.brachy_source.air_kerma_per_history_strength}\n"
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
        
        return outstring

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
        return (self.model_dump())

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