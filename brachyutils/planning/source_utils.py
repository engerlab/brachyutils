import json
from pathlib import Path
from typing import Union, Literal, List
import pydicom
from collections import defaultdict
from pydantic import BaseModel, ConfigDict, model_validator, Field
from brachyutils.geometry.applicator_utils import BrachyApplicator

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
    - air_kerma_per_history_strength: float
    - air_kerma_strength: float
    - activity : float
    - source_dict: dict | Path | str: either a dictionary containing the source information, or a path to a json or a dicom plan file.
    ### Functions:
    - validate(): checks if the fields are valid for export.
    - to_dict(): converts the object to a dictionary.
    """

    treatment_type: Literal["HDR", "PLDR", "TLDR"] = "HDR"
    source_geometry: str = "GenericHDR"
    core_material: str = "G4_Ir"
    mass_number: int = 192
    atomic_number: int = 77
    air_kerma_per_history_strength: float = 1.158e-11 #cGy.cm^2/history
    air_kerma_strength: float = 36422.8 #U = cGy.cm^2/hour
    activity: float = 10.0 #Ci
    # source_dict: Union[dict, Path, str] = None

# Exclude ensures this field doesn't show up when you export the model to a dict/json
    pth_source: Path | str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def load_data_from_file(self) -> "BrachySource":
        if self.pth_source is not None:
            pth = Path(self.pth_source)
            
            if not pth.exists():
                raise ValueError(f"Path {pth} does not exist.")
            
            if pth.suffix == ".json":
                with open(pth, "r") as f:
                    data = json.load(f)
            elif pth.suffix == ".dcm":
                data = self.load_from_dicom(pth_dicom=pth)
            else:
                raise ValueError(f"File {pth} is not a json nor a dicom file.")
            
            # Dynamically update the instance attributes
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        return self

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
            "air_kerma_per_history_strength": self.air_kerma_per_history_strength,
            "air_kerma_strength": self.air_kerma_strength,
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
            + f"/source/core/material {self.core_material}\n"
            + f"/source/core/A {self.mass_number}\n"
            + f"/source/core/Z {self.atomic_number}\n"
            + f"/parallel_world/AKS {self.air_kerma_strength}\n"
            + f"/parallel_world/ak_per_history {self.air_kerma_per_history_strength}\n"
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
            json.dump(self.to_dict(), f, indent=4)
            
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

        source_dict = defaultdict(str)

        #load the constants dictionaries
        if getattr(cls, "SOURCE_CONSTANTS", None) is None:
            cls.SOURCE_CONSTANTS = json.load(open(Path(__file__).parent.parent.parent / "admin/constants/source_data.json", "r"))
        if getattr(cls, "ELEMENTS", None) is None:
            cls.ELEMENTS = json.load(open(Path(__file__).parent.parent.parent / "admin/constants/elements.json", "r"))

        #Figure out treatment type
        source_dict["treatment_type"] = plan_dcm.get("BrachyTreatmentType", "HDR")
        if source_dict["treatment_type"] == "MANUAL": # THIS IS WEIRD ##########################################################
            source_dict["treatment_type"] = "PLDR"
        try:
            model_name = plan_dcm.TreatmentMachineSequence[0].ManufacturerModelName
        except (AttributeError, IndexError):
            # If the sequence doesn't exist or is empty, look at the root level.
            model_name = getattr(plan_dcm, "ManufacturerModelName", "Unknown")

        #get air-kerma strength
        source_dict["air_kerma_strength"] = plan_dcm.SourceSequence[0].ReferenceAirKermaRate

        #figure out source model name
        source_dict["source_geometry"] = str(model_name)
        model_name = cls.check_manual_override_model_name(model_name)
        cls.update_source_data_for_model(source_dict, model_name)

        #check to make sure that the source parameters for the model from source_data.json match those from the DICOM (if they exist)
        cls.check_source_data_dicom_match(source_dict, plan_dcm)

        return source_dict

    @classmethod
    def check_manual_override_model_name(cls, model_name: str) -> str:
        r"""
        Purpose:
            - to check if the model name is a manual override and return the correct model name.
        Input:
            - model_name: str
        Output:
            - model_name: str
        Dependencies:
            - None
        """
        if "microselectron-hdr v2" in model_name.lower():
            model_name = "MicroSelectronV2"
        elif "microselectron v3" in model_name.lower():
            model_name = "MicroSelectronV3"
        elif "flexitron hdr 192-ir" in model_name.lower():
            model_name = "FlexiSource"
        elif "variseed" in model_name.lower():
            model_name = "AGX100"

        if model_name not in cls.SOURCE_CONSTANTS.keys():
            raise ValueError(f"Source model name {model_name} is not recognized. Please check the source_data.json file.")
        
        return model_name

    @classmethod
    def update_source_data_for_model(cls, source_dict: dict, model_name: str) -> None:
        r"""
        Purpose:
            - Updates the parameters of the BrachySource using the identified source model + the source_data.json file in /admin/constants.
        Input:
            - source_dict: dict
            - model_name: str
        Output:
            - None
        Dependencies:
            - None
        """
        source_data = cls.SOURCE_CONSTANTS[model_name]
        source_dict["core_material"] = source_data["core"]
        element = source_data["isotope"].split("-")[0]
        source_dict["atomic_number"] = list(cls.ELEMENTS["symbols"]).index(element) + 1
        source_dict["mass_number"] = int(source_data["isotope"].split("-")[1])
        source_dict["air_kerma_per_history_strength"] = source_data["AKHS"]
        #compute activity
        source_dict["activity"] = source_dict["air_kerma_strength"] / (source_data["AKSA"] * 3.7e10)  # convert to Ci

    @classmethod
    def check_source_data_dicom_match(cls, source_dict: dict, plan_dcm: pydicom.Dataset) -> None:
        # source_dict["source_geometry"] = plan_dcm.get("SourceModelName", "MicroSelectronV2")
        dcm_isotope = plan_dcm.SourceSequence[0].SourceIsotopeName
        isotope_candidates = ["Ir-192", "Co-60", "I-125", "Pd-103", "Cs-137"]
        for isotope in isotope_candidates: #to deal with some weirdness for how isotope names are written in the dicom
            if isotope.lower() in dcm_isotope.lower():
                dcm_isotope = isotope
                break
        if dcm_isotope != f"{cls.ELEMENTS['symbols'][source_dict['atomic_number']-1]}-{source_dict['mass_number']}":
            raise ValueError(f"Source isotope from DICOM ({dcm_isotope}) does not match source isotope from source_data.json ({cls.ELEMENTS['symbols'][source_dict['atomic_number']-1]}-{source_dict['mass_number']}). Please check the source_data.json file.")
        