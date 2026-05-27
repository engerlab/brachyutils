from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, computed_field

pth_brachyutils = Path(__file__).parent.parent.parent.resolve()

class ExportConfig_PlanAndMac(BaseModel):
    """
    ### Purpose:
    - Configuration for exporting both .plan and .mac files.
    
    ### Attributes:
    - dir_export: Directory where the plan and mac files are exported.
    - combined_only: If true, only combined files are written.
    - name_combined: The base name for the combined files (used for both .plan and .mac).
    - body_mesh_name: Name of the body structure to be saved as a separate STL.
    - pth_phantom: The relative path to the egsphant file.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    dir_export: Path | None = Field(default=None, description="Directory where the plan and mac files are exported.")
    combined_only: bool = Field(default=True, description="If true, only combined files are written.")
    name_combined: str = Field(default="combined", description="The base name for the combined files (used for both .plan and .mac).")
    auto_mvm: bool = Field(default=True, description="Whether to automatically determine if MVM at export time based on phantom size.")
    body_mesh_name: str | None = Field(default=None, description="Name of the body structure for MVM phantom.")
    body_mesh_material: str = Field(default="SoftTissue", description="Material name for the body structure in MVM phantom.")
    pth_phantom: Path = Field(default=Path("egsphant.seq.nrrd"), description="The relative path to the egsphant file.")
    
    @computed_field
    def pth_plan_combined(self) -> Path | None:
        if self.dir_export is None:
            return None
        return self.dir_export / f"{self.name_combined}.plan"

    @computed_field
    def pth_mac_combined(self) -> Path | None:
        if self.dir_export is None:
            return None
        return self.dir_export / f"{self.name_combined}.mac"
        
    @computed_field    
    def pth_body_stl(self) -> Path | None:
        if self.dir_export is None:
            return None
        return self.dir_export / f"{self.body_mesh_name}.stl"

class ExportConfig_Egsphant(BaseModel):
    r"""
    ### Purpose:
    - The Export info needed for exporting Egsphant files.
    - If using Monte Carlo simulations from RapidBrachyMC, It is recommended that
    the user crop the egsphant to a small region around the relevant anatomy and
    use provide the body_mesh_name to save the body structure as a separate STL file.

    ### Attributes:
    - dir_export: Directory where Egsphant file is exported.
    - name: File name for Egsphant output.
    - file_extension: Allowed file extensions for Egsphant files.
    - material_dict: Dictionary of material names and their properties.
    - assign_material_from_ct: Assign materials from CT data or based on contours.
    - crop_by_contour: Name of the contour(s) to crop by. Union is used if a list.
    - marginInMM: Margin in mm to add around the contour when cropping the phantom.
    - resampled_spacing: Spacing for resampling the phantom.
    - resampled_origin: Origin for resampling the phantom.
    - background_material: Material name for background.
    - strict_name_match: Enforce strict name matching for materials.
    - body_mesh_name: Name of the body structure to be saved as a separate STL.
    - try_direct_export: Try to export the brachyplan's phantom's egsphant directly without recreating it, if possible
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dir_export: str | Path = Field(None, description="Directory where Egsphant file is exported.")
    name: str = Field("egsphant", description="File name for Egsphant output.")
    file_extension: Literal[".seq.nrrd", ".egsphant"] = Field(
        ".seq.nrrd", 
        description="Allowed file extensions for Egsphant files.")
    material_dict: dict | Path = Field(
        Path(pth_brachyutils/"admin/constants/structure_materials_prostate.json"),
        description="Dictionary of material names and their properties.")
    assign_material_from_ct: bool = Field(False, description="Whether to assign materials from CT data or based on contours.")
    crop_by_contour: str | List[str] = Field(None, description="Name of the contour to crop by. \
If a list of strings is provided, the union of the contours will be used to crop the phantom.")
    marginInMM: float = Field(10.0, description="Margin in mm to add around the contour when cropping the phantom.")
    resampled_spacing: List[float] = Field(None, description="Spacing for resampling the phantom.")
    resampled_origin: List[float] = Field(None, description="Origin for resampling the phantom.")
    background_material: str = Field("Air", description="Material name for background.")
    strict_name_match: bool = Field(True, description="Whether to enforce strict name matching for materials.")
    @computed_field
    def pth_egsphant(self)->Path:
        return self.dir_export/(self.name+self.file_extension)
class ExportConfig_CatheterTable(BaseModel):
    r"""
    ### Purpose:
    - Configuration for exporting catheter tables.

    ### Attributes:
    - dir_export: Directory where catheter table is exported.
    - name: File name for catheter table output.
    - file_extension: File extension for catheter table export.
    - remove_text: Text to remove from dwell names.
    - one_markup_per_catheter: Whether to create one markup per catheter.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dir_export: str | Path = Field(None, description="Directory where catheter table is exported.")
    name: str = Field("catheter_table", description="File name for catheter table output.")
    file_extension: Literal[".json", ".mrk.json"] = Field(
        ".mrk.json", description="File extension for catheter table export.)")
    remove_text: bool = Field(True, description="Text to remove from dwell names.")
    one_markup_per_catheter: bool = Field(False, description="Whether to create one markup per catheter.")
    @computed_field
    def pth_catheter_table(self)->Path:
        self.dir_export = Path(self.dir_export)
        return self.dir_export/(self.name+self.file_extension)

class ExportConfig_Dose(BaseModel):
    r"""
    ### Purpose:
    - Configuration for exporting dose data from the plan.

    ### Attributes:
    - dir_export: Directory where dose files are exported.
    - name_combined: File name for combined dose output.
    - file_extension: Allowed file extensions for dose files.
    - write_dose_rate_maps: Whether to write individual dose rate maps to files.
    - multi_processing: Enable multiprocessing for export (yes/no toggle).
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True  # Enables auto-docs from Field desc [web:48]
    )
    dir_export: str | Path = Field(None, description="Directory where dose files are exported.")
    name_combined: str = Field("combined", description="File name for combined dose output.")
    file_extension: Literal[".seq.nrrd", ".3ddose"] = Field(
        ".seq.nrrd", description="Allowed file extensions for dose files."
    )
    write_dose_rate_maps: bool = Field(
        False, description="Whether to write individual dose rate maps to files."
    )
    multi_processing: bool = Field(
        True, description="Enable multiprocessing for export (yes/no toggle)."
    )
    @computed_field
    def pth_combined(self)->Path:
        self.dir_export = Path(self.dir_export)
        return self.dir_export/(self.name_combined+self.file_extension)

class ExportConfig_BrachyPlan(BaseModel):
    r"""
    ### Purpose:
    - Configuration for exporting various components of a brachytherapy treatment plan.
    The components are catheter table, dose, egsphant, plan file, and mac file.

    ### Attributes:
    - dir_export: **[required]** Base directory where all plan components are exported.
    - export_config_dose: Configuration for exporting dose data.
    - export_config_cathetertable: Configuration for exporting catheter table.
    - export_config_egsphant: Configuration for exporting egsphant file.
    - export_config_plan_and_mac: Configuration for exporting plan and mac file.
    - applicator_geometry: Whether to export applicator geometry into an STL file.
    - structure_set: Whether to export structure set info into a JSON file.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dir_export:str | Path = Field(..., description="Base directory where all plan components are exported.")
    export_config_dose: ExportConfig_Dose | bool = Field(False, description="Configuration for exporting dose data.")
    export_config_cathetertable: ExportConfig_CatheterTable | bool = Field(False, description="Configuration for exporting catheter table.")
    export_config_egsphant: ExportConfig_Egsphant | bool = Field(False, description="Configuration for exporting egsphant file.")
    export_config_plan_and_mac: ExportConfig_PlanAndMac | bool = Field(False, description="Configuration for exporting .plan and .mac files.")
    # TODO: in future, add these export configs if neeeded

    # export_config_applicator: ExportConfig_Applicator = None
    # export_config_phantom: ExportConfig_BrachyStructure = None

    applicator_geometry: bool = Field(False, description="Whether to export applicator geometry into a stl file.")
    structure_set: bool = Field(False, description="Whether to export structure set info into a json file.")

    @model_validator(mode="before")
    def validate_inputs(data):
        for k, v in data.items():
            if k.startswith("export_config_") and isinstance(v, bool):
                data[k] = {} if v else False
            if k=="dir_export":
                data[k] = Path(v)
        return data

    @model_validator(mode="after")
    def validate_config(self):
        # make sure that the paths of dir exports are 
        # set correctly for all the inner attributes
        for _, value in self:                
            if isinstance(value, BaseModel):
                if value.dir_export is None:
                    value.dir_export = self.dir_export
        return self