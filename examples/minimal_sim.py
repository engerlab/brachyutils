from brachyutils import (
    get_uniform_phantom, BrachyPlan,
    CatheterTable, Catheter, DwellPosition
)
def export_plan_air_phantom():
    """
    Create a minimal simulation of a brachytherapy plan with an air phantom.
    """
    dir_export = "temp_data/mc/test_Pd103"
    catheterTable = CatheterTable(
        catheter_list=[
            Catheter(
                index=0,
                dwells=[
                    DwellPosition(
                        index=0,
                        position=[50, 50, 50],
                        relativePos=0,
                        rotation=[0.0, 0.0, 0.0],
                        time=1.0,
                    )]
                )
            ]
    )
    sim_dict = {
        "source_dict": {
            "treatment_type": "TLDR",
            "source_geometry": "IsoAid_Advantage",
            "core_material": "G4_Pd",
            "mass_number": "103",
            "atomic_number": "46",
            "air_kerma_per_history": 1.149000e-11,
            "reference_air_kerma": 5e04,
        },
        "pth_plan": "combined.plan",
        "pth_phantom": "ct.egsphant",
        "number_histories": 10000,
        "total_time": 1,
        "number_of_threads": 10,
        "PrintProgress": 1000,
    }
    content_to_export = {
        "egsphant": True,
        "plan": True,
        "mac": True,
        "ApplicatorMaterials": False,
        "applicator_geometry": False,
    }
    # Create a minimal simulation of a brachytherapy plan with an air phantom
    air_phantom = get_uniform_phantom(voxel_value=-1000)
    # Export the plan to a JSON file
    plan = BrachyPlan(
        phantom=air_phantom,
        simulation_dict=sim_dict,
        catheter_table=catheterTable
        )
    plan.export_brachy_plan(dir_export=dir_export, content_to_export=content_to_export)
    print("Brachytherapy plan exported to minimal_brachy_plan.json")

if __name__ == "__main__":
    export_plan_air_phantom()