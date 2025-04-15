from brachyutils import get_uniform_phantom, BrachyPlan

def export_plan_air_phantom():
    """
    Create a minimal simulation of a brachytherapy plan with an air phantom.
    """
    # Create a minimal simulation of a brachytherapy plan with an air phantom
    air_phantom = get_uniform_phantom(voxel_value=-1000)
    # Export the plan to a JSON file
    plan = BrachyPlan(phantom=air_phantom)
    plan.export_to_json("minimal_brachy_plan.json")
    print("Brachytherapy plan exported to minimal_brachy_plan.json")

if __name__ == "__main__":
    export_plan_air_phantom()