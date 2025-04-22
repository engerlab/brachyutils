from brachyutils.planning.simulation_utils import BrachySimulation, BrachySource
from pathlib import Path

def test_brachy_source():
    source_obj = BrachySource(
        treatment_type="PLDR",
        source_geometry="MicroSelectronV2",
        core_material="G4_Ir",
        mass_number="192",
        atomic_number="77",
        air_kerma_per_history=1.149000e-11,
        reference_air_kerma_rate=4.278729e04,
    )
    print(source_obj.to_dict())

    source_dict = {
        "treatment_type": "PLDR",
        "source_geometry": "MicroSelectronV2",
        "core_material": "G4_Ir",
        "mass_number": "192",
        "atomic_number": "77",
        "air_kerma_per_history": 1.149000e-11,
        "reference_air_kerma_rate": 4.278729e04,
    }
    source_obj = BrachySource(**source_dict)
    print(source_obj.to_dict())

    source_path = list(Path("data_test/prostate-glen-p1-dcm").glob("RP*.dcm")).pop()
    source_obj = BrachySource(source_dict=source_path)
    print(source_obj.to_dict())

def test_brachy_simulation():
    sim_dict = {
        "brachy_source": {
            "treatment_type": "HDR",
            "source_geometry": "MicroSelectronV2",
            "core_material": "G4_Ir",
            "mass_number": "192",
            "atomic_number": "77",
            "air_kerma_per_history": 1.149000e-11,
            "reference_air_kerma": 4.278729e04, 
        },
        "pth_plan": "combined.plan",
        "pth_phantom": "ct.egsphant",
        "number_histories": 1000000,
        "total_time": 5983,
        "number_of_threads": 12,
        "print_progress": 10000,
        "beam_on": 10000,
        "control_verbose": 0,
        "dose_format": "nrrd",
        "tracking_verbose": 0,
        "world_material": "Air",
        "run_verbose": 0,
    }
    sim = BrachySimulation(simulation_dict=sim_dict)
    print(sim.to_string())


if __name__ == "__main__":
    test_brachy_source()
    # test_brachy_simulation()
    
