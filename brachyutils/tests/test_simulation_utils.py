from brachyutils.simulation_utils import BrachySimulation


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
    test_brachy_simulation()
