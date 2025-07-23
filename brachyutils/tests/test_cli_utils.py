from pathlib import Path

def test_convert_dose():
    from brachyutils.cli_utils import convert_dose, DoseType
    from brachyutils.cli_utils import DoseType
    dir_inputs = Path("data_test/prostate-glen-p1-dose")
    dir_output = Path("data_test/test_export_plan/prostate")
    type_out = DoseType.THREE_DDOSE

    convert_dose(
        pth_inputs=list(dir_inputs.glob("*6.*.nrrd")),
        type_out=type_out,
        dir_output=dir_output,
        multi_proc=True
        )

def test_convert_phantom():
    from brachyutils.cli_utils import convert_phantom, PhantomType
    # # for a single file
    # dir_input = Path("data_test/prostate-glen-p1-dcm")
    # dir_output = Path("data_test/test_export_plan/prostate")

    # # for multiple files
    dir_input = Path("/home/ubuntu/YourLocalHome/Data/prostate/prostate-glen-2023")
    dir_output = Path("data_test/test_export_plan/prostate")

    type_out = PhantomType.NRRD
    convert_phantom(
        pth_inputs=list(dir_input.glob("*/")),
        type_out=type_out,
        dir_output=dir_output,
        multi_proc=True
    )

if __name__ == "__main__":
    # test_convert_dose()
    test_convert_phantom()
