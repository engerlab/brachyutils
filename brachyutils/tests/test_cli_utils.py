from pathlib import Path

def test_convert_dose():
    from brachyutils.cli_utils import convert_dose
    dir_inputs = Path("data_test/prostate-glen-p1-dose")
    dir_output = Path("data_test/test_export_plan/prostate")
    type_out = ".3ddose"

    convert_dose(
        pth_inputs=list(dir_inputs.glob("*combined*.nrrd")),
        type_out=type_out,
        dir_output=dir_output,
        multi_proc=False)

if __name__ == "__main__":
    test_convert_dose()
    print("Test passed successfully.")

