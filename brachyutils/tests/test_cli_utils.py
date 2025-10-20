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
    # dir_input = Path("/home/ubuntu/YourLocalHome/Data/prostate/prostate-glen-2023")
    # dir_output = Path("data_test/test_export_plan/prostate")

    # # for aapm tg 132 data
    dir_input = Path("/home/ubuntu/YourLocalHome/Data/registration/AAPM-TG132-Data/geometric phantom/Geometric TG132Phantom")
    dir_output = Path("data_test/test_export_plan/aapm_tg132")
    type_out = PhantomType.NRRD
    convert_phantom(
        pth_inputs=list(dir_input.glob("*/")),
        type_out=type_out,
        dir_output=dir_output,
        multi_proc=False
    )

def test_convert_egsphant():
    from brachyutils.cli_utils import convert_egsphant, EgsphantType
    dir_input = Path("data_test/prostate-glen-p1-planFiles/cropped_ct.egsphant")
    # dir_input = Path("data_test/test_export_plan/prostate/cropped_ct.seq.nrrd")
    dir_output = Path("data_test/test_export_plan/prostate")
    type_out = EgsphantType.EGS

    convert_egsphant(
        pth_inputs=[dir_input],
        type_out=type_out,
        dir_output=dir_output,
        multi_proc=False
    )

if __name__ == "__main__":
    # test_convert_dose()
    # test_convert_phantom()
    test_convert_egsphant()