import matplotlib.pyplot as plt
import numpy as np
from brachyutils.film_utils import CalibrationCurve, FilmCalibration


def test_create_lewis_calibration_curve():
    test_curve_type = "Lewis"
    test_doses = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    test_r_pv = np.power(2.0, -test_doses)
    test_g_pv = np.copy(test_r_pv)
    test_b_pv = np.copy(test_r_pv)
    test_r_std = np.ones(test_r_pv.shape)
    test_g_std = np.ones(test_g_pv.shape)
    test_b_std = np.ones(test_b_pv.shape)
    test_calibration_curve = CalibrationCurve(
        test_doses,
        test_r_pv,
        test_g_pv,
        test_b_pv,
        test_r_std,
        test_g_std,
        test_b_std,
        test_curve_type,
    )
    fit_ground_truth = np.array(
        [-0.27395561, 1.85194717, 1.45033183]
    )  # ground truth extracted using a separate python script
    assert np.allclose(test_calibration_curve.r_opt, fit_ground_truth, rtol=1e-6)
    assert np.allclose(test_calibration_curve.g_opt, fit_ground_truth, rtol=1e-6)
    assert np.allclose(test_calibration_curve.b_opt, fit_ground_truth, rtol=1e-6)


def test_create_devic_calibration_curve():
    test_curve_type = "Devic"
    test_doses = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    test_r_pv = np.power(2.0, -test_doses)
    test_g_pv = np.copy(test_r_pv)
    test_b_pv = np.copy(test_r_pv)
    test_r_std = np.ones(test_r_pv.shape)
    test_g_std = np.ones(test_g_pv.shape)
    test_b_std = np.ones(test_b_pv.shape)
    test_calibration_curve = CalibrationCurve(
        test_doses,
        test_r_pv,
        test_g_pv,
        test_b_pv,
        test_r_std,
        test_g_std,
        test_b_std,
        test_curve_type,
    )
    fit_ground_truth = np.array(
        [1.14490056, 0.78425477]
    )  # ground truth extracted using a separate python script
    assert np.allclose(test_calibration_curve.r_opt, fit_ground_truth, rtol=1e-6)
    assert np.allclose(test_calibration_curve.g_opt, fit_ground_truth, rtol=1e-6)
    assert np.allclose(test_calibration_curve.b_opt, fit_ground_truth, rtol=1e-6)


def test_load_calibration_films():
    test_calibration_films_path = "../../data_test/test_calibration_films/"
    pixel_range = 65535.0  # 2^32 -1
    test_file_dict = dict()  # dict maps dose to array of file names
    test_file_dict[0] = ["0Gy062.tif", "0Gy063.tif"]
    test_file_dict[0.25] = [
        "0_25_1Gy058.tif",
        "0_25_1Gy059.tif",
        "0_25_2Gy060.tif",
        "0_25_2Gy061.tif",
    ]
    test_file_dict[0.5] = [
        "0_5_1Gy053.tif",
        "0_5_1Gy054.tif",
        "0_5_2Gy056.tif",
        "0_5_2Gy057.tif",
    ]
    test_file_dict[1] = ["1_1Gy049.tif", "1_1Gy050.tif", "1_2Gy051.tif", "1_2Gy052.tif"]
    test_file_dict[1.5] = [
        "1_5_1Gy045.tif",
        "1_5_1Gy046.tif",
        "1_5_2Gy047.tif",
        "1_5_2Gy048.tif",
    ]
    test_file_dict[2] = ["2Gy043.tif", "2Gy044.tif"]
    test_file_dict[2.5] = ["2_5Gy041.tif", "2_5Gy042.tif"]
    test_file_dict[3] = ["3Gy039.tif", "3Gy040.tif"]
    test_file_dict[4] = ["4Gy037.tif", "4Gy038.tif"]
    test_file_dict[5] = ["5Gy035.tif", "5Gy036.tif"]
    test_file_dict[6] = ["6Gy033.tif", "6Gy034.tif"]
    test_file_dict[7] = ["7Gy031.tif", "7Gy032.tif"]
    test_file_dict[8] = ["8Gy029.tif", "8Gy030.tif"]
    test_file_dict[9] = ["9Gy027.tif", "9Gy028.tif"]
    test_file_dict[10] = ["10Gy025.tif", "10Gy026.tif"]
    test_file_dict[12] = ["12Gy023.tif", "12Gy024.tif"]
    test_file_dict[14] = [
        "14Gy021.tif",
        "14Gy022.tif",
    ]  # ignoring _end files, of unknown purpose
    test_file_dict[16] = ["16Gy019.tif", "16Gy020.tif"]
    test_file_dict[18] = ["18Gy017.tif", "18Gy018.tif"]
    test_file_dict[20] = ["20Gy015.tif", "20Gy016.tif"]
    test_file_dict[24] = ["24Gy013.tif", "24Gy014.tif"]
    test_file_dict[28] = ["28Gy011.tif", "28Gy012.tif"]
    test_file_dict[32] = ["32Gy008.tif", "32Gy009.tif"]
    test_file_dict[36] = ["36Gy006.tif", "36Gy007.tif"]
    test_file_dict[40] = ["40Gy004.tif", "40Gy005.tif"]
    # add the proper directory
    for d in test_file_dict.keys():
        for i in range(len(test_file_dict[d])):
            test_file_dict[d][i] = test_calibration_films_path + test_file_dict[d][i]
    test_calibration = FilmCalibration()
    test_calibration.pixel_range = pixel_range
    test_calibration.calibration_curve_type = "Lewis"
    test_calibration.calibration_file_dict = test_file_dict
    # for d in test_file_dict.keys():
    #    test_calibration.add_calibration_files_for_dose(test_file_dict[d])
    test_calibration.load_calibration()
    test_calibration.create_calibration_curve()
    r_opt_ground_truth = np.array([0.09951261, 2.66066357, 3.07706001])
    g_opt_ground_truth = np.array([0.05982643, 5.28318534, 5.74549658])
    b_opt_ground_truth = np.array([0.16486698, 10.10602592, 12.20989423])
    # the following assertions on the calibration curve opt parameters appear highly sensitive
    # to the python/scipy version. We will use a large tolerance of 0.2 based on observed variation
    assert np.allclose(
        test_calibration.calibration_curve.r_opt, r_opt_ground_truth, rtol=2e-1
    )
    assert np.allclose(
        test_calibration.calibration_curve.g_opt, g_opt_ground_truth, rtol=2e-1
    )
    assert np.allclose(
        test_calibration.calibration_curve.b_opt, b_opt_ground_truth, rtol=2e-1
    )


def test_load_calibraton_object():
    pass


def main():
    r"""
    This main function creates or loads a calibration object for film calibration.

    Returns:
    None
    """
    FilmCalibration.create_or_load_calibration_object()
    plt.show()


if __name__ == "__main__":
    main()
