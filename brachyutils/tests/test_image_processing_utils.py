from brachyutils.image_processing_utils import RegistrationWithOpenTPS
from brachyutils.geometry_utils import BrachyPhantom

def test_register_opentps():
    pth_static = "path/to/static/phantom"
    pth_moving = "path/to/moving/phantom"
    pth_output = "path/to/output/phantom"

    static_phantom = BrachyPhantom(dir_dicom=pth_static)
    moving_phantom = BrachyPhantom(dir_dicom=pth_moving)
    
    mode = {"deformable": False, "algorithm": "quick"}
    mode = {"deformable": False, "algorithm": "demons"}
    mode = {"deformable": False, "algorithm": "morphons"}
    mode = {"deformable": True, "algorithm": None}
    registration_obj = RegistrationWithOpenTPS(
        static_phantom=static_phantom,
        moving_phantom=moving_phantom,
        deformable=mode["deformable"],
        algorithm=mode["algorithm"],
        )

    deformed_phantom = registration_obj.register()
    deformed_phantom.write_image_to_nrrd(pth_output)

if __name__ == "__main__":
    print("testing the registration class")