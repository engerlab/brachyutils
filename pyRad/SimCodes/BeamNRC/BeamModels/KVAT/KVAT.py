"""KVAT model."""
import math
import os

from pyRad.utils import dicom_to_spherical


class KVAT(object):
    """KVAT model for UVic."""

    model_name = "KVAT"
    folder = "BEAM_slbl5"  # Relative path to BEAMnrc model from $EGS_HOME
    pegs_file = "allkV"

    default_dsource = 300.0  # (mm) distance between phsp plane and iso
    model_data = {
        "breast": {
            "num_beamlets": {1: 27, 2: 27, 3: 31, 4: 27},
        },
        "prostate": {
            "num_beamlets": {1: 27, 2: 27, 3: 27, 4: 27},
        },
        "lung": {
            "num_beamlets": {1: 31, 2: 31, 3: 31, 4: 29}
        },
        "cylinder": {
            "num_beamlets": 1
        },
        "phantom": {
            "num_beamlets": 9
        },
        "PrecRT": {
            "num_beamlets": {4: 70},
            "dsource": 400.0,
            "sad": 493.90
        }
    }

    def generate_cpt_beamlets(self, beamlet_creation):
        all_files_created = []

        for cpt in beamlet_creation.control_points:
            if not hasattr(cpt, "model"):
                cpt.model = "breast"

            if cpt.model == "breast" or cpt.model == "lung" or cpt.model == "prostate" or cpt.model == "PrecRT":
                if not hasattr(cpt, "target_size"):
                    # Default to 4 cm target size
                    cpt.target_size = 4

                for beamlet_index in range(self.model_data[cpt.model]["num_beamlets"][cpt.target_size]):
                    filename = self._make_beamlet_phsp_input(beamlet_index, cpt, beamlet_creation)
                    all_files_created.append(filename)
            elif cpt.model == "phantom" or cpt.model == "cylinder":
                for beamlet_index in range(self.model_data[cpt.model]["num_beamlets"]):
                    filename = self._make_beamlet_phsp_input(beamlet_index, cpt, beamlet_creation)
                    all_files_created.append(filename)

        return all_files_created

    def _make_beamlet_phsp_input(self, beamlet_index, cpt, beamlet_creation):
        server = beamlet_creation.server

        beamnrc_path = server.get_path("BeamNRC")
        beam_folder = self.folder
        beam_model_path = os.path.join(beamnrc_path, beam_folder)
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        phantom_filename = beamlet_creation.phantom_filename
        phantom_path = os.path.join(dosxyznrc_path, phantom_filename)

        if cpt.model == "breast":
            num_beams = self.model_data[cpt.model]["num_beamlets"][cpt.target_size]

            phsp_folder = "%icm_breast_%ibeams" % (int(cpt.target_size), num_beams)
            phsp_file = "%icm_breast_%ibeams_w%i.egsphsp1" % (int(cpt.target_size), num_beams, beamlet_index + 1)
            phsp_path = os.path.join(beam_model_path, phsp_folder, phsp_file)
        elif cpt.model == "prostate":
            num_beams = self.model_data[cpt.model]["num_beamlets"][cpt.target_size]

            phsp_folder = "%icm_prostate_%ibeams" % (int(cpt.target_size), num_beams)
            phsp_file = "%icm_prostate_%ibeams_w%i.egsphsp1" % (int(cpt.target_size), num_beams, beamlet_index + 1)
            phsp_path = os.path.join(beam_model_path, phsp_folder, phsp_file)
        elif cpt.model == "lung":
            num_beams = self.model_data[cpt.model]["num_beamlets"][cpt.target_size]

            phsp_folder = "%icm_lung_%ibeams" % (int(cpt.target_size), num_beams)
            phsp_file = "%icm_lung_%ibeams_w%i.egsphsp1" % (int(cpt.target_size), num_beams, beamlet_index + 1)
            phsp_path = os.path.join(beam_model_path, phsp_folder, phsp_file)
        elif cpt.model == "PrecRT":
            num_beams = self.model_data[cpt.model]["num_beamlets"][cpt.target_size]

            phsp_folder = "%icm_PrecRT_%ibeams" % (int(cpt.target_size), num_beams)
            phsp_file = "%icm_PrecRT_%ibeams_w%i.egsphsp1" % (int(cpt.target_size), num_beams, beamlet_index + 1)
            phsp_path = os.path.join(beam_model_path, phsp_folder, phsp_file)
        elif cpt.model == "phantom":
            phsp_folder = "2cmcentretarget_16cmcylinderphantom_30cmcollimator"
            phsp_file = "2cmcentretarget_16cmcylinderphantom_30cmcollimator_w%i.egsphsp1" % (beamlet_index + 1)
            phsp_path = os.path.join(beam_model_path, phsp_folder, phsp_file)
        elif cpt.model == "cylinder":
            phsp_folder = "3cmtoptarget_cylinderphantom"
            phsp_file = "3cmtoptarget_16cmcylinderphantom_30cmcollimator_w5.egsphsp1"
            phsp_path = os.path.join(beam_model_path, phsp_folder, phsp_file)

        theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle,
                                                cpt.couch_angle,
                                                cpt.col_angle)

        if not hasattr(cpt, "dsource"):
            cpt.dsource = self.default_dsource

        params = {
            "phsp_path": phsp_path,
            "phantom_path": phantom_path,
            "theta": math.degrees(theta),
            "phi": math.degrees(phi),
            "phicol": math.degrees(phicol),
            "dsource": cpt.dsource / 10.0,
            "iso_x": cpt.iso[0] / 10.0,
            "iso_y": cpt.iso[1] / 10.0,
            "iso_z": cpt.iso[2] / 10.0,
            "nhist": int(beamlet_creation.settings["nhist"]),
            "title": "beamletGeneration"
        }

        template = self._load_dosxyz_template_string()
        filename = "%s_%ikeV_%.1f_%i.egsinp" % (beamlet_creation.settings["name"], int(cpt.energy), cpt.gantry_angle, beamlet_index)
        with open(filename, "w") as myfile:
            myfile.write(template.format(**params))

        return filename

    def _load_dosxyz_template_string(self):
        template_dir = os.path.dirname(__file__)  # Looks at the current directory
        template_filename = "KVAT_dosxyznrc.egsinp"
        template_path = os.path.join(template_dir, template_filename)
        template_file = open(template_path)
        template_string = template_file.read()
        template_file.close()

        return template_string
