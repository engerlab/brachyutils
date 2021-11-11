"""
Generic Varian electron beam model module.

Copyright Marc-Andre Renaud, 2017
"""
import os
import math
import subprocess
import numpy

from pyRad.utils import dicom_to_spherical

class GenericVarian_E(object):
    """Generic Varian beam model."""

    model_name = "GenericVarian_E"
    folder = "BEAM_GenericVarian_E"
    pegs_file = "radify"

    dcoll = 51.01
    default_dsource = 40.0

    y_z_jaw_front = 28.0
    y_z_jaw_back = 35.6485
    x_z_jaw_front = 36.7
    x_z_jaw_back = 44.3485
    x_jaw_width = 7.80
    y_jaw_width = 7.77
    y_jaw_arc_radius = 28.16

    jaw_table = {
        "open" : [-200, 200], 
        "4": {
            "25x25": [-160, 160],
            "20x20": [-135, 135],
            "15x15": [-110, 110],
            "10x10": [-55, 55],
            "6x6": [-50, 50],
            "10x6": [-80, 65]
        },
        "6": {
            "25x25": [-160, 160],
            "20x20": [-135, 135],
            "15x15": [-110, 110],
            "10x10": [-55, 55],
            "6x6": [-50, 50],
            "10x6": [-80, 65]
        },
        "9": {
            "25x25": [-150, 150],
            "20x20": [-125, 125],
            "15x15": [-100, 100],
            "10x10": [-50, 50],
            "6x6": [-50, 50],
            "10x6": [-80, 65]
        },
        "12": {
            "25x25": [-150, 150],
            "20x20": [-125, 125],
            "15x15": [-95, 95],
            "10x10": [-75, 75],
            "6x6": [-55, 55],
            "10x6": [-80, 55]
        },
        "15": {
            "25x25": [-140, 140],
            "20x20": [-115, 115],
            "15x15": [-95, 95],
            "10x10": [-75, 75],
            "6x6": [-55, 55],
            "10x6": [-80, 50]
        },
        "16": {
            "25x25": [-140, 140],
            "20x20": [-115, 115],
            "15x15": [-90, 90],
            "10x10": [-75, 75],
            "6x6": [-55, 55],
            "10x6": [-80, 50]
        },
        "18": {
            "25x25": [-135, 135],
            "20x20": [-110, 110],
            "15x15": [-90, 90],
            "10x10": [-75, 75],
            "6x6": [-55, 55],
            "10x6": [-80, 50]
        },
        "20": {
            "25x25": [-135, 135],
            "20x20": [-110, 110],
            "15x15": [-85, 85],
            "10x10": [-70, 70],
            "6x6": [-55, 55],
            "10x6": [-80, 50]
        },
        "22": {
            "25x25": [-135, 135],
            "20x20": [-110, 110],
            "15x15": [-85, 85],
            "10x10": [-70, 70],
            "6x6": [-55, 55],
            "10x6": [-80, 50]
        }
    }

    def __init__(self):
        """
        Constructor.

        Set up the paths to the template files for this beam model. I'm
        almost certain there is a better way to do this.
        """
        self.template_folder = os.path.dirname(__file__)

    def make_sim_inputs(self, simulation):
        """
        Create input files for DOSXYZnrc simulation using a BEAMnrc beam model.
        """
        params = {}
        params["name"] = simulation.name

        # Generic beam model parameters
        params["applicator_type"] = simulation.settings["linac"]["applicator"]
        
        if "cutout" in simulation.settings["linac"]:
            params["cutout"] = simulation.settings["linac"]["cutout"]

        params["cpts"] = self._process_beams(simulation.beams, params["applicator_type"])
        try:
            params["energy"] = params["cpts"][0]["energy"]
        except KeyError:
            print "Energy not found, assuming 6 MV"
            params["energy"] = 6

        energy_identifier = str(int(params["energy"]))
        # Specific data related to Generic models
        generic_params = simulation.settings["linac"]["beam_parameters"][energy_identifier]
        params["e_energy"] = generic_params["e_energy"]
        params["e_fwhm"] = generic_params["e_fwhm"]

        beamnrc_path = simulation.server.get_path("BeamNRC")
        beam_folder = self.folder
        beam_model_path = os.path.join(beamnrc_path, beam_folder)
        dosxyznrc_path = os.path.join(beamnrc_path, "dosxyznrc")

        params["pegs_file"] = self.pegs_file
        params["beam_inputfile"] = params["name"] + "_beam.egsinp"
        params["beam_model"] = self.folder
        params["title"] = "Plan recalculation"
        params["nhist"] = simulation.settings["nhist"]
        params["phantom_path"] = os.path.join(dosxyznrc_path,
                                              simulation.phantom_filename)

        zero_air_dose = simulation.settings.get("zero_air_dose", 1)
        if not zero_air_dose:
            zero_air_dose = 0
        else:
            zero_air_dose = 1
        params["zero_air_dose"] = zero_air_dose

        # Automatically put dynamic delivery mode since static modes
        # are taken into account by duplicating control points.
        params["delivery_mode"] = 1

        params["commissioning"] = simulation.settings.get("commissioning", False)
        params["commissioning_type"] = simulation.settings.get("commissioning_type", None)
        params["filler_depth"] = simulation.settings.get("filler_depth", 0.0)
        if params["commissioning_type"] is "profile":
            measurement_depth = simulation.settings.get("measurement_depth", 10.0)
            comm_dict = self._process_commissioning(simulation.beams, measurement_depth)
            params.update(comm_dict)

        #mlc_filename = self._make_mlc_file(params)
        jaws_filename = self._make_jaws_file(params)
        params["jaws_file"] = os.path.join(beam_model_path,
                                           jaws_filename)

        files_created = {}
        #files_created["mlc_file"] = mlc_filename
        files_created["jaws_file"] = [jaws_filename]
        files_created["beamnrc_file"] = [self._write_beamnrc_input(params)]
        files_created["dosxyznrc_file"] = [self._write_dosxyznrc_input(params)]

        return files_created

    def get_beamnrc_template(self, params):
        """
        Return filename of BEAMnrc template. Different energies sometimes
        have different template files.
        """
        ident = str(params["energy"]) + "E"

        if params.get("commissioning", False):
            template = self.model_name + "_{}_commissioning.egsinp".format(ident)
        else:
            template = self.model_name + "_{}.egsinp".format(ident)

        return template

    def get_dosxyznrc_template(self, params):
        """
        Return filename of DOSXYZnrc template. All energies should share
        the same template.
        """
        commissioning_type = params.get("commissioning_type", None)

        if commissioning_type is not None:
            template = "dosxyznrc_{ctype}_commissioning.egsinp".format(ctype=commissioning_type)
        else:
            template = "dosxyznrc_template.egsinp"

        return template

    @staticmethod
    def get_calibration(simulation, plan_obj):
        """
        Return the calibration factor to go from DOSXYZnrc dose to absolute dose.

        :param plan_obj: Plan object with the number of monitor units.
        """
        parsed_plan = plan_obj.parse_plan()

        try:
            energy = str(parsed_plan["beams"][0]["cpts"][0]["energy"])
        except KeyError:
            energy = "6"

        '''
        if "FFF" in parsed_plan["beams"][0] and parsed_plan["beams"][0]["FFF"]:
            energy += "FFF"
        '''

        if energy not in simulation.settings["machine"]["calibration_factors"]:
            raise Exception("Calibration factor not found for energy: %s" % energy)

        calibration = simulation.settings["machine"]["calibration_factors"][energy]

        return parsed_plan["total_mus"] * (1.0 / calibration)

    def _process_beams(self, beams, applicator_type):
        """
        Transform beam data into the format required by BeamNRC.

        DICOM defines most attributes at isocenter, BeamNRC needs them at
        the actual physical height of the components.
        """

        total_mus = sum([beam["beam_meterset"] for beam in beams])

        processed_cpts = []

        for beam in beams:
            is_static = "static" in beam and beam["static"] is True

            for cpt in beam["cpts"]:
                energy_ident = str(int(cpt.energy))
                if applicator_type == "air":
                    x_jaws = self.jaw_table["open"]
                    y_jaws = self.jaw_table["open"]
                else:
                    x_jaws = self.jaw_table[energy_ident][applicator_type]
                    y_jaws = self.jaw_table[energy_ident][applicator_type]

                cpt_dict = {}
                cpt_dict.update(self._process_y_jaws(y_jaws))
                cpt_dict.update(self._process_x_jaws(x_jaws))

                cpt_dict["weight"] = cpt.cum_weight / total_mus
                cpt_dict["iso"] = [x / 10.0 for x in cpt.iso]

                theta, phi, phicol = dicom_to_spherical(cpt.gantry_angle, cpt.couch_angle, cpt.col_angle)
                cpt_dict["theta"] = math.degrees(theta)
                cpt_dict["phi"] = math.degrees(phi)
                cpt_dict["phicol"] = math.degrees(phicol)

                cpt_dict["dsource"] = self.default_dsource
                cpt_dict["energy"] = cpt.energy

                processed_cpts.append(cpt_dict)

        return processed_cpts

    def _process_commissioning(self, beams, measurement_depth=10.0):
        x_jaws_iso = [x / 10.0 for x in beams[0]["cpts"][0].x_jaw]
        y_jaws_iso = [y / 10.0 for y in beams[0]["cpts"][0].y_jaw]
        phantom_neg_x = x_jaws_iso[0] - 5.0
        phantom_neg_y = y_jaws_iso[0] - 5.0

        depth_vox_size = 0.5
        phantom_depth = measurement_depth - 0.5 * (3 * depth_vox_size)

        vox_size = 0.25
        num_vox_x = int(2 * abs(phantom_neg_x) / vox_size)
        num_vox_y = int(2 * abs(phantom_neg_y) / vox_size)

        return {
            "phantom_neg_x": phantom_neg_x,
            "phantom_neg_y": phantom_neg_y,
            "num_vox_x": num_vox_x,
            "num_vox_y": num_vox_y,
            "phantom_depth": phantom_depth,
            "vox_size": vox_size,
            "depth_vox_size": depth_vox_size
        }

    def _process_y_jaws(self, jaw_pos):
        """
        Convert from DICOM-specified y jaw positions to BEAMnrc. DICOM jaw
        positions are defined as the field projection at isocenter.

        On Varian linacs, the Y jaws travel in an arc based on the projected
        field such that the jaw face is parallel to the beam divergence.
        BEAMnrc does not allow left/right jaws to have a different height,
        so the average height of both jaws is taken. These are very small
        adjustments and should not introduce any inaccuracies.
        """
        # Calculate left y jaw height
        jaw_dict = {}

        theta = math.atan2(abs(jaw_pos[0]) / 10.0, 100.0)
        costheta = math.cos(theta)
        left_jaw_front = self.y_jaw_arc_radius * costheta

        # Right y jaw height
        theta = math.atan2(abs(jaw_pos[1]) / 10.0, 100.0)
        costheta = math.cos(theta)
        right_jaw_front = self.y_jaw_arc_radius * costheta

        # Take the mean of the two in case of asymmetric jaws.
        # Difference in height between the "zero" position and the arc position
        # Should be a negative value.
        difference = 0.5 * (left_jaw_front + right_jaw_front) - self.y_jaw_arc_radius

        y_z_jaw_front = self.y_z_jaw_front + difference
        y_z_jaw_back = (y_z_jaw_front + self.y_jaw_width) * costheta

        jaw_dict["y_z_jaw_front"] = y_z_jaw_front
        jaw_dict["y_z_jaw_back"] = y_z_jaw_back
        adjusted_y_jaw_width = y_z_jaw_back - y_z_jaw_front

        # JAW POSITIONS
        # DICOM->BEAMnrc: Swap signs for y
        y_jaws = [-jaw_pos[1] / 10.0, -jaw_pos[0] / 10.0]

        # Jaws focus to a 3 mm^2 square at the target plane, not to a point.
        y_neg_front = (y_jaws[0] + 0.15) * (jaw_dict["y_z_jaw_front"] / 100.0) - 0.15
        y_neg_back = (y_jaws[0] + 0.15) * (jaw_dict["y_z_jaw_back"] / 100.0) - 0.15
        y_pos_front = (y_jaws[1] - 0.15) * (jaw_dict["y_z_jaw_front"] / 100.0) + 0.15
        y_pos_back = (y_jaws[1] - 0.15) * (jaw_dict["y_z_jaw_back"] / 100.0) + 0.15

        # Translate the jaws back to their position if they were projected to a point,
        # but keep the angle defined by the 3 mm projection.
        diff_neg = abs(0.15 * (jaw_dict["y_z_jaw_front"] + 0.5 * adjusted_y_jaw_width) / 100.0 - 0.15)
        diff_pos = abs(-0.15 * (jaw_dict["y_z_jaw_front"] + 0.5 * adjusted_y_jaw_width) / 100.0 + 0.15)

        jaw_dict["y_jaw_neg_front"] = y_neg_front + diff_neg
        jaw_dict["y_jaw_neg_back"] = y_neg_back + diff_neg
        jaw_dict["y_jaw_pos_front"] = y_pos_front - diff_pos
        jaw_dict["y_jaw_pos_back"] = y_pos_back - diff_pos

        return jaw_dict

    def _process_x_jaws(self, jaw_pos):
        """
        Convert from DICOM-specified x jaw positions to BEAMnrc. DICOM jaw
        positions are defined as the field projection at isocenter.

        On Varian linacs, the X jaws travel linearly with field size. However,
        they rotate such that the jaw face stays parallel to the beam
        divergence. Front of jaws stays at constant height.
        """
        # Calculate left x jaw back height
        jaw_dict = {}

        cm_jaws = [jaw_pos[0] / 10.0, jaw_pos[1] / 10.0]
        costheta = 100.0 / math.sqrt(100.0 * 100.0 + cm_jaws[0] * cm_jaws[0])
        left_jaw_back = self.x_z_jaw_front + self.x_jaw_width * costheta

        costheta = 100.0 / math.sqrt(100.0 * 100.0 + cm_jaws[1] * cm_jaws[1])
        right_jaw_back = self.x_z_jaw_front + self.x_jaw_width * costheta

        jaw_dict["x_z_jaw_front"] = self.x_z_jaw_front
        # Take the mean of the two back jaw positions
        jaw_dict["x_z_jaw_back"] = 0.5 * (left_jaw_back + right_jaw_back)
        adjusted_x_jaw_width = jaw_dict["x_z_jaw_back"] - self.x_z_jaw_front

        # Jaws focus to a 3 mm^2 square at the target plane, not to a point.
        x_neg_front = (cm_jaws[0] + 0.15) * (jaw_dict["x_z_jaw_front"] / 100.0) - 0.15
        x_neg_back = (cm_jaws[0] + 0.15) * (jaw_dict["x_z_jaw_back"] / 100.0) - 0.15
        x_pos_front = (cm_jaws[1] - 0.15) * (jaw_dict["x_z_jaw_front"] / 100.0) + 0.15
        x_pos_back = (cm_jaws[1] - 0.15) * (jaw_dict["x_z_jaw_back"] / 100.0) + 0.15

        # Translate the jaws back to their position if they were projected to a point,
        # but keep the angle defined by the 3 mm projection.
        diff_neg = abs(0.15 * (jaw_dict["x_z_jaw_front"] + 0.5 * adjusted_x_jaw_width) / 100.0 - 0.15)
        diff_pos = abs(-0.15 * (jaw_dict["x_z_jaw_front"] + 0.5 * adjusted_x_jaw_width) / 100.0 + 0.15)

        jaw_dict["x_jaw_neg_front"] = x_neg_front + diff_neg
        jaw_dict["x_jaw_neg_back"] = x_neg_back + diff_neg
        jaw_dict["x_jaw_pos_front"] = x_pos_front - diff_pos
        jaw_dict["x_jaw_pos_back"] = x_pos_back - diff_pos

        return jaw_dict

    @staticmethod
    def _make_jaws_file(params):
        """
        Create a BeamNRC jaw file for control points in a plan.

        :param params: Dict with processed control points including jaw positions
        """
        jaws_filename = params["name"] + ".jaws"
        num_cpts = len(params["cpts"])

        with open(jaws_filename, "w") as jaws_file:
            jaws_file.write("JAW file\n")
            jaws_file.write("{}\n".format(num_cpts))
            for cpt in params["cpts"]:
                jaws_file.write("{weight:.6}\n".format(weight=cpt["weight"]))
                jaws_file.write("{y_z_jaw_front:.5}, {y_z_jaw_back:.5}, {y_jaw_pos_front:.5}, {y_jaw_pos_back:.5}, {y_jaw_neg_front:.5}, {y_jaw_neg_back:.5}\n".format(**cpt))
                jaws_file.write("{x_z_jaw_front:.5}, {x_z_jaw_back:.5}, {x_jaw_pos_front:.5}, {x_jaw_pos_back:.5}, {x_jaw_neg_front:.5}, {x_jaw_neg_back:.5}\n".format(**cpt))

        return jaws_filename

    @staticmethod
    def _make_cutout_string(cutout):
        num_poly = 1
        num_holes = 0
        num_vertices = len(cutout)
        #print cutout
        cwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        os.chdir("../../../../utils/polypartition")

        #polygon must be defined in counter-clockwise order
        if sum([(p2[0] - p1[0])*(p2[1] - p1[1]) for p1, p2 in zip(cutout, cutout[1:])]) > 0:
            cutout = cutout[::-1]

        with open("cutout.txt", 'w') as outputf:
            outputf.write(str(num_poly) + '\n')
            outputf.write(str(num_vertices) + '\n')
            outputf.write(str(num_holes) + '\n')
            for point in cutout[:-1]:
                outputf.write(' '.join(["{:.3f}".format(point[0]/10.0), "{:.3f}".format(point[1]/10.0)]) + '\n')
            last_point = cutout[-1]
            outputf.write(' '.join(["{:.3f}".format(last_point[0]/10.0), "{:.3f}".format(last_point[1]/10.0)]) + '\n')

        subprocess.call(["./polypartition"])

        with open("partitioned_cutout.txt", 'r') as inputf:
            num_poly_output = int(inputf.readline().rstrip('\n'))
            vertex_counts = []
            poly_list = []

            for i in range(num_poly_output):
                num_vertices = int(inputf.readline().rstrip('\n'))
                vertex_counts.append(num_vertices)
                numholes = inputf.readline()
                vertex_list = []

                for j in range(num_vertices):
                    vertex = tuple(inputf.readline().rstrip('\n').split(' '))
                    vertex_list.append(vertex)
                    
                poly_list.append(vertex_list)

        RMAX='50.00'
        CM_TITLE = 'BLOCK'
        ZMIN='93.6'
        ZMAX='94.99'
        ZFOCUS='-1000000'
        XPMAX='10'
        YPMAX='10'
        XNMAX='-10'
        YNMAX='-10'
        ECUT='0.7'
        PCUT='0.01'
        DOSE_ZONE='0'
        IREGION_TO_BIT='1'
        MED_OPENING='AIR700ICRU'
        MED_BLOCK='CERROBEND700'
        with open("block_template.txt", 'w') as outputf:
            outputf.write("*********** start of CM BLOCK with identifier block ***********\n")
            outputf.write(RMAX+'\n')
            outputf.write(CM_TITLE+'\n')
            outputf.write(', '.join([ZMIN, ZMAX, ZFOCUS]) + ',\n')
            outputf.write(str(num_poly_output) + ',\n')
            for num in vertex_counts:
                outputf.write(str(num) + ',\n')
            for poly in poly_list:
                for vertex in poly:
                    outputf.write(vertex[0] + '\n' + vertex[1] + ',\n')
            outputf.write(', '.join([XPMAX, YPMAX, XNMAX, YNMAX]) + ',\n')
            outputf.write(', '.join([ECUT, PCUT, DOSE_ZONE, IREGION_TO_BIT]) + ',\n')
            outputf.write(', '.join([ECUT, PCUT, DOSE_ZONE, IREGION_TO_BIT]) + ',\n')
            outputf.write(MED_OPENING + '\n')
            outputf.write(', '.join([ECUT, PCUT, DOSE_ZONE, IREGION_TO_BIT]) + ',\n')
            outputf.write(MED_BLOCK + '\n')
        
        with open("block_template.txt", 'r') as inputf:
            cutout_string = inputf.read().rstrip('\n')
        
        os.chdir(cwd)
        return cutout_string
        
    def _write_beamnrc_input(self, params):
        """
        Write BEAMnrc input file from params dict.

        The template string is formatted twice. Once to include the applicator
        template, then once more to fill in all the parameters defined by
        the params dict.
        """
        beamnrc_filename = self.get_beamnrc_template(params)
        template_path = os.path.join(self.template_folder, beamnrc_filename)
        with open(template_path) as template_file:
            template_string = template_file.read()

       
        #applicator_field_string = params["applicator_type"][0]
        applicator_field_string = params["applicator_type"]
        applicator_path = os.path.join(self.template_folder, "applicator_" + applicator_field_string + ".app")
        with open(applicator_path) as applicator_template_file:
            applicator_string = applicator_template_file.read()

        if "cutout" in params:
            cutout = params["cutout"]
            cutout_string = self._make_cutout_string(cutout)
        else: #if no cutout is specified, then we use one of the standard cutouts
            cutout_path = os.path.join(self.template_folder, "cutout_" + applicator_field_string + ".cut")
            with open(cutout_path) as cutout_template_file:
                cutout_string = cutout_template_file.read()

        applicator_string += cutout_string
        
    
        
        template_string = template_string.format(applicator_template=applicator_string)

        beam_filename = params["name"] + "_beam.egsinp"
        with open(beam_filename, "w") as myfile:
            myfile.write(template_string.format(**params))

        return beam_filename

    def _write_dosxyznrc_input(self, params):
        """
        Write DOSXYZnrc input file from params dict.
        """
        cpt_strings = []
        cpt_string = "%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.6f"
        for cpt in params["cpts"]:
            f_string = cpt_string % (cpt["iso"][0], cpt["iso"][1], cpt["iso"][2],
                                     cpt["theta"], cpt["phi"], cpt["phicol"],
                                     cpt["dsource"], cpt["weight"])

            cpt_strings.append(f_string)

        full_string = "\n".join(cpt_strings)
        params["cpt_string"] = full_string
        params["num_cpts"] = len(params["cpts"])

        dosxyznrc_filename = self.get_dosxyznrc_template(params)
        template_path = os.path.join(self.template_folder, dosxyznrc_filename)
        template_file = open(template_path)
        template_string = template_file.read()
        template_file.close()

        dosxyznrc_filename = params["name"] + ".egsinp"
        with open(dosxyznrc_filename, "w") as myfile:
            myfile.write(template_string.format(**params))

        return dosxyznrc_filename
