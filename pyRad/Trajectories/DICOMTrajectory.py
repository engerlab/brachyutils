class DICOMTrajectory(object):
    """
        Attributes:
        d_source (float): Distance from radiation source to isocenter.
        d_coll (float): Distance between radiation source and MLC.
        plan (Plan obj): Plan object
    """

    defaults = {
        "iso_col_size": 5,
        "iso_row_size": 5,
        "d_source": 1000.0,
        "sad": 1000.0,
        "d_coll": 515.785,  # temporary, from CL21EX beam model
    }

    def __init__(self, attrs):
        for key in self.defaults:
            setattr(self, key, self.defaults[key])

        for k, v in attrs.items():
            setattr(self, k, v)

    def create_control_points(self, cpt_dicts):
        cpt_list = []
        beams = self.plan.parse_plan()["beams"]
        cpt_index = 0
        for beam in beams:
            previous_gantry = None
            previous_couch = None
            previous_col = None

            for beam_cpt in beam["cpts"]:
                if (beam_cpt["gantry_angle"] != previous_gantry or
                   beam_cpt["couch_angle"] != previous_couch or
                   beam_cpt["col_angle"] != previous_col):

                    cpt = beam_cpt.copy()
                    cpt["ptv"] = self.ptv.roi_num
                    cpt["iso_row_size"] = float(self.iso_row_size)
                    cpt["iso_col_size"] = float(self.iso_col_size)
                    cpt["d_coll"] = self.d_coll
                    cpt["d_source"] = cpt["sad"]
                    cpt["index"] = cpt_index
                    cpt["energy"] = int(cpt["energy"])
                    cpt_list.append(cpt)
                    cpt_index += 1
                    previous_gantry = cpt["gantry_angle"]
                    previous_couch = cpt["couch_angle"]
                    previous_col = cpt["col_angle"]

        self.control_points = cpt_list

        return cpt_list
