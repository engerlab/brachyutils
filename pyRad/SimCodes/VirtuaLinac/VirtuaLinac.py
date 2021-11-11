import os
import numpy
import importlib
import errno
from pyRad.utils import egsdose_to_dicom
from xml.etree import ElementTree


class VirtuaLinac(object):
    def __init__(self, attrs=None):
        if attrs:
            for k, v in attrs.items():
                setattr(self, k, v)

    def submit_sim(self, simulation):
        self._get_phantom_center(simulation)
        xml_file = self._make_xml_file(simulation)
        return

    def _get_phantom_center(self, simulation):
        topleft = numpy.array(simulation.phantom_parameters["topleft"])
        num_voxels = numpy.array(simulation.phantom_parameters["num_voxels"])
        spacing = numpy.array(simulation.phantom_parameters["spacing"])

        self.phantom_center = 0.5 * ((topleft - 0.5 * spacing) + num_voxels * spacing)

        return self.phantom_center

    def _make_xml_file(self, simulation):
        top = ElementTree.Element("VarianResearchBeam")
        top.set("SchemaVersion", "1.0")

        beam_doms = [self._make_xml_beam(beam) for beam in simulation.beams]
        top.extend(beam_doms)

        xml_filename = simulation.name + ".xml"

        tree = ElementTree.ElementTree(top)
        tree.write(xml_filename)

        return xml_filename

    def _make_xml_beam(self, beam):
        beam_dom = ElementTree.Element("SetBeam")
        beam_id = ElementTree.SubElement(beam_dom, "Id")
        beam_id.text = str(beam["index"]).zfill(4)

        mlc_model = ElementTree.SubElement(beam_dom, "MLCModel")
        mlc_model.text = "NDS120HD"

        # No idea what this tag does
        accs = ElementTree.SubElement(beam_dom, "Accs")

        control_points = ElementTree.SubElement(beam_dom, "ControlPoints")
        # First control point has more info
        cp_doms = [self._make_xml_cpt(beam["cpts"][0], first=True)]
        cp_doms += [self._make_xml_cpt(cpt) for cpt in beam["cpts"][1:]]

        control_points.extend(cp_doms)

        return beam_dom

    def _make_xml_cpt(self, cpt, first=False):
        cpt_dom = ElementTree.Element("Cp")

        if first:
            sub = ElementTree.SubElement(cpt_dom, "SubBeam")
            seq = ElementTree.SubElement(sub, "Seq")
            seq.text = str(0)
            name = ElementTree.SubElement(sub, "Name")
            name.text = self.sim_params["name"] + "_%i" % cpt.arc_index

        mu = ElementTree.SubElement(cpt_dom, "Mu")
        mu.text = str(cpt.weight)

        iso = numpy.array(cpt.iso)

        # Offset between iso and phantom... we want to translate in the opposite
        # direction to put the iso at (0, 0, 0) in VirtuaLinac
        phantom_offset = -(iso - self.phantom_center)
        couchlat = ElementTree.subElement(cpt_dom, "CouchLat")
        couchlat.text = str(phantom_offset[0])

        couchlng = ElementTree.subElement(cpt_dom, "CouchLng")
        couchlng.text = str(phantom_offset[2])

        couchvrt = ElementTree.subElement(cpt_dom, "CouchVrt")
        couchvrt.text = str(phantom_offset[1])

        if hasattr(cpt, "energy"):
            energy = ElementTree.SubElement(cpt_dom, "Energy")
            energy.text = str(cpt.energy)

        if hasattr(cpt, "dose_rate"):
            drate = ElementTree.SubElement(cpt_dom, "DRate")
            drate.text = str(cpt.dose_rate)

        if hasattr(cpt, "gantry_angle"):
            gantry_angle = ElementTree.SubElement(cpt_dom, "GantryRtn")
            gantry_angle.text = str(cpt.gantry_angle)

        if hasattr(cpt, "col_angle"):
            col_angle = ElementTree.SubElement(cpt_dom, "CollRtn")
            col_angle.text = str(cpt.col_angle)

        if hasattr(cpt, "couch_angle"):
            couch_angle = ElementTree.SubElement(cpt_dom, "CouchRtn")
            couch_angle.text = str(cpt.couch_angle)

        if hasattr(cpt, "x_jaw"):
            x_jaw_neg = ElementTree.SubElement(cpt_dom, "X2")
            x_jaw_pos = ElementTree.SubElement(cpt_dom, "X1")

            x_jaw_neg.text = str(cpt.x_jaw[0])
            x_jaw_pos.text = str(cpt.x_jaw[1])

        if hasattr(cpt, "y_jaw"):
            y_jaw_neg = ElementTree.SubElement(cpt_dom, "Y2")
            y_jaw_pos = ElementTree.SubElement(cpt_dom, "Y1")

            y_jaw_neg.text = str(cpt.y_jaw[0])
            y_jaw_pos.text = str(cpt.y_jaw[1])

        if hasattr(cpt, "apertures"):
            mlc = ElementTree.SubElement(cpt_dom, "Mlc")
            mlc_id = ElementTree.SubElement(mlc, "ID")
            mlc_id.text = str(1)

            neg_positions = [leaf[0] for leaf in cpt.apertures]
            pos_positions = [leaf[1] for leaf in cpt.apertures]

            b_leaves = ElementTree.SubElement(mlc, "B")
            b_leaves.text = " ".join([str(x) for x in neg_positions])

            a_leaves = ElementTree.SubElement(mlc, "A")
            a_leaves.text = " ".join([str(x) for x in pos_positions])

        return cpt_dom

    def _start_instance(self):
        pass

    def _start_spot_instance(self):
        pass
