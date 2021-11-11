"""
DVH calculation module.

Copyright Marc-Andre Renaud, 2017
"""
import pydicom as dicom

import numpy


class DVH(object):
    """
    PyTPS DVH calculator.

    Attributes:
    dose_coordinates (CoordinateSystem): coordinate system of dose.
    dose_path (string): path to dicom dose, for now.
    roi (Structure): roi to calculate DVH for.
    """

    @staticmethod
    def make_dvh_figure(rois, output="pdf"):
        import matplotlib
        matplotlib.use('svg')
        from matplotlib import pyplot
        linestyles = ['-', '--', '-.', ':', ' ', '']
        unique_doses = []

        for roi in rois:
            for plan_dvhs in roi["dvhs"]:
                roi_line = None
                for dose_dvh in plan_dvhs:
                    try:
                        dose_index = unique_doses.index(dose_dvh["name"])
                    except ValueError:
                        unique_doses.append(dose_dvh["name"])
                        dose_index = len(unique_doses) - 1

                    title = roi["roi_name"] + " - " + dose_dvh["name"]
                    x_range = numpy.arange(len(dose_dvh["dvh"])) * 0.1
                    dvh_data = numpy.array(dose_dvh["dvh"])
                    dvh_data = dvh_data * 100.0 / dvh_data[0]
                    if roi_line is None:
                        roi_line, = pyplot.plot(x_range, dvh_data, label=title, ls=linestyles[dose_index % len(linestyles)])
                    else:
                        pyplot.plot(x_range, dvh_data, label=title, color=roi_line.get_color(), ls=linestyles[dose_index % len(linestyles)])

        pyplot.xlabel("dose / Gy")
        pyplot.ylabel("volume / %")

        filename = "dvh." + output
        #pyplot.legend(frameon=False)
        pyplot.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)
        pyplot.gca().set_xlim(left=0)
        pyplot.gca().set_ylim(bottom=0)
        pyplot.savefig(filename, bbox_inches="tight")

        pyplot.clf()
        return filename

    @staticmethod
    def make_robust_dvh_figure(rois, output="pdf"):
        import matplotlib
        matplotlib.use('svg')
        from matplotlib import pyplot
        linestyles = ['-', '--', '-.', ':', ' ', '']

        unique_doses = []

        for roi in rois:
            for plan_dvhs in roi["dvhs"]:
                dose_dvhs = [numpy.array(dose_dvh["dvh"]) for dose_dvh in plan_dvhs]
                max_length = max([len(dvh) for dvh in dose_dvhs])

                padded_dvhs = [numpy.pad(dvh, (0, max_length - len(dvh)), 'constant') for dvh in dose_dvhs]
                normalized_dvhs = [padded_dvh * 100.0 / padded_dvh[0] for padded_dvh in padded_dvhs]

                dose_dvhs = numpy.array(normalized_dvhs)
                min_values = numpy.min(dose_dvhs, axis=0)
                max_values = numpy.max(dose_dvhs, axis=0)

                main_dose = next(dvh for dvh in plan_dvhs if dvh["main_dose"])
                main_dvh = numpy.array(main_dose["dvh"])
                main_dvh = numpy.pad(main_dvh, (0, max_length - len(main_dvh)), 'constant')
                main_dvh = main_dvh * 100.0 / main_dvh[0]

                try:
                    dose_index = unique_doses.index(main_dose["name"])
                except ValueError:
                    unique_doses.append(main_dose["name"])
                    dose_index = len(unique_doses) - 1


                x_range = numpy.arange(len(main_dvh)) * 0.1

                title = roi["roi_name"] + " - " + main_dose["ref_plan_name"]
                roi_line, = pyplot.plot(x_range, main_dvh, label=title, ls=linestyles[dose_index % len(linestyles)])
                pyplot.fill_between(x_range, min_values, max_values, facecolor=roi_line.get_color(), alpha=0.5)

        pyplot.xlabel("dose / Gy")
        pyplot.ylabel("volume / %")

        filename = "dvh_robust." + output
        #pyplot.legend(frameon=False)
        pyplot.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)
        pyplot.gca().set_xlim(left=0)
        pyplot.gca().set_ylim(bottom=0)
        pyplot.savefig(filename, bbox_inches="tight")

        pyplot.clf()
        return filename

    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

        if hasattr(self, "dose_path"):
            self.rtdose = dicom.read_file(self.dose_path, force=True)

    def compute_dvh(self, normalized=False):
        structure_mask = self.roi.get_mask(self.dose_coordinates)
        if len(structure_mask) == 0:
            return []

        # Dose values in cGy
        dose_values = (self.rtdose.pixel_array * (self.rtdose.DoseGridScaling * 10))
        num_bins = int(dose_values.max()) + 1
        roi_dose_values = dose_values[structure_mask]

        # Volume in cubic centimeters
        unit_volume = (self.dose_coordinates.spacing[0]
                       * self.dose_coordinates.spacing[1]
                       * self.dose_coordinates.spacing[2] / 1000.0)

        histogram, bin_edges = numpy.histogram(roi_dose_values,
                                               bins=num_bins,
                                               range=(0, num_bins))

        volume_histogram = histogram * unit_volume
        volume_histogram = numpy.append(numpy.trim_zeros(volume_histogram, trim="b"), 0)

        cum_dvh = numpy.cumsum(volume_histogram[::-1])[::-1]

        if normalized:
            cum_dvh = cum_dvh * 100.0 / cum_dvh[0]

        return cum_dvh.tolist()

    def find_vol_dose(self, dvh, vol):
        dose_bin = 0
        for i in range(len(dvh)):
            if vol > dvh[i]:
                dose_bin = i - 1
                break

        dose1 = dose_bin * 0.1  # Turn into Gy
        interp = dose1 + ((vol - dvh[dose_bin]) / (dvh[dose_bin + 1] - dvh[dose_bin])) * 0.1

        return interp

    def get_scaling(self, dose, volume):
        abs_dvh = numpy.array(self.compute_dvh())
        rel_dvh = (abs_dvh / abs_dvh.max()) * 100.0

        scaling = dose / self.find_vol_dose(rel_dvh, volume)
        return scaling
