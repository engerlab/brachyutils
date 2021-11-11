import numpy
import pydicom as dicom
from scipy.interpolate import RegularGridInterpolator


class DoseStats(object):
    """
        PyTPS DoseStats calculator

        Attributes:
        dose_coordinates (CoordinateSystem): coordinate system of dose.
        (Optional) dose_path (string): path to dicom dose, for now.
        (Optional) dose_name (string): name of dose
        (Optional) ref_plan_uid (string): UID of referenced plan.
    """

    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

        if hasattr(self, "dose_path"):
            self.rtdose = dicom.read_file(self.dose_path, force=True)

    def compute_dvh(self, roi):
        structure_mask = roi.get_mask(self.dose_coordinates)
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

    def get_scaling(self, dose, volume, roi):
        abs_dvh = numpy.array(self.compute_dvh(roi))
        rel_dvh = (abs_dvh / abs_dvh.max()) * 100.0

        scaling = dose / self.find_vol_dose(rel_dvh, volume)
        return scaling

    def roi_median_dose(self, roi):
        dose_values = (self.rtdose.pixel_array * self.rtdose.DoseGridScaling)
        structure_mask = roi.get_mask(self.dose_coordinates)
        roi_dose_values = dose_values[structure_mask]
        return numpy.median(roi_dose_values)

    def dose_at_point(self, point):
        dose_values = self.rtdose.pixel_array * self.rtdose.DoseGridScaling
        vox = self.dose_coordinates.vox_from_point(point)

        return dose_values[vox[2]][vox[1]][vox[0]]

    def renormalize(self, dose, volume, roi):
        scaling = self.get_scaling(dose, volume, roi)
        self.rtdose.DoseGridScaling *= scaling
        self.rtdose.save_as(self.dose_path)

        return scaling

    def profile(self, start_point, end_point, output="eps"):
        start_point = numpy.array(start_point)
        end_point = numpy.array(end_point)

        # 1 mm resolution profiles
        resolution = 1.0
        profile_length = numpy.linalg.norm(end_point - start_point)
        direction = (end_point - start_point) / profile_length
        num_points = int(profile_length / resolution) + 1

        direction = direction[::-1]
        start_point = start_point[::-1]
        end_point = end_point[::-1]

        sampled_points = [direction * (i * resolution) + start_point for i in range(num_points)]

        dose_values = self.rtdose.pixel_array * self.rtdose.DoseGridScaling

        x_values = self.rtdose.ImagePositionPatient[0] + numpy.arange(self.rtdose.Columns) * self.rtdose.PixelSpacing[0] * self.rtdose.ImageOrientationPatient[0]
        y_values = self.rtdose.ImagePositionPatient[1] + numpy.arange(self.rtdose.Rows) * self.rtdose.PixelSpacing[1] * self.rtdose.ImageOrientationPatient[4]
        z_values = self.rtdose.ImagePositionPatient[2] + numpy.array(self.rtdose.GridFrameOffsetVector) * self.rtdose.ImageOrientationPatient[0] * self.rtdose.ImageOrientationPatient[4]

        if x_values[1] < x_values[0]:
            x_values = x_values[::-1]
            dose_values = dose_values[:, :, ::-1]

        if y_values[1] < y_values[0]:
            y_values = y_values[::-1]
            dose_values = dose_values[:, ::-1, :]

        if z_values[1] < z_values[0]:
            z_values = z_values[::-1]
            dose_values = dose_values[::-1, :, :]

        interpolator = RegularGridInterpolator((z_values, y_values, x_values), dose_values, bounds_error=False, fill_value=0.0)
        profile = interpolator(sampled_points)

        x_values = numpy.arange(num_points) * resolution / 10.0
        return (x_values, profile)

    @staticmethod
    def make_profile_figure(doses, start_point, end_point, output="pdf"):
        import matplotlib
        matplotlib.use('svg')
        from matplotlib import pyplot
        linestyles = ['-', '--', '-.', ':', ' ', '']
        unique_plans = []
        modalities = {}

        for dose in doses:
            x_values, profile = dose.profile(start_point, end_point)

            try:
                plan_index = unique_plans.index(dose.ref_plan_uid)
            except ValueError:
                unique_plans.append(dose.ref_plan_uid)
                plan_index = len(unique_plans) - 1

            try:
                modality = dose.dose_name.split("_")[-1]

                if modality in modalities:
                    pyplot.plot(x_values, profile, label=dose.dose_name, color=modalities[modality], ls=linestyles[plan_index % len(linestyles)])
                else:
                    plot_line, = pyplot.plot(x_values, profile, label=dose.dose_name, ls=linestyles[plan_index % len(linestyles)])
                    modalities[modality] = plot_line.get_color()

            except ValueError:
                pyplot.plot(x_values, profile, label=dose.dose_name, ls=linestyles[plan_index % len(linestyles)])

        pyplot.xlabel("distance / cm")
        pyplot.ylabel("dose / Gy")
        filename = "profile." + output
        pyplot.legend(bbox_to_anchor=(1.04,1), loc="upper left", frameon=False)
        pyplot.savefig(filename, bbox_inches="tight")

        pyplot.clf()

        return filename

