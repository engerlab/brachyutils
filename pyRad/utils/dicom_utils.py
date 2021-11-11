import numpy
import random
import pydicom as dicom
import os
from math import cos, sin, acos, atan2, radians, sqrt, pi as PI
from numpy import clip, copysign

from pyRad.utils import SimDose

def create_uid():
    # UID prefixed granted by Medical Connections ltd. to
    # Marc-Andre Renaud (moi@marcandre.io)
    UID_PREFIX = '1.2.826.0.1.3680043.9.2843'
    last_four = [str(random.randint(1, 99999999)) for i in range(4)]
    return ".".join([UID_PREFIX] + last_four)


def dicom_to_spherical(gantry_angle, couch_angle, col_angle, orient="HFS"):
    if orient is "FFS":
        couch_angle = numpy.mod(couch_angle + 180.0, 360.0)
    gamma = radians(gantry_angle)
    col = radians(col_angle)
    rho = radians(couch_angle)
    # distort Couch and Gantry angles slightly to avoid
    # very special cases
    if couch_angle in (90.0, 270.0) and gantry_angle in (90.0, 270.0):
        rho = rho * 0.999999
        gamma = gamma * 0.999999

    sgsr = numpy.sin(gamma)*numpy.sin(rho)
    sgcr = numpy.sin(gamma)*numpy.cos(rho)
    theta = numpy.arccos(-sgsr)
    phi = numpy.arctan2(-numpy.cos(gamma), sgcr)
    CouchAngle2CollPlane = numpy.arctan2(-numpy.sin(rho)*numpy.cos(gamma), numpy.cos(rho))
    phicol = (col - numpy.pi / 2) + CouchAngle2CollPlane
    phicol = numpy.pi - phicol

    return (theta, numpy.mod(phi, 2 * numpy.pi), numpy.mod(phicol, 2 * numpy.pi))


def dicom_to_spherical_2(gantry_angle, couch_angle, col_angle=None):
    gamma = radians(gantry_angle)
    rho = radians(couch_angle)

    phi = atan2(-cos(gamma), sin(gamma)*cos(rho))

    costheta = clip(sin(gamma) * sin(rho), -1.0, 1.0)

    theta = acos(costheta)
    phicol = None
    if col_angle is not None:
        col = radians(col_angle)
        prefactor = (cos(col) * cos(gamma) * cos(rho) * sin(phi)
                     - sin(col) * sin(rho) * sin(phi)
                     - cos(col) * sin(gamma) * cos(phi))

        prefactor = clip(prefactor, -1.0, 1.0)

        phicol = acos(prefactor)

        direction = (cos(col) * sin(rho) * sin(phi)
                     + sin(col) * (cos(gamma) * cos(rho) * sin(phi) - sin(gamma) * cos(phi)))

        phicol = copysign(abs(phicol), -direction)

    if phi < 0:
        phi = 2.0 * PI + phi

    return (theta, phi, phicol)


def load_egsdose(filename):
    with open(filename, "rb") as newfile:
        num_voxels = [int(i) for i in newfile.readline().split()]
        x_pos = numpy.array(newfile.readline().split(), dtype=numpy.float)
        y_pos = numpy.array(newfile.readline().split(), dtype=numpy.float)
        z_pos = numpy.array(newfile.readline().split(), dtype=numpy.float)

        x_spacing = (x_pos[1] - x_pos[0])
        y_spacing = (y_pos[1] - y_pos[0])
        z_spacing = (z_pos[1] - z_pos[0])

        huge_dose_array = numpy.array(newfile.readline().strip().split(), dtype=numpy.float)
        bench_dose = numpy.reshape(huge_dose_array, (num_voxels[2], num_voxels[1], num_voxels[0]))

        huge_uncert_array = numpy.array(newfile.readline().strip().split(), dtype=numpy.float)
        bench_uncert = numpy.reshape(huge_uncert_array, (num_voxels[2], num_voxels[1], num_voxels[0]))

        dose_dict = {}
        dose_dict["grid"] = bench_dose
        dose_dict["uncert"] = bench_uncert
        dose_dict["num_voxels"] = num_voxels
        dose_dict["vox_size"] = [x_spacing, y_spacing, z_spacing]
        dose_dict["topleft"] = [x_pos[0], y_pos[0], z_pos[0]]
        dose_dict["x_pos"] = x_pos
        dose_dict["y_pos"] = y_pos
        dose_dict["z_pos"] = z_pos

    return dose_dict

def simdose_to_dicom(sim_dose,
                     patient_uid,
                     plan_uid,
                     orient,
                     norm=None,
                     norm_point=None):

    orient = [int(i) for i in orient]
    spacing = numpy.array([
        sim_dose.x_voxels[1] - sim_dose.x_voxels[0],
        sim_dose.y_voxels[1] - sim_dose.y_voxels[0],
        sim_dose.z_voxels[1] - sim_dose.z_voxels[0]
    ])

    phantom_img_pos = numpy.array([
        sim_dose.x_voxels[0] + 0.5 * spacing[0],
        sim_dose.y_voxels[0] + 0.5 * spacing[1],
        sim_dose.z_voxels[0] + 0.5 * spacing[2]
    ]) * 10.0

    dicom_img_pos = numpy.array([
        sim_dose.x_voxels[::orient[0]][0] + 0.5 * spacing[0] * orient[0],
        sim_dose.y_voxels[::orient[4]][0] + 0.5 * spacing[1] * orient[4],
        sim_dose.z_voxels[::orient[0]][0] + 0.5 * spacing[2] * orient[0]  # orient[0] Not a typo
    ]) * 10.0

    spacing *= 10.0

    grid_frame_offset = numpy.arange(sim_dose.doses.shape[0]) * spacing[2]

    dose_grid = sim_dose.doses
    uncert_grid = sim_dose.uncerts

    if norm_point is not None and norm is not None:
        norm_vox = numpy.array((norm_point - phantom_img_pos) / spacing, dtype=int)
        norm = norm / dose_grid[norm_vox[2]][norm_vox[1]][norm_vox[0]]

    if norm:
        dose_grid = dose_grid * norm
        uncert_grid = uncert_grid * norm

    dose_grid_scaling = dose_grid.max() / 65530
    dose_grid = dose_grid / dose_grid_scaling
    dose_grid = numpy.array(dose_grid, dtype=numpy.uint16)[::orient[0], ::orient[4], ::orient[0]]

    uncert_grid_scaling = uncert_grid.max() / 65530
    uncert_grid = uncert_grid / uncert_grid_scaling
    uncert_grid = numpy.array(uncert_grid, dtype=numpy.uint16)[::orient[0], ::orient[4], ::orient[0]]

    template_dir = os.path.dirname(__file__)
    template_filename = os.path.join(template_dir, "dose_template.dcm")
    dose_template = dicom.read_file(template_filename)
    dose_template.InstitutionalDepartmentName = "Radify"
    dose_template.SeriesDescription = "MC Dose exported from Radify"
    dose_template.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = plan_uid
    dose_template.StudyInstanceUID = patient_uid
    dose_template.GridFrameOffsetVector = grid_frame_offset.tolist()
    dose_template.ImagePositionPatient = dicom_img_pos.tolist()
    dose_template.ImageOrientationPatient = orient
    dose_template.PixelSpacing = [spacing[0], spacing[1]]
    dose_template.Rows = dose_grid.shape[1]
    dose_template.Columns = dose_grid.shape[2]
    dose_template.NumberOfFrames = dose_grid.shape[0]
    dose_template.PatientID = "Radify %s" % (patient_uid)
    dose_template.DoseGridScaling = dose_grid_scaling
    dose_template.PixelData = dose_grid.tostring()
    dose_template.SOPInstanceUID = create_uid()

    error_template = dicom.read_file(template_filename)
    error_template.InstitutionalDepartmentName = "Radify"
    error_template.SeriesDescription = "MC Errors exported from Radify"
    error_template.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = plan_uid
    error_template.StudyInstanceUID = patient_uid
    error_template.GridFrameOffsetVector = grid_frame_offset.tolist()
    error_template.ImagePositionPatient = dicom_img_pos.tolist()
    error_template.ImageOrientationPatient = orient
    error_template.PixelSpacing = [spacing[0], spacing[1]]
    error_template.Rows = dose_grid.shape[1]
    error_template.Columns = dose_grid.shape[2]
    error_template.NumberOfFrames = dose_grid.shape[0]
    error_template.PatientID = "Radify %s" % (patient_uid)
    error_template.DoseGridScaling = uncert_grid_scaling
    error_template.PixelData = uncert_grid.tostring()
    error_template.SOPInstanceUID = create_uid()

    return (dose_template, error_template)


def egsdose_to_dicom(egsdose_file,
                     patient_uid,
                     plan_uid,
                     orient,
                     remove_original=False,
                     norm=None,
                     norm_point=None):
    with open(egsdose_file) as dose:
        orient = [int(i) for i in orient]
        voxels = dose.readline().strip().split()
        num_voxels = [int(vox) for vox in voxels]
        huge_array = dose.read().replace("\n", " ").split()
        x_voxels = numpy.array(huge_array[0:num_voxels[0] + 1], dtype=numpy.float)[::orient[0]]
        y_voxels = numpy.array(huge_array[(num_voxels[0] + 1):(num_voxels[0] + num_voxels[1] + 2)], dtype=numpy.float)[::orient[4]]
        z_voxels = numpy.array(huge_array[(num_voxels[0] + num_voxels[1] + 2):(num_voxels[0] + num_voxels[1] + num_voxels[2] + 3)], dtype=numpy.float)

        start_offset = num_voxels[0] + num_voxels[1] + num_voxels[2] + 3

        dose_grid = numpy.zeros((num_voxels[2], num_voxels[1], num_voxels[0]))
        error_grid = numpy.zeros((num_voxels[2], num_voxels[1], num_voxels[0]))
        for k in range(dose_grid.shape[0]):
            for i in range(dose_grid.shape[1]):
                offset = i * dose_grid.shape[2] + k * dose_grid.shape[1] * dose_grid.shape[2] + start_offset
                dose_grid[k][i] = numpy.array(huge_array[offset:offset + dose_grid.shape[2]], dtype=numpy.float)

        try:
            start_offset += dose_grid.shape[2] * dose_grid.shape[1] * dose_grid.shape[0]

            for k in range(dose_grid.shape[0]):
                for i in range(dose_grid.shape[1]):
                    offset = i * dose_grid.shape[2] + k * dose_grid.shape[1] * dose_grid.shape[2] + start_offset
                    error_grid[k][i] = numpy.array(huge_array[offset:offset + dose_grid.shape[2]], dtype=numpy.float)
        except ValueError:
            # Nonstandard 3ddose file format.
            pass

        # DICOM uses millimeters.
        x_spacing = round(abs((x_voxels[1] - x_voxels[0]) * 10), 2)
        y_spacing = round(abs((y_voxels[1] - y_voxels[0]) * 10), 2)
        pixel_spacing = [x_spacing, y_spacing]
        slice_thick = round(abs((z_voxels[1] - z_voxels[0]) * 10), 2)
        # Want the middle of the topleft pixel, by DICOM convention
        topleft_x = x_voxels[0] * 10 + 0.5 * x_spacing * orient[0]
        topleft_y = y_voxels[0] * 10 + 0.5 * y_spacing * orient[4]
        first_z = z_voxels[0] * 10 + 0.5 * slice_thick
        last_z = z_voxels[-1] * 10 - 0.5 * slice_thick
        if orient[0] < 0:
            image_position_patient = [topleft_x, topleft_y, last_z]
        else:
            image_position_patient = [topleft_x, topleft_y, first_z]

        image_position_patient = [round(x, 3) for x in image_position_patient]
        image_orientation_patient = orient

        grid_frame_offset = numpy.array(range(dose_grid.shape[0])) * slice_thick

    if norm_point is not None and norm is not None:
        topleft_array = numpy.array([topleft_x, topleft_y, first_z])
        spacing_array = numpy.array([x_spacing, y_spacing, slice_thick])
        norm_voxel = [int(x) for x in (norm_point - topleft_array) / spacing_array]
        norm = norm / dose_grid[norm_voxel[2]][norm_voxel[1]][norm_voxel[0]]

    if norm:
        dose_grid = dose_grid * norm
        error_grid = error_grid * norm

    dose_grid_scaling = dose_grid.max() / 65530
    dose_grid = dose_grid / dose_grid_scaling
    dose_grid = numpy.array(dose_grid, dtype=numpy.uint16)[::orient[0], ::orient[4], ::orient[0]]

    error_grid_scaling = error_grid.max() / 65530
    error_grid = error_grid / error_grid_scaling
    error_grid = numpy.array(error_grid, dtype=numpy.uint16)[::orient[0], ::orient[4], ::orient[0]]

    template_dir = os.path.dirname(__file__)
    template_filename = os.path.join(template_dir, "dose_template.dcm")
    dose_template = dicom.read_file(template_filename)
    dose_template.InstitutionalDepartmentName = "WebTPS"
    dose_template.SeriesDescription = "MC Dose exported from WebTPS"
    dose_template.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = plan_uid
    dose_template.StudyInstanceUID = patient_uid
    dose_template.GridFrameOffsetVector = grid_frame_offset.tolist()
    dose_template.ImagePositionPatient = image_position_patient
    dose_template.ImageOrientationPatient = image_orientation_patient
    dose_template.PixelSpacing = pixel_spacing
    dose_template.Rows = dose_grid.shape[1]
    dose_template.Columns = dose_grid.shape[2]
    dose_template.NumberOfFrames = dose_grid.shape[0]
    dose_template.PatientID = "WebTPS %s" % (patient_uid)
    dose_template.DoseGridScaling = dose_grid_scaling
    dose_template.PixelData = dose_grid.tostring()
    dose_template.SOPInstanceUID = create_uid()

    error_template = dicom.read_file(template_filename)
    error_template.InstitutionalDepartmentName = "WebTPS"
    error_template.SeriesDescription = "MC Errors exported from WebTPS"
    error_template.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = plan_uid
    error_template.StudyInstanceUID = patient_uid
    error_template.GridFrameOffsetVector = grid_frame_offset.tolist()
    error_template.ImagePositionPatient = image_position_patient
    error_template.PixelSpacing = pixel_spacing
    error_template.Rows = dose_grid.shape[1]
    error_template.Columns = dose_grid.shape[2]
    error_template.NumberOfFrames = dose_grid.shape[0]
    error_template.PatientID = "WebTPS %s" % (patient_uid)
    error_template.DoseGridScaling = error_grid_scaling
    error_template.PixelData = error_grid.tostring()
    error_template.SOPInstanceUID = create_uid()

    if remove_original:
        os.remove(egsdose_file)

    return (dose_template, error_template)
