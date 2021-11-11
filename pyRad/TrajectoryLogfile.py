import struct
import numpy


class TrajectoryLogfile(object):
    axis_enum = {
        0: 'col_angle',
        1: 'gantry_angle',
        2: 'jaw_y1',
        3: 'jaw_y2',
        4: 'jaw_x1',
        5: 'jaw_x2',
        6: 'Couch Vrt',
        7: 'Couch Lng',
        8: 'Couch Lat',
        9: 'Couch Rtn',
        10: 'Couch Pit',
        11: 'Couch Rol',
        40: 'MU',
        41: 'Beam Hold',
        42: 'cpt',
        50: 'MLC',
        60: 'TargetPosition',
        61: 'TrackingTarget',
        62: 'TrackingBase',
        63: 'TrackingPhase',
        64: 'TrackingConformityIndex'
    }

    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def parse_logfile(self):
        with open(self.log_filename, 'rb') as logfile:
            header = self._read_header(logfile)

            subbeam_list = []
            for i in range(header["numberOfSubbeams"]):
                subbeam = self._read_subbeam(logfile)
                subbeam_list.append(subbeam)

            axes = self._read_axes(logfile, header)

            parsed_logfile = {
                "header": header,
                "axes": axes,
                "subbeams": subbeam_list
            }

            self.parsed_logfile = parsed_logfile

            return self.parsed_logfile

    def _read_header(self, logfile):
        header = {}

        header['signature'] = logfile.read(16).split('\0', 1)[0]
        header['version'] = logfile.read(16).split('\0', 1)[0]
        header['headerSize'] = struct.unpack('<i', logfile.read(4))[0]
        header['samplingInterval'] = struct.unpack('<i', logfile.read(4))[0]
        header['numAxesSampled'] = struct.unpack('<i', logfile.read(4))[0]

        axesOrder = {}
        for i in range(header['numAxesSampled']):
            currentAxis = struct.unpack('<i', logfile.read(4))[0]
            header[self.axis_enum[currentAxis]] = currentAxis
            axesOrder[i] = self.axis_enum[currentAxis]

        header['axesOrder'] = axesOrder

        samplePerAxis = numpy.zeros(header['numAxesSampled'], dtype=int)

        axesIndex = {}
        for i in range(header['numAxesSampled']):
            samplePerAxis[i] = struct.unpack('<i', logfile.read(4))[0]
            axesIndex[axesOrder[i]] = sum(samplePerAxis) - samplePerAxis[i]

        header['axesIndex'] = axesIndex
        header['samplePerAxis'] = samplePerAxis
        header['axisScale'] = 'Machine Scale' if(struct.unpack('<i', logfile.read(4))[0] == 1) else 'Modified IEC'
        header['numberOfSubbeams'] = struct.unpack('<i', logfile.read(4))[0]
        header['truncated'] = True if(struct.unpack('<i', logfile.read(4))[0] == 1) else False
        header['numberOfSnapShots'] = struct.unpack('<i', logfile.read(4))[0]
        header['MLCModel'] = 'NDS 120' if(struct.unpack('<i', logfile.read(4))[0] == 2) else 'NDS 120 HD'

        logfile.seek(1024)

        return header

    def _read_subbeam(self, logfile):
        subbeam = {}

        subbeam['cpt'] = struct.unpack('<i', logfile.read(4))[0]
        subbeam['mu'] = struct.unpack('<f', logfile.read(4))[0]
        subbeam['radTime'] = struct.unpack('<f', logfile.read(4))[0]
        subbeam['seq'] = struct.unpack('<i', logfile.read(4))[0]
        subbeam['nameOfTheSubbeam'] = logfile.read(512).split('\0', 1)[0]
        subbeam['reserved'] = logfile.read(32).split('\0', 1)[0]

        return subbeam

    def _read_axes(self, logfile, header):
        axes = {}
        axesLength = sum(header['samplePerAxis'])

        expectedMatrix = numpy.zeros((header['numberOfSnapShots'], axesLength), dtype=float)
        actualMatrix = numpy.zeros((header['numberOfSnapShots'], axesLength), dtype=float)

        for i in range(header['numberOfSnapShots']):
            for j in range(axesLength):
                expectedMatrix[i, j] = struct.unpack('<f', logfile.read(4))[0]
                actualMatrix[i, j] = struct.unpack('<f', logfile.read(4))[0]

        axes["expected"] = expectedMatrix
        axes["actual"] = actualMatrix
        axes["indices"] = header["axesIndex"]

        return axes

