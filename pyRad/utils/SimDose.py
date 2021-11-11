"""Dose module to load and represent voxelized Monte Carlo dose data"""
import struct
import numpy as np
import os

class SimDose(object):
    """Voxelized Monte Carlo dose data"""
    def __init__(self, num_voxels, x_voxels, y_voxels, z_voxels, doses, uncerts, name=None):
        self.num_voxels = num_voxels
        self.x_voxels = x_voxels
        self.y_voxels = y_voxels
        self.z_voxels = z_voxels
        self.doses = doses
        self.uncerts = uncerts
        self.name = name

    def print_dose_statistics(self):
        print("Number of nonzero dose voxels: {}".format(np.count_nonzero(self.doses)))

        avg_dose = np.average(self.doses[np.nonzero(self.doses)])
        print("Average dose of nonzero dose voxels: {}".format(avg_dose))

        avg_uncert = np.average(self.uncerts[np.nonzero(self.doses)])
        print("Average uncertainty of nonzero dose voxels: {}".format(avg_uncert))

    def get_voxel_centers(self):
        """
        Return the position of the center of every voxel. Useful for interpolation.
        """
        x_centers = self.x_voxels[:-1] + 0.5 * (self.x_voxels[1:] - self.x_voxels[:-1])
        y_centers = self.y_voxels[:-1] + 0.5 * (self.y_voxels[1:] - self.y_voxels[:-1])
        z_centers = self.z_voxels[:-1] + 0.5 * (self.z_voxels[1:] - self.z_voxels[:-1])

        return (x_centers, y_centers, z_centers)

    @classmethod
    def from_file(cls, path):
        extension = os.path.splitext(path)[1]
        if extension == ".3ddose":
            return cls.from_3ddose(path)
        elif extension == ".bindos":
            return cls.from_bindos(path)
        elif extension == ".randydos":
            return cls.from_randydos(path)
        else:
            raise Exception("Dose file type not recognised")

    @classmethod
    def from_3ddose(cls, path):
        "Create Dose instance from 3ddose file"
        print("Loading {}".format(path))

        with open(path, "r") as dose_file:
            num_voxels = np.array([int(i) for i in dose_file.readline().split()], dtype=int)
            x_pos = np.array(dose_file.readline().split(), dtype=np.float)
            y_pos = np.array(dose_file.readline().split(), dtype=np.float)
            z_pos = np.array(dose_file.readline().split(), dtype=np.float)

            dose_array = np.array(dose_file.readline().strip().split(), dtype=np.float)
            doses = np.reshape(dose_array, (num_voxels[2], num_voxels[1], num_voxels[0]))

            try:
                uncert_array = np.array(dose_file.readline().strip().split(), dtype=np.float)
                uncerts = np.reshape(uncert_array, (num_voxels[2], num_voxels[1], num_voxels[0]))
            except ValueError:
                uncerts = np.zeros((num_voxels[2], num_voxels[1], num_voxels[0]))

        return cls(num_voxels, x_pos, y_pos, z_pos, doses, uncerts, os.path.basename(path))

    @classmethod
    def from_bindos(cls, path):
        "Create Dose instance from bindos file"
        print("Loading {}".format(path))

        with open(path, "rb") as binfile:
            # num_voxels (3 ints)
            vox_fmt = "=3i"
            data = binfile.read(struct.calcsize(vox_fmt))
            num_voxels = np.array(struct.unpack(vox_fmt, data), dtype=int)

            # x_voxels (num_voxels[0] + 1 floats)
            voxels_fmt = "={}f".format(num_voxels[0]+1)
            data = binfile.read(struct.calcsize(voxels_fmt))
            x_pos = np.frombuffer(data, dtype=np.float32)

            # y_voxels (num_voxels[1] + 1 floats)
            voxels_fmt = "={}f".format(num_voxels[1]+1)
            data = binfile.read(struct.calcsize(voxels_fmt))
            y_pos = np.frombuffer(data, dtype=np.float32)

            # z_voxels (num_voxels[2] + 1 floats)
            voxels_fmt = "={}f".format(num_voxels[2]+1)
            data = binfile.read(struct.calcsize(voxels_fmt))
            z_pos = np.frombuffer(data, dtype=np.float32)

            # number of nonzero dose voxels (1 int)
            nonzero_fmt = "=i"
            data = binfile.read(struct.calcsize(nonzero_fmt))
            num_nonzero = struct.unpack(nonzero_fmt, data)[0]

            # voxel indices of nonzero dose voxels (num_nonzero ints)
            voxel_indices_fmt = "={}i".format(num_nonzero)
            data = binfile.read(struct.calcsize(voxel_indices_fmt))
            voxel_indices = np.frombuffer(data, dtype=np.int32)

            # nonzero dose voxels (num_nonzero floats)
            nonzero_doses_fmt = "={}f".format(num_nonzero)
            data = binfile.read(struct.calcsize(nonzero_doses_fmt))
            nonzero_doses = np.frombuffer(data, dtype=np.float32)

            # nonzero uncert voxels (num_nonzero floats)
            nonzero_uncerts_fmt = "={}f".format(num_nonzero)
            data = binfile.read(struct.calcsize(nonzero_uncerts_fmt))
            nonzero_uncerts = np.frombuffer(data, dtype=np.float32)

            doses = np.zeros(num_voxels[0] * num_voxels[1] * num_voxels[2], dtype=np.float32)
            uncerts = np.zeros(num_voxels[0] * num_voxels[1] * num_voxels[2], dtype=np.float32)

            doses[voxel_indices] = nonzero_doses
            uncerts[voxel_indices] = nonzero_uncerts

            doses = np.reshape(doses, (num_voxels[2], num_voxels[1], num_voxels[0]))
            uncerts = np.reshape(uncerts, (num_voxels[2], num_voxels[1], num_voxels[0]))

        return cls(num_voxels, x_pos, y_pos, z_pos, doses.astype(np.float), uncerts.astype(np.float), os.path.basename(path))

    @classmethod
    def from_randydos(cls, path):
        "Create Dose instance from randydos file"
        print("Loading {}".format(path))

        with open(path, "rb") as randyfile:
            # num_voxels (3 ints)
            vox_fmt = "=3i"
            data = randyfile.read(struct.calcsize(vox_fmt))
            num_voxels = struct.unpack(vox_fmt, data)

            # x_voxels (num_voxels[0] + 1 floats)
            voxels_fmt = "={}f".format(num_voxels[0]+1)
            data = randyfile.read(struct.calcsize(voxels_fmt))
            x_pos = np.frombuffer(data, dtype=np.float32)

            # y_voxels (num_voxels[1] + 1 floats)
            voxels_fmt = "={}f".format(num_voxels[1]+1)
            data = randyfile.read(struct.calcsize(voxels_fmt))
            y_pos = np.frombuffer(data, dtype=np.float32)

            # z_voxels (num_voxels[2] + 1 floats)
            voxels_fmt = "={}f".format(num_voxels[2]+1)
            data = randyfile.read(struct.calcsize(voxels_fmt))
            z_pos = np.frombuffer(data, dtype=np.float32)

            # number of nonzero dose voxels (1 int)
            nonzero_fmt = "=i"
            data = randyfile.read(struct.calcsize(nonzero_fmt))
            num_nonzero = struct.unpack(nonzero_fmt, data)[0]

            # number of voxel blocks (1 int)
            num_block_fmt = "=i"
            data = randyfile.read(struct.calcsize(num_block_fmt))
            num_blocks = struct.unpack(num_block_fmt, data)[0]

            # Nonzero voxel blocks (2 * num_blocks ints)
            block_fmt = "={}i".format(2 * num_blocks)
            data = randyfile.read(struct.calcsize(block_fmt))
            blocks = struct.unpack(block_fmt, data)
            # For some reason performance suffers greatly if frombuffer is used
            #blocks = np.frombuffer(data, dtype=np.int32)

            # nonzero dose voxels (num_nonzero floats)
            nonzero_doses_fmt = "={}f".format(num_nonzero)
            data = randyfile.read(struct.calcsize(nonzero_doses_fmt))
            nonzero_doses = np.frombuffer(data, dtype=np.float32)

            # nonzero uncert voxels (num_nonzero floats)
            nonzero_uncerts_fmt = "={}f".format(num_nonzero)
            data = randyfile.read(struct.calcsize(nonzero_uncerts_fmt))
            nonzero_uncerts = np.frombuffer(data, dtype=np.float32)

            doses = np.zeros(num_voxels[0] * num_voxels[1] * num_voxels[2], dtype=np.float32)
            uncerts = np.zeros(num_voxels[0] * num_voxels[1] * num_voxels[2], dtype=np.float32)

            # Snippet from Randy
            processed = 0
            for block_start, block_end in zip(blocks[::2], blocks[1::2]):
                block_size = block_end - block_start
                doses[block_start:block_end] = nonzero_doses[processed:processed + block_size]
                uncerts[block_start:block_end] = nonzero_uncerts[processed:processed + block_size]
                processed += block_size

            doses = np.reshape(doses, (num_voxels[2], num_voxels[1], num_voxels[0]))
            uncerts = np.reshape(uncerts, (num_voxels[2], num_voxels[1], num_voxels[0]))

            return cls(num_voxels, x_pos, y_pos, z_pos, doses.astype(np.float), uncerts.astype(np.float), os.path.basename(path))