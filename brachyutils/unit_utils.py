"""Unit conversion constants used throughout :mod:`brachyutils`."""

import math

__all__ = [
	"GY",
	"CGY",
	"MM",
	"CM",
	"M",
	"S",
	"HR",
	"U",
	"BQ",
	"CI",
	"RAD",
	"DEG",
]

GY = 1.0  # Gy
CGY = 0.01  # Gy
MM = 1.0  # mm
CM = 10.0  # mm
M = 1000.0  # mm
S = 1.0  # s
HR = 3600.0  # s
U = CGY * CM * CM / HR
BQ = 1.0  # Bq
CI = 3.7e10  # Bq
RAD = 180.0 / math.pi
DEG = 1.0 #degrees
