class Dose(object):
    """
    Wrapper for a treatment plan dose.

    Attributes:
    coords (CoordinateSystem object): Coordinate system of dose grid.
    dose_path (string): File system path of the dose file.
    ref_plan (Plan object): Referenced plan.
    """

    def __init__(self, attrs):
        self.coords = attrs["cords"]
        self.dose_path = attrs["dose_path"]
        self.ref_plan = attrs.get("ref_plan", None)
