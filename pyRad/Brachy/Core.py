from ..BaseTPSClass import BaseTPSClass


class Core(BaseTPSClass):
    """
        Brachytherapy active core.
        Parameters:
        name (str): Name of active core, ie Ir-192, Gd-153.
        A (int): Number of nucleons
        Z (int): Atomic number (number of protons)
        ak_per_history (float): Air kerma per photon history
        length (float): Length of active core
        radius (float): Radius of active core
    """
    def __init__(self, attrs=None):
        if attrs:
            super(Core, self).__init__(attrs)
