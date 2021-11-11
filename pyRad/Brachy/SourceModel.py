from ..BaseTPSClass import BaseTPSClass


class SourceModel(BaseTPSClass):
    """
        Brachytherapy source model.
        Parameters:
        name (str): Name of source model.
        core (Core object): Active core of the source
    """
    def __init__(self, attrs=None):
        if attrs:
            super(SourceModel, self).__init__(attrs)
