import json


class BaseTPSClass(object):
    def __init__(self, attrs):
        if isinstance(attrs, dict):
            self.load_from_dict(attrs)

        elif isinstance(attrs, str):
            self.load_from_file(attrs)

    def apply_defaults(self, defaults):
        for key, value in defaults.iteritems():
            if not hasattr(self, key):
                setattr(self, key, value)

    def load_from_dict(self, attr_dict):
        for k, v in attr_dict.items():
            setattr(self, k, v)

    def load_from_file(self, filename):
        try:
            f = open(filename)
            attr_dict = json.load(f)
            f.close()
            self.load_from_dict(attr_dict)
        except IOError:
            raise Exception("Could not open filename {}.".format(filename))
