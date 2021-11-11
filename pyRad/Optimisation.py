import importlib
import json
import os
import numpy as np

class ROIRule(object):
    """
        Attributes:
        roi (Structure obj)
        priority (int)
        constraints (list of dict)
    """

    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class Optimisation(object):
    """
        Attributes:
        settings (dict of optimisation settings)
        server (Server instance)
    """

    def __init__(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

        optModule = importlib.import_module("pyRad.OptModules.{}".format(self.settings["opt_module"]))
        optModuleClass = getattr(optModule, self.settings["opt_module"])
        self.opt = optModuleClass(self.settings)

    def create_opt_file(self):
        opt_output = self.opt.create_opt_file()

        if isinstance(opt_output, basestring):
            opt_filename = opt_output
        else:
            opt_filename = opt_output[0]
            extra_files = opt_output[1]

            server = self.settings["beamlet_generation"].server
            directory = server.get_path("optimisation")
            extra_files_dir = os.path.join(directory, self.settings["name"])
            server.put_files(extra_files, extra_files_dir, make_dir=True)

        return opt_filename

    def submit_opt(self, opt_filename):
        server = self.settings["beamlet_generation"].server
        directory = server.get_path("optimisation")
        server.put_files([opt_filename], directory)

        if "mem" in self.settings:
            mem_use = self.settings["mem"] * 1.5
        else:
            mem_use = self.settings["beamlet_generation"].folder_size * 1.5

        command = '''
        cd %s
        python run_batch.py %s -m %i
        exit
        ''' % (directory, opt_filename, mem_use)
        server.exec_shell_command(command)

        return opt_filename

    @staticmethod
    def parse_opt_output(weights_filename):
        with open(weights_filename) as weightfile:
            opt_results = json.load(weightfile)

        if "control_points" in opt_results:
            particles = {}

            for cpt in opt_results["control_points"]:
                if "particle" in cpt and "energy" in cpt:
                    particle = cpt["particle"]
                    energy = cpt["energy"]
                    if particle not in particles:
                        particles[particle] = {}

                    if energy not in particles[particle]:
                        particles[particle][energy] = []

                    particles[particle][energy].append(cpt)

        beams = []
        for particle, energies in particles.iteritems():
            for energy, cpts in energies.iteritems():
                beams.append({
                    "cpts": cpts,
                    "radiation_type": particle
                })

        opt_results["beams"] = beams

        return opt_results

    @staticmethod
    def check_progress(opt_name, server):
        path_to_3ddose = os.path.join(server.get_path("optimisation"),
                                      "combined_doses",
                                      opt_name + ".3ddose")
        try:
            server.stat(path_to_3ddose)
            return "Done"
        except IOError:
            path_to_progress = os.path.join(server.get_path("optimisation"),
                                            "combined_doses",
                                            opt_name + ".progress")

            try:
                progress_command = "cat {}".format(path_to_progress)
                stdin, stdout, stderr = server.exec_command(progress_command)
                output = stdout.read()
                output = [float(x) for x in output.split("/")]
                return {"progress": round(output[0] / output[1] * 100, 1)}
            except ValueError:
                return {"progress": 20.0}

    @staticmethod
    def get_cost(opt_name, server):
        path_to_cost = os.path.join(server.get_path("optimisation"),
                                    "combined_doses",
                                    opt_name + ".cost")

        local_path = opt_name + ".cost"
        cost = {"structures": []}

        try:
            server.get_file(path_to_cost, local_path)
            with open(local_path) as jsonfile:
                cost = json.load(jsonfile)

            os.remove(local_path)
        except IOError:
            pass

        return cost

    @staticmethod
    def get_dvh(opt_name, server, attempt=0):
        path_to_dvh = os.path.join(server.get_path("optimisation"),
                                    "combined_doses",
                                    opt_name + ".dvh")

        local_path = opt_name + ".dvh"
        dvh = {}

        if (attempt < 3):
            try:
                server.get_file(path_to_dvh, local_path)
                with open(local_path) as jsonfile:
                    dvh = json.load(jsonfile)

                os.remove(local_path)
            except IOError:
                pass
            except ValueError:
                attempt += 1
                dvh = Optimisation.get_dvh(opt_name, server, attempt)

        return dvh

    @staticmethod
    def convert_rules(structure_set, rules, coordinates):
        spacing = coordinates.spacing
        voxel_volume = spacing[0] * spacing[1] * spacing[2] / 1000.0

        rule_objects = []
        for rule_index, rule in enumerate(rules):
            roi_obj = structure_set.get_roi_object(rule["roi"])
            rule_dict = {
                "name": roi_obj.roi_name,
                "downsample": rule.get("downsample", False),
                "pct_volume": rule.get("pct_volume", 100.0),
                "robust": rule.get("robust", False),
                "importance": rule["importance"],
                "mask": np.flatnonzero(roi_obj.get_mask(coordinates, phantom=True)).tolist(),
                "voxel_volume": voxel_volume,
                "hard_constraints": [],
                "dv_constraints": [],
                "mean_constraints": []
            }

            if "body" in rule_dict["name"].lower():
                rule_dict["mask"] = rule_dict["mask"][::4]

            for constraint in rule["constraints"]:
                if constraint["class"] == "volume":
                    vol = constraint["volume"]
                    if 0 < vol < 100:
                        rule_dict["dv_constraints"].append(constraint)
                    else:
                        rule_dict["hard_constraints"].append(constraint)
                elif constraint["class"] == "mean":
                    rule_dict["mean_constraints"].append(constraint)

            print("Processed mask for %s" % roi_obj.roi_name)
            rule_objects.append(rule_dict)

        rule_objects.sort(key=lambda rule: rule["importance"])
        for index, rule in enumerate(rule_objects):
            print("Prioritising ROI: %s" % rule["name"])
            for i in range(index + 1, len(rule_objects)):
                if rule["importance"] < rule_objects[i]["importance"]:
                    set_important = set(rule_objects[i]["mask"])
                    rule["mask"] = [vox_num for vox_num in rule["mask"] if vox_num not in set_important]

            if len(rule["mask"]) == 0:
                del rule_objects[index]

        return rule_objects
