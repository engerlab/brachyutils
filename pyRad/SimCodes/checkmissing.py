from subprocess import call
import glob
import time
import os
import sys

def run_dosxyz(name):
    beamlets = glob.glob("%s_*.egsinp" % name)
    doses = glob.glob("%s_*.3ddose" % name)
    beamlets = set([".".join(b.split(".")[:-1]) for b in beamlets])
    doses = set([".".join(d.split(".")[:-1]) for d in doses])
    beamlets_left = list(beamlets.symmetric_difference(doses))
    beamlets_left = [b + ".egsinp" for b in beamlets_left]
    print("%i beamlets left" % len(beamlets_left))

    for filename in beamlets_left:
        call("/home/egs_git/EGSnrc/HEN_HOUSE/scripts/run_user_code_batch dosxyznrc %s %s" % (filename, "700icru"), shell=True)

if __name__ == "__main__":
    run_dosxyz(sys.argv[1])
