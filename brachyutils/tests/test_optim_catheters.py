from pathlib import Path
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import Catheter_Gurobi
from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
from gurobipy import Model

def test_catheter_gurobi_initialization():
    # we need a catheter table first!
    pth_dicom = Path("data_test/prostate-glen-p1-dcm").resolve()
    cat_table = CatheterTable(catheter_list=list(pth_dicom.glob("RP*.dcm"))[0])

    catheter_vars = []
    model = Model("test_model")
    for catheter in cat_table:
        catheter_vars.append(
            Catheter_Gurobi(
            catheter=catheter,
            model=model,
            )
        )
    model.update()
    model.printStats()


if __name__ == "__main__":
    test_catheter_gurobi_initialization()