from brachyutils.planning.optimization.optim_cath.cluster_box_optim import get_geometric_constraints
from brachyutils.tests.test_cluster_box import test_cluster_box

def test_get_geometric_constraints():
    cbox = test_cluster_box(return_box=True)
    constraint_dict = get_geometric_constraints(cluster_box=cbox)
    print("debug here")
    

if __name__ == "__main__":
    print("Testing cluster box optimization")
    test_get_geometric_constraints()