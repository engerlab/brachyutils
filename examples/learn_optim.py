## LEARN WITH OR-TOOLS
from ortools.math_opt.python import mathopt

# Create model using MathOpt
model = mathopt.Model(name="quadratic_model")

# Method 1: Direct multiplication (recommended)
x = model.add_variable(lb=0.0, ub=10.0, name="x")
y = model.add_variable(lb=0.0, ub=10.0, name="y")
# objective = x * x + y * y + 2 * x * y
objective = x + y + 2 * x
model.minimize(objective)

result = mathopt.solve(model, mathopt.SolverType.GUROBI)

if result.status == mathopt.SolveStatus.OPTIMAL:
    print(f"Optimal solution: x = {x.value}, y = {y.value}")
    print(f"Objective value: {result.objective_value}")
else:
    print("No optimal solution found.")


## LEARN WITH GUROBI
# import gurobipy as gb
# from gurobipy import GRB

# print("doing example 1")
# # Example 1
# model = gb.Model("ResourceAllocation")
# x1 = model.addVar(lb=0, name="Product1")
# x2 = model.addVar(lb=0, name="Product2")

# model.setObjective(10*x1 + 15*x2, GRB.MAXIMIZE)
# # model.addConstr(2*x1 + 1*x2 <= 100, "MachineConstraint")
# # model.addConstr(1*x1 + 3*x2 <= 90, "Machine2Constraint")

# model.optimize()

# if model.status == GRB.OPTIMAL:
#     print(f"Optimal solution: Product1 = {x1.X}, Product2 = {x2.X}")
#     print(f"Objective value: {model.ObjVal}")
# else:
#     print("No optimal solution found.")
#  end

# print("doing example 2")
# # Example 2
# resources = ["Carlos", "Joe", "Monika"]
# jobs = ["Tester", "Developer", "Architect"]
# scores = {
#     ("Carlos", "Tester"): 53,
#     ("Carlos", "Developer"): 61,
#     ("Carlos", "Architect"): 70,
#     ("Joe", "Tester"): 60,
#     ("Joe", "Developer"): 55,
#     ("Joe", "Architect"): 80,
#     ("Monika", "Tester"): 70,
#     ("Monika", "Developer"): 65,
#     ("Monika", "Architect"): 75,
# }
# model = gb.Model("ResourceAllocation")

# # if resource r is assigned to job j, x[r,j] = 1
# x = model.addVars(scores.keys(), vtype=GRB.BINARY, name="assign")

# for j in jobs:
#     model.addConstr(sum(x[r, j] for r in resources) == 1, f"Job_{j}")

# for r in resources:
#     model.addConstr(sum(x[r, j] for j in jobs) <= 1, f"Resource_{r}")

# model.setObjective(gb.quicksum(scores[r, j] * x[r, j] for r, j in scores), GRB.MAXIMIZE)

# # Print results
# if model.status == GRB.OPTIMAL:
#     print("Optimal assignments:")
#     for r, j in scores:
#         if x[r, j].x > 0.5:  # Using 0.5 as threshold for binary variables due to potential floating-point issues
#             print(f"{r} assigned to {j} (score: {scores[r, j]})")
#     print(f"Total matching score: {model.objVal}")