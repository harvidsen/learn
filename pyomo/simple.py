import matplotlib.pyplot as plt
import pyomo.environ as pyo
from instance_generator import create_random_instance
from plotting import plot_price_and_production
from pyomo.opt import SolverFactory

model = pyo.AbstractModel()

# timesteps
model.t = pyo.Param(within=pyo.NonNegativeIntegers)
model.T = pyo.RangeSet(1, model.t)

# power plants
model.i = pyo.Param(within=pyo.NonNegativeIntegers)
model.I = pyo.RangeSet(1, model.i)

# prices
model.c = pyo.Param(model.T)

# production
model.p = pyo.Var(model.I, model.T, domain=pyo.NonNegativeReals)
model.max_prod = pyo.Param(model.I)


def objective(m):
    return -sum(m.c[t] * m.p[i, t] for i in m.I for t in m.T)


model.OBJ = pyo.Objective(rule=objective)


def max_production(m, i, t):
    return m.p[i, t] <= m.max_prod[i]


model.MaxProduction = pyo.Constraint(model.I, model.T, rule=max_production)


def run():
    opt = SolverFactory("glpk")

    instance = create_random_instance(model, num_timesteps=24, num_plants=3, seed=42)
    print("Solving small")
    result = opt.solve(instance)
    print("Finished", result)
    plot_price_and_production(instance)
    plt.show()

    instance = create_random_instance(
        model, num_timesteps=24 * 7 * 25, num_plants=50, seed=42
    )
    print("Solving large")
    result = opt.solve(instance)
    print("Finished", result)


if __name__ == "__main__":
    run()
