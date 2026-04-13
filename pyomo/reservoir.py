import matplotlib.pyplot as plt
import pyomo.environ as pyo
from instance_generator import create_random_instance
from plotting import plot_price_and_production
from pyomo.opt import SolverFactory
from simple import model

# inflow
model.q = pyo.Param(model.I, model.T)

# reservoir level
model.v = pyo.Var(model.T, domain=pyo.NonNegativeReals)
start_resevoir_level = 50.0


def max_reservoir_level(m, t):
    return m.v[t] <= 100.0


model.MaxReservoirLevel = pyo.Constraint(model.T, rule=max_reservoir_level)


def reservoir_balance(m, i, t):
    if t == 1:
        v0 = start_resevoir_level
    else:
        v0 = m.v[t - 1]

    return m.v[t] == v0 + m.q[i, t] - m.p[i, t]


model.ReservoirBalance = pyo.Constraint(model.I, model.T, rule=reservoir_balance)


opt = SolverFactory("glpk")

instance = create_random_instance(model, num_timesteps=24, num_plants=3, seed=42)
print("Solving small")
result = opt.solve(instance)
print("Finished", result)


plot_price_and_production(instance)
plt.show()
