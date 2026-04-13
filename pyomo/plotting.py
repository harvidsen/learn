"""Plotting utilities for solved Pyomo hydropower models."""

import matplotlib.pyplot as plt
import pyomo.environ as pyo


def plot_price_and_production(instance: pyo.ConcreteModel) -> plt.Figure:
    """Plot price and total production over timesteps for a solved instance.

    Parameters
    ----------
    instance : pyo.ConcreteModel
        A solved concrete Pyomo model instance that has parameters ``c``
        (price, indexed by T) and ``p`` (production, indexed by I × T).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the two-axis plot.
    """
    timesteps = sorted(instance.T)
    prices = [pyo.value(instance.c[t]) for t in timesteps]

    # Total production across all plants per timestep
    total_production = [
        sum(pyo.value(instance.p[i, t]) for i in instance.I) for t in timesteps
    ]

    # Per-plant production for stacked area breakdown
    plants = sorted(instance.I)
    per_plant = {i: [pyo.value(instance.p[i, t]) for t in timesteps] for i in plants}

    fig, ax_price = plt.subplots(figsize=(12, 5))
    ax_prod = ax_price.twinx()

    # Stacked area for per-plant production
    stack_data = [per_plant[i] for i in plants]
    ax_prod.stackplot(
        timesteps,
        stack_data,
        labels=[f"Plant {i}" for i in plants],
        alpha=0.4,
    )
    # Total production outline
    ax_prod.plot(
        timesteps,
        total_production,
        color="tab:blue",
        linewidth=1.5,
        label="Total production",
    )

    # Price line on top
    ax_price.plot(
        timesteps, prices, color="tab:red", linewidth=2, label="Price", zorder=5
    )

    ax_price.set_xlabel("Timestep")
    ax_price.set_ylabel("Price", color="tab:red")
    ax_price.tick_params(axis="y", labelcolor="tab:red")

    ax_prod.set_ylabel("Production", color="tab:blue")
    ax_prod.tick_params(axis="y", labelcolor="tab:blue")

    # Combine legends from both axes
    lines_price, labels_price = ax_price.get_legend_handles_labels()
    lines_prod, labels_prod = ax_prod.get_legend_handles_labels()
    ax_price.legend(
        lines_price + lines_prod,
        labels_price + labels_prod,
        loc="upper left",
        fontsize="small",
    )

    fig.suptitle("Price and Production Over Time")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from instance_generator import create_random_instance
    from pyomo.opt import SolverFactory
    from simple import model as simple_model

    opt = SolverFactory("glpk")

    instance = create_random_instance(
        simple_model, num_timesteps=48, num_plants=3, seed=42
    )
    opt.solve(instance)

    fig = plot_price_and_production(instance)
    plt.show()
