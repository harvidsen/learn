import numpy as np
import pyomo.environ as pyo


def _generate_ar_series(
    rng: np.random.Generator,
    length: int,
    start: float,
    ar_coeff: float,
    noise_std: float,
    max_diff: float,
    lower_bound: float = -np.inf,
    upper_bound: float = np.inf,
) -> list[float]:
    """Generate an AR(1) series with bounded step sizes and value clamps.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    length : int
        Number of values to generate.
    start : float
        Starting value.
    ar_coeff : float
        Autoregressive coefficient (0 < ar_coeff < 1).
    noise_std : float
        Standard deviation of the Gaussian noise term.
    max_diff : float
        Maximum absolute change between consecutive values.
    lower_bound : float
        Hard lower bound on values.
    upper_bound : float
        Hard upper bound on values.

    Returns
    -------
    list[float]
        The generated series.
    """
    mean = start
    values = [start]
    for _ in range(1, length):
        noise = rng.normal(0, noise_std)
        new_val = ar_coeff * values[-1] + (1 - ar_coeff) * mean + noise
        diff = np.clip(new_val - values[-1], -max_diff, max_diff)
        new_val = np.clip(values[-1] + diff, lower_bound, upper_bound)
        values.append(float(new_val))
    return values


def create_random_instance(
    abstract_model: pyo.AbstractModel,
    num_timesteps: int,
    num_plants: int,
    price_start: float = 50.0,
    price_ar_coeff: float = 0.8,
    price_max_diff: float = 10.0,
    price_max: float = 100.0,
    inflow_start: float = 5.0,
    inflow_ar_coeff: float = 0.7,
    inflow_max_diff: float = 5.0,
    inflow_max: float = 10.0,
    seed: int | None = None,
) -> pyo.ConcreteModel:
    """Create an instance of the abstract model with random data.

    Prices follow an AR(1) process, clamped so consecutive prices differ
    by no more than ``max_diff``.  If the model contains an inflow
    parameter ``q`` (indexed by plants and timesteps), an AR(1) inflow
    series is generated per plant, clamped to [0, ``inflow_max``].

    Parameters
    ----------
    abstract_model : pyo.AbstractModel
        The abstract Pyomo model to instantiate.
    num_timesteps : int
        Number of timesteps.
    num_plants : int
        Number of power plants.
    price_start : float
        Starting price at t=1.
    price_ar_coeff : float
        Autoregressive coefficient for prices (0 < price_ar_coeff < 1).
    price_max_diff : float
        Maximum absolute price change between consecutive timesteps.
    inflow_start : float
        Starting inflow value per plant.
    inflow_ar_coeff : float
        Autoregressive coefficient for inflow (0 < inflow_ar_coeff < 1).
    inflow_max_diff : float
        Maximum absolute inflow change between consecutive timesteps.
    inflow_max : float
        Maximum inflow in any single timestep.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pyo.ConcreteModel
        A concrete instance of the abstract model populated with random data.
    """
    rng = np.random.default_rng(seed)

    # Generate autoregressive prices with bounded differences
    prices = _generate_ar_series(
        rng,
        num_timesteps,
        price_start,
        price_ar_coeff,
        noise_std=price_max_diff / 2,
        max_diff=price_max_diff,
        upper_bound=price_max,
    )

    # Random max production capacity per plant (between 20 and 100)
    max_prods = rng.uniform(20, 100, size=num_plants)

    data = {
        None: {
            "t": {None: num_timesteps},
            "i": {None: num_plants},
            "c": {t + 1: prices[t] for t in range(num_timesteps)},
            "max_prod": {i + 1: float(max_prods[i]) for i in range(num_plants)},
        }
    }

    # Generate inflow data if the model has the q parameter
    if hasattr(abstract_model, "q"):
        inflows: dict[tuple[int, int], float] = {}
        for i in range(num_plants):
            plant_inflow = _generate_ar_series(
                rng,
                num_timesteps,
                inflow_start,
                inflow_ar_coeff,
                noise_std=inflow_max_diff / 2,
                max_diff=inflow_max_diff,
                lower_bound=0.0,
                upper_bound=inflow_max,
            )
            for t in range(num_timesteps):
                inflows[(i + 1, t + 1)] = plant_inflow[t]
        data[None]["q"] = inflows

    return abstract_model.create_instance(data)
