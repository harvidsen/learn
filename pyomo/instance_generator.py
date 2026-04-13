import numpy as np
import pyomo.environ as pyo


def create_random_instance(
    abstract_model: pyo.AbstractModel,
    num_timesteps: int,
    num_plants: int,
    price_start: float = 50.0,
    ar_coeff: float = 0.8,
    max_diff: float = 10.0,
    seed: int | None = None,
) -> pyo.ConcreteModel:
    """Create an instance of the abstract model with random data.

    Prices follow an AR(1) process, clamped so consecutive prices differ
    by no more than ``max_diff``.

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
    ar_coeff : float
        Autoregressive coefficient (0 < ar_coeff < 1).
    max_diff : float
        Maximum absolute price change between consecutive timesteps.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pyo.ConcreteModel
        A concrete instance of the abstract model populated with random data.
    """
    rng = np.random.default_rng(seed)

    # Generate autoregressive prices with bounded differences
    mean_price = price_start
    prices = [price_start]
    for _ in range(1, num_timesteps):
        noise = rng.normal(0, max_diff / 2)
        new_price = ar_coeff * prices[-1] + (1 - ar_coeff) * mean_price + noise
        diff = np.clip(new_price - prices[-1], -max_diff, max_diff)
        new_price = prices[-1] + diff
        prices.append(new_price)

    # Random max production capacity per plant (between 20 and 100)
    max_prods = rng.uniform(20, 100, size=num_plants)

    data = {
        None: {
            "t": {None: num_timesteps},
            "i": {None: num_plants},
            "c": {t + 1: float(prices[t]) for t in range(num_timesteps)},
            "max_prod": {i + 1: float(max_prods[i]) for i in range(num_plants)},
        }
    }

    return abstract_model.create_instance(data)
