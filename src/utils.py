import numpy as np
from scipy.stats import qmc

def generate_deterministic_dr_params(dr_lower_bound, dr_upper_bound, power_of_two_samples=None, n_samples=None, seed=42):
    rng = np.random.default_rng(seed)
    # 1. Create a Sobol sequence sampler
    sampler = qmc.Sobol(d=len(dr_lower_bound), scramble=True, rng=rng)

    # 2. Generate samples in the unit hypercube
    if n_samples is not None:
        samples = sampler.random(n=n_samples)
    else:
        samples = sampler.random_base2(m=power_of_two_samples)

    # 3. Scale samples to the desired DR parameter ranges
    dr_params = qmc.scale(samples, dr_lower_bound, dr_upper_bound)

    return dr_params
