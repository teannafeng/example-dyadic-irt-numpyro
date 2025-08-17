from jax import random
from numpyro.infer import MCMC, NUTS
from .model import dyadic_pcm_basic

def fit_model(data_dict: dict, 
              seed: int = 12345, 
              num_warmup : int = 1000, 
              num_samples : int = 1000, 
              num_chains : int = 4, 
              thinning: int = 20,
              chain_method : str = "sequential"):
    
    kernel = NUTS(dyadic_pcm_basic)
    mcmc = MCMC(
        kernel, 
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        num_chains=num_chains, 
        thinning=thinning, 
        chain_method=chain_method
    )
    mcmc.run(random.PRNGKey(seed), **data_dict)
    return mcmc