from jax import nn
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def make_cov2(sigma1, sigma2, rho):
    """
    Assemble a 2x2 covariance matrix given variances 
    and correlation.

    Args:
        sigma1: variance of variable 1
        sigma2: variance of variable 2
        rho: correlation between variables
    """
    cov = jnp.array([
        [sigma1**2, rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2, sigma2**2]
    ])
    return cov

def dyadic_pcm_basic(I: int, A: int, U: int, N: int, M: int, 
                     aa: jnp.ndarray, pp: jnp.ndarray, ii: jnp.ndarray, 
                     dd: jnp.ndarray, mm: jnp.ndarray, x: jnp.ndarray):
    """
    Implement the basic dyadic partial credit model (Sim et al., 2019).

    Args:
        I: Number of items
        A: Number of actors
        U: Number of undirected pairs
        N: Number of responses
        M: Number of item parameters per item (same for all items)
        aa: Actor IDs (aa = 1, ..)
        pp: Partner IDs (pp = 1, ..)
        ii: Item IDs (ii = 1, ..)
        dd: Undirected pair IDs (pp = 1, ..)
        mm: Role IDs (1 or 2)
        x: Item responses (x = 0, 1 ... m_i)

    Notes:
        - Indices (aa, pp, ii, dd, mm) are expected to be 1-based; this functions
          shifts them to 0-based.
        - `M` is treated as global (max category across items). If some items
          have fewer categories, `delta` still has length M for each item.
    """
    # Convert to 0-based indices
    aa_idx = aa - 1
    pp_idx = pp - 1
    ii_idx = ii - 1
    dd_idx = dd - 1
    mm_idx = mm - 1

    # Priors for covariances
    sigmaA = numpyro.sample("sigmaA", dist.HalfCauchy(5.0)) # sd of alpha
    sigmaB = numpyro.sample("sigmaB", dist.HalfCauchy(5.0)) # sd of beta
    sigmaG = numpyro.sample("sigmaG", dist.HalfCauchy(5.0)) # sd of gamma

    rhoAB = numpyro.sample("rhoAB", dist.Uniform(-1.0, 1.0)) # cor between alpha and beta (within person)
    rhoG  = numpyro.sample("rhoG",  dist.Uniform(-1.0, 1.0)) # cor between gammas (within pair)

    SigmaAB = make_cov2(sigmaA, sigmaB, rhoAB) # covariance matrix of alpha and beta
    SigmaG  = make_cov2(sigmaG, sigmaG, rhoG)  # covariance matrix of gammas

    # Priors for item parameters
    delta = numpyro.sample(
        "delta",
        dist.Normal(0.0, 10.0).expand((I, M)).to_event(2)
    )

    # Priors for AB: alpha and beta for each person
    AB = numpyro.sample(
        "AB",
        dist.MultivariateNormal(
            loc=jnp.zeros(2),
            covariance_matrix=SigmaAB
        ).expand((A,)).to_event(1)
    )

    # Priors for GG: gammas for each undirected pair
    GG = numpyro.sample(
        "GG",
        dist.MultivariateNormal(
            loc=jnp.zeros(2),
            covariance_matrix=SigmaG
        ).expand((U,)).to_event(1)
    )

    # Get terms for linear predictor
    alpha_n = AB[aa_idx, 0]
    beta_n  = AB[pp_idx, 1]
    gamma_n = jnp.take_along_axis(GG[dd_idx], mm_idx[:, None], axis=1).squeeze(1)
    delta_n = delta[ii_idx, :]

    # pcminteract: item response prob for each obs.
    base = (alpha_n + beta_n + gamma_n)[:, None]
    unsummed = jnp.concatenate([jnp.zeros((N, 1)), base - delta_n], axis=1)
    cum = jnp.cumsum(unsummed, axis=1)
    probs = nn.softmax(cum, axis=1)

    # Likelihood 
    numpyro.sample("x_obs", dist.Categorical(probs=probs), obs=x)
