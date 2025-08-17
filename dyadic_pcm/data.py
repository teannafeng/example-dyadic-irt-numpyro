import pandas as pd
import jax.numpy as jnp

def prepare_data(df: pd.DataFrame) -> dict:
    """
    Prepare model data given raw input data.

     Args:
        df: pd.DataFrame
            Input dataframe with columns:
              - "item": item IDs
              - "actor": actor IDs
              - "partner": partner IDs
              - "unique.pair": undirected pair IDs
              - "selector": role IDs (1 or 2)
              - "x": responses (0, ..., M)
    """
    I = int(df["item"].max())
    A = int(df["actor"].max())
    U = int(df["unique.pair"].max())
    N = int(len(df))
    M = int(df["x"].max())

    data = dict(
        I  = I,
        A  = A,
        U  = U,
        N  = N,
        M  = M,
        aa = jnp.array(df["actor"].to_numpy(), dtype=jnp.int32),
        pp = jnp.array(df["partner"].to_numpy(), dtype=jnp.int32),
        ii = jnp.array(df["item"].to_numpy(), dtype=jnp.int32),
        x  = jnp.array(df["x"].to_numpy(), dtype=jnp.int32),
        dd = jnp.array(df["unique.pair"].to_numpy(), dtype=jnp.int32),
        mm = jnp.array(df["selector"].to_numpy(), dtype=jnp.int32),
    )

    # Do quick sanity checks
    assert data["ii"].min() >= 1 and data["ii"].max() <= I
    assert data["aa"].min() >= 1 and data["aa"].max() <= A
    assert data["dd"].min() >= 1 and data["dd"].max() <= U
    assert ( (data["mm"] == 1) | (data["mm"] == 2) ).all()
    assert data["x"].min() >= 0 and data["x"].max() <= M

    return data
