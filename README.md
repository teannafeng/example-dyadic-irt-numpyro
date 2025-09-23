This repository implements the basic dyadic IRT partial credit model from [Gin et al. (2020)](https://link.springer.com/article/10.1007/s11336-020-09718-1) in NumPyro.

- The original Stan implementaion can be found in the Stan case study: [Dyadic IRT Model](https://mc-stan.org/learn-stan/case-studies/dyadic_irt_model.html#example-application).
- This version re-creates the basic model in NumPyro, with support for Stan-like summaries and visualization.

## Installation

```bash
# Step 1. Create a virtual environment
python3 -m venv .venv   # Use 'python -m venv .venv' on Windows

# Step 2. Activate the virtual environment; pick one of the following
source .venv/bin/activate # on Mac
.venv\Scripts\activate # on Windows

# Step 3. (Optional) Upgrade pip
pip install --upgrade pip

# Step 4. Install dependencies
pip install -r requirements.txt

# Step 5. (Optional) See the list of packages installed
pip list # make sure 'venv' is activated from Step 2

# When done, deactivate the virtual environment
deactivate
```

## Usage

The quickest way to see the model in action is to open the demo notebook:

- [demo.ipynb](./example/demo.ipynb)

It walks through:

- Loading the example dataset (`dyadic_irt_data.csv`)
- Preparing data with `prepare_data`
- Fitting the model and estimating the parameters with `fit_model`
- Summarizing results and plotting MCMC traces using [ArviZ](https://python.arviz.org/)

You can run the notebook interactively (e.g. in Jupyter or VS Code), or just read it to see the example.

## References

Gin, B., Sim, N., Skrondal, A., & Rabe-Hesketh, S. (2020). A dyadic IRT model. *Psychometrika, 85*(3), 815–836. [https://doi.org/10.1007/s11336-020-09718-1](https://doi.org/10.1007/s11336-020-09718-1)

## Folder structure

```text
example-dyadic-irt-numpyro/                
│
├── dyadic_pcm/                
│   ├── __init__.py            
│   ├── model.py               # model definition (dyadic_pcm_basic)
│   ├── data.py                # prepare_data()
│   └── run.py                 # fit_model()
│
├── example/                  
│   ├── demo.ipynb             # walk-through notebook
│   └── dyadic_irt_data.csv    # example dataset
│
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt           # package dependencies
```
