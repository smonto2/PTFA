# Probabilistic Targeted Factor Analysis (PTFA)

`ptfa` provides an implementation of Probabilistic Targeted Factor Analysis, a probabilistic extension of Partial Least Squares (PLS) designed to extract latent factors from features $(X)$ with pre-specified targets $(Y)$ in mind for optimal prediction. It leverages an Expectation-Maximization (EM) algorithm for robust parameter estimation, accommodating challenges such as missing data, stochastic volatility, and dynamic factors.

The framework balances flexibility and efficiency, providing an alternative to traditional methods like Principal Component Analysis (PCA) and standard PLS by incorporating probabilistic foundations.

## Features

- Joint estimation of latent factors and parameters.
- Robust against noise, missing data, and model uncertainty.
- Extensible to stochastic volatility, mixed-frequency data and dynamic factor models.
- Competitive performance in high-dimensional forecasting tasks.

## Installation

You can install `ptfa` from PyPI:

```bash
pip install ptfa
```

## Routines

The `ptfa` module includes several classes aimed at implementing PTFA in a variety of real-world data settings:
- `ProbabilisticTFA`: main workhorse class providing factor extraction from features `X` to predict targets `Y` by extracting `n_components` number of common latent factors.
- `ProbabilisticTFA_MixedFrequency`: adapts to situations where natural measurement frequency of `X` is larger than `Y` (e.g., using monthly information to predict quarterly variables).
- `ProbabilisticTFA_StochasticVolatility`: adapts main class to deal with stochastic volatility (variance changing with time) in features and targets.
- `ProbabilisticTFA_DynamicFactors`: when factors can exhibit time-series persistence, we fit a vector autoregressive of order 1 (VAR-1) process on the latent factors.

All classes have the following methods in common:
- `__init__(self, n_components)`: creates the class instance with specified number of latent components.
- `fit(self, X, Y, ...)`: fits the PTFA model to the given data using a tailored EM algorithm for each class and extracts latent factors.
- `fitted(self, ...)`: computes the in-sample fitted values for the targets.
- `predict(self, X)`: out-of-sample predicted values of targets using new features `X`.

In addition, each class comes equipped with specific functions to handle the respective data-generating processes. More details on the routines and the additional arguments `...` each command can take can be found in the documentation for each class in the [GitHub repository](https://github.com/smonto2/PTFA/tree/main/src/ptfa/)).

Finally, all classes can handle missing-at-random data in the form of [`numpy.nan` entries](https://numpy.org/doc/stable/reference/constants.html#numpy.nan) in the data arrays `X` and `Y`. Alternatively, these arrays can be directly passed as [`numpy.MaskedArray` objects](https://numpy.org/doc/stable/reference/maskedarray.html#masked-arrays).

## Usage

Here is a quick example of how to use the main class for factor extraction and forecasting, called `ProbabilisticTFA`:

```python
import numpy as np
from ptfa import ProbabilisticTFA

# Example data: predictors (X) and targets (Y)
X = np.random.rand(100, 10)  # 100 observations, 10 predictors
Y = np.random.rand(100, 2)   # 100 observations, 2 targets

# Initialize PTFA model with desired number of components
model = ProbabilisticTFA(n_components=3)

# Fit the model to data X and Y using EM algorithm
model.fit(X, Y)

# Calculate in-sample fitted values
Y_fitted = model.fitted()

# Calculate out-of-sample forecasts
X = np.random.rand(100, 10)
Y_predicted = model.predict(X)

print("Fitted targets:")
print(Y_fitted)

print("Predicted targets:")
print(Y_predicted)

```

## Contributing

Feel free to open issues or contribute to the repository through pull requests. We welcome suggestions and improvements.

## BibTeX Citation
If you use `ptfa` we would appreciate if you cite our work as: 
```bibtex
@article{HerculanoMontoya-Blandon2024,
  title={Probabilistic Targeted Factor Analysis},
  author={Herculano, Miguel C. and Montoya-Blandon, Santiago},
  journal={arXiv preprint 2412.06688},
  year={2024},
  url={https://arxiv.org/abs/2412.06688},
}
```
## Licence 

This project is licensed under the MIT License.
