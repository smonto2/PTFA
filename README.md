# Probabilistic Targeted Factor Analysis (PTFA)

**PTFA** is a probabilistic extension of Partial Least Squares (PLS), designed to extract latent factors from predictors \(X\) and targets \(Y\) for optimal prediction. It leverages an Expectation-Maximization (EM) algorithm for robust parameter estimation, accommodating challenges such as missing data, stochastic volatility, and dynamic factors.

The framework balances flexibility and efficiency, providing an alternative to traditional methods like PCA and standard PLS by incorporating probabilistic foundations.

## Features

- Joint estimation of latent factors and parameters.
- Robust against noise, missing data, and model uncertainty.
- Extensible to stochastic volatility, mixed-frequency data and dynamic factor models.
- Competitive performance in high-dimensional forecasting tasks.

## Installation

You can install PTFA from PyPI:

```bash
pip install ptfa
```

## Usage

Here is a quick example of how to use the ProbabilisticTFA class:

```python
import numpy as np
from ptfa import ProbabilisticTFA

# Example data: predictors (X) and targets (Y)
X = np.random.rand(100, 10)  # 100 observations, 10 predictors
Y = np.random.rand(100, 2)   # 100 observations, 2 targets

# Initialize PTFA model with desired number of components
model = ProbabilisticTFA(n_components=3)

# Fit the model
model.fit(X, Y)

# Calculate in-sample predictions
Y_predicted = model.fitted()

# Calculate out-of-sample forecasts
X = np.random.rand(100, 10)
Y_forecast = model.predict(X)

print("Predicted targets:")
print(Y_predicted)

print("Forecasted targets:")
print(Y_forecast)

```

## Contributing

Feel free to open issues or contribute to the repository through pull requests. We welcome suggestions and improvements.

## BibTeX Citation
If you use PTFA we would appreciate if you cite our work as: 
```bibtex
@article{HerculanoMontoya-Blandon2024,
  title={Probabilistic Targeted Factor Analysis},
  author={Herculano, Miguel C. and Montoya-Blandon, Santiago},
  journal={xxx},
  year={2024},
  volume={xxx},
  pages={xxx},
  doi={xxx}
}
```
## Licence 

This project is licensed under the MIT License.
