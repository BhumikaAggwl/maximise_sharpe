import numpy as np
from src.portopt.optimizer import optimize_portfolio, portfolio_performance

def test_optimizer_shape():
    mu = np.array([0.1, 0.12, 0.08])
    sigma = np.array([[0.04, 0.006, 0.004],
                      [0.006, 0.09, 0.008],
                      [0.004, 0.008, 0.025]])
    res = optimize_portfolio(mu, sigma, rf=0.02)
    assert np.isclose(np.sum(res.x), 1.0)
    assert all(w >= 0 for w in res.x)
