"""
portopt package initializer.
Provides easy imports for data loading, optimization, visualization, and sensitivity modules.
"""

from .data_loader import get_data
from .optimizer import optimize_portfolio, portfolio_performance
from .visualization import (
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_sensitivity
)
from .sensitivity import sensitivity_rf

__all__ = [
    "get_data",
    "optimize_portfolio",
    "portfolio_performance",
    "plot_efficient_frontier",
    "plot_portfolio_weights",
    "plot_sensitivity",
    "sensitivity_rf",
]
