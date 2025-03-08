"""
Portfolio Optimization with Machine Learning

A comprehensive portfolio optimization library implementing:
- Multi-factor risk models using PCA and factor analysis
- Black-Litterman optimization with Bayesian inference
- Dynamic rebalancing with transaction cost optimization
"""

__version__ = "1.0.0"
__author__ = "Portfolio Optimization Team"

try:
    from .factor_models import FactorModel, PCAFactorModel
    from .black_litterman import BlackLittermanOptimizer
    from .rebalancing import DynamicRebalancer
    from .utils import RiskMetrics, PerformanceAnalytics
    
    __all__ = [
        "FactorModel",
        "PCAFactorModel", 
        "BlackLittermanOptimizer",
        "DynamicRebalancer",
        "RiskMetrics",
        "PerformanceAnalytics"
    ]
except ImportError:
    # Dependencies not available, provide basic structure
    __all__ = []
    import warnings
    warnings.warn("Portfolio optimization dependencies not available. Please install numpy, pandas, scipy, and scikit-learn.")