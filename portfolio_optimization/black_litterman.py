"""
Black-Litterman Portfolio Optimization with Bayesian Inference

This module implements the Black-Litterman model for incorporating market views
into portfolio optimization, improving risk-adjusted returns through Bayesian inference.
"""

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize
from typing import Optional, Dict, List, Tuple, Union
import warnings


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimizer with Bayesian inference.
    
    This implementation incorporates investor views into the portfolio optimization
    process using Bayesian methods to improve risk-adjusted returns.
    """
    
    def __init__(self, 
                 risk_aversion: float = 3.0,
                 tau: float = 0.025,
                 market_cap_weights: Optional[np.ndarray] = None):
        """
        Initialize Black-Litterman optimizer.
        
        Parameters:
        -----------
        risk_aversion : float
            Risk aversion coefficient (default: 3.0)
        tau : float
            Uncertainty scaling factor (default: 0.025)
        market_cap_weights : np.ndarray, optional
            Market capitalization weights for equilibrium returns
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.market_cap_weights = market_cap_weights
        
        # Model components
        self.expected_returns = None
        self.covariance_matrix = None
        self.equilibrium_returns = None
        self.posterior_returns = None
        self.posterior_covariance = None
        self.optimal_weights = None
        
        # Views
        self.views_matrix = None
        self.views_returns = None
        self.views_uncertainty = None
        
        self.is_fitted = False
        
    def set_market_equilibrium(self, 
                             returns: pd.DataFrame,
                             market_cap_weights: Optional[np.ndarray] = None) -> 'BlackLittermanOptimizer':
        """
        Set market equilibrium assumptions.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical asset returns
        market_cap_weights : np.ndarray, optional
            Market capitalization weights
            
        Returns:
        --------
        self : BlackLittermanOptimizer
            Self for method chaining
        """
        self.asset_names = returns.columns.tolist()
        self.n_assets = len(self.asset_names)
        
        # Calculate sample covariance matrix
        self.covariance_matrix = returns.cov().values
        
        # Set market cap weights (equal weights if not provided)
        if market_cap_weights is not None:
            self.market_cap_weights = market_cap_weights
        elif self.market_cap_weights is None:
            self.market_cap_weights = np.ones(self.n_assets) / self.n_assets
            
        # Calculate implied equilibrium returns
        self.equilibrium_returns = self._calculate_equilibrium_returns()
        
        return self
        
    def _calculate_equilibrium_returns(self) -> np.ndarray:
        """
        Calculate implied equilibrium returns using reverse optimization.
        
        Returns:
        --------
        equilibrium_returns : np.ndarray
            Implied equilibrium returns
        """
        return self.risk_aversion * self.covariance_matrix @ self.market_cap_weights
        
    def add_views(self, 
                  views_matrix: np.ndarray,
                  views_returns: np.ndarray,
                  views_uncertainty: Optional[np.ndarray] = None) -> 'BlackLittermanOptimizer':
        """
        Add investor views to the optimization.
        
        Parameters:
        -----------
        views_matrix : np.ndarray
            Picking matrix P (n_views, n_assets)
        views_returns : np.ndarray
            Expected returns for views Q (n_views,)
        views_uncertainty : np.ndarray, optional
            Uncertainty matrix for views Omega (n_views, n_views)
            
        Returns:
        --------
        self : BlackLittermanOptimizer
            Self for method chaining
        """
        self.views_matrix = views_matrix
        self.views_returns = views_returns
        
        n_views = len(views_returns)
        
        # Calculate views uncertainty if not provided
        if views_uncertainty is None:
            # Default: diagonal matrix with view-specific uncertainties
            alpha = 0.1  # Confidence level (lower = more confident)
            views_variance = np.diag(
                np.diag(views_matrix @ (self.tau * self.covariance_matrix) @ views_matrix.T)
            ) / alpha
            self.views_uncertainty = views_variance
        else:
            self.views_uncertainty = views_uncertainty
            
        return self
        
    def optimize(self) -> 'BlackLittermanOptimizer':
        """
        Perform Black-Litterman optimization.
        
        Returns:
        --------
        self : BlackLittermanOptimizer
            Self for method chaining
        """
        if self.equilibrium_returns is None:
            raise ValueError("Must set market equilibrium first")
            
        # Prior parameters
        mu_prior = self.equilibrium_returns
        sigma_prior = self.tau * self.covariance_matrix
        
        if self.views_matrix is not None:
            # Bayesian update with views
            P = self.views_matrix
            Q = self.views_returns
            Omega = self.views_uncertainty
            
            # Posterior covariance
            sigma_prior_inv = linalg.inv(sigma_prior)
            omega_inv = linalg.inv(Omega)
            
            posterior_cov_inv = sigma_prior_inv + P.T @ omega_inv @ P
            self.posterior_covariance = linalg.inv(posterior_cov_inv)
            
            # Posterior mean
            posterior_mean_term1 = sigma_prior_inv @ mu_prior
            posterior_mean_term2 = P.T @ omega_inv @ Q
            self.posterior_returns = self.posterior_covariance @ (
                posterior_mean_term1 + posterior_mean_term2
            )
        else:
            # No views - use prior
            self.posterior_returns = mu_prior
            self.posterior_covariance = sigma_prior
            
        # Calculate optimal portfolio weights
        self.optimal_weights = self._calculate_optimal_weights()
        
        self.is_fitted = True
        return self
        
    def _calculate_optimal_weights(self) -> np.ndarray:
        """
        Calculate optimal portfolio weights using mean-variance optimization.
        
        Returns:
        --------
        weights : np.ndarray
            Optimal portfolio weights
        """
        # Total covariance matrix (posterior + original)
        total_cov = self.posterior_covariance + self.covariance_matrix
        
        # Solve for optimal weights: w = (1/λ) * Σ^(-1) * μ
        cov_inv = linalg.inv(total_cov)
        unconstrained_weights = cov_inv @ self.posterior_returns / self.risk_aversion
        
        # Normalize to sum to 1 (budget constraint)
        ones = np.ones(self.n_assets)
        scaling_factor = 1.0 / (ones.T @ cov_inv @ ones)
        constrained_weights = scaling_factor * cov_inv @ ones
        
        # Combine unconstrained and constrained solutions
        # This is a simplified approach - more sophisticated constraints can be added
        return unconstrained_weights / np.sum(unconstrained_weights)
        
    def get_portfolio_weights(self) -> pd.Series:
        """
        Get optimal portfolio weights.
        
        Returns:
        --------
        weights : pd.Series
            Portfolio weights with asset names as index
        """
        if not self.is_fitted:
            raise ValueError("Must optimize first")
            
        return pd.Series(self.optimal_weights, index=self.asset_names)
        
    def get_expected_returns(self) -> pd.Series:
        """
        Get posterior expected returns.
        
        Returns:
        --------
        returns : pd.Series
            Expected returns with asset names as index
        """
        if not self.is_fitted:
            raise ValueError("Must optimize first")
            
        return pd.Series(self.posterior_returns, index=self.asset_names)
        
    def calculate_portfolio_metrics(self) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Returns:
        --------
        metrics : dict
            Dictionary containing portfolio metrics
        """
        if not self.is_fitted:
            raise ValueError("Must optimize first")
            
        weights = self.optimal_weights
        total_cov = self.posterior_covariance + self.covariance_matrix
        
        # Portfolio expected return
        portfolio_return = weights @ self.posterior_returns
        
        # Portfolio variance and volatility
        portfolio_variance = weights @ total_cov @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Maximum weight concentration
        max_weight = np.max(np.abs(weights))
        
        # Number of effective assets (inverse of HHI)
        hhi = np.sum(weights**2)
        effective_assets = 1.0 / hhi if hhi > 0 else 0
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_weight': max_weight,
            'effective_assets': effective_assets,
            'total_weight': np.sum(weights)
        }


class ViewsBuilder:
    """
    Helper class for building views matrices for Black-Litterman optimization.
    """
    
    @staticmethod
    def relative_view(asset1_idx: int, asset2_idx: int, n_assets: int) -> np.ndarray:
        """
        Create a relative view (asset1 outperforms asset2).
        
        Parameters:
        -----------
        asset1_idx : int
            Index of first asset
        asset2_idx : int
            Index of second asset
        n_assets : int
            Total number of assets
            
        Returns:
        --------
        view_vector : np.ndarray
            View vector (1, n_assets)
        """
        view = np.zeros(n_assets)
        view[asset1_idx] = 1.0
        view[asset2_idx] = -1.0
        return view.reshape(1, -1)
        
    @staticmethod
    def absolute_view(asset_idx: int, n_assets: int) -> np.ndarray:
        """
        Create an absolute view (single asset return).
        
        Parameters:
        -----------
        asset_idx : int
            Index of asset
        n_assets : int
            Total number of assets
            
        Returns:
        --------
        view_vector : np.ndarray
            View vector (1, n_assets)
        """
        view = np.zeros(n_assets)
        view[asset_idx] = 1.0
        return view.reshape(1, -1)
        
    @staticmethod
    def portfolio_view(weights: np.ndarray) -> np.ndarray:
        """
        Create a portfolio view (weighted combination).
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
            
        Returns:
        --------
        view_vector : np.ndarray
            View vector (1, n_assets)
        """
        return weights.reshape(1, -1)
        
    @staticmethod
    def sector_view(sector_indices: List[int], n_assets: int, 
                   equal_weight: bool = True) -> np.ndarray:
        """
        Create a sector view (sector outperformance).
        
        Parameters:
        -----------
        sector_indices : List[int]
            Indices of assets in the sector
        n_assets : int
            Total number of assets
        equal_weight : bool
            Whether to use equal weights within sector
            
        Returns:
        --------
        view_vector : np.ndarray
            View vector (1, n_assets)
        """
        view = np.zeros(n_assets)
        if equal_weight:
            weight = 1.0 / len(sector_indices)
            for idx in sector_indices:
                view[idx] = weight
        else:
            # Custom weights can be implemented here
            pass
            
        return view.reshape(1, -1)


def calculate_implied_views(
    returns: pd.DataFrame,
    market_weights: np.ndarray,
    portfolio_weights: np.ndarray,
    risk_aversion: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate implied views from portfolio deviations.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
    market_weights : np.ndarray
        Market capitalization weights
    portfolio_weights : np.ndarray
        Target portfolio weights
    risk_aversion : float
        Risk aversion parameter
        
    Returns:
    --------
    views_matrix : np.ndarray
        Implied views matrix
    views_returns : np.ndarray
        Implied view returns
    """
    # Calculate deviations from market weights
    weight_deviations = portfolio_weights - market_weights
    
    # Use significant deviations as views
    threshold = 0.01  # 1% threshold
    significant_deviations = np.abs(weight_deviations) > threshold
    
    if not np.any(significant_deviations):
        return None, None
        
    # Create views for significant deviations
    n_views = np.sum(significant_deviations)
    n_assets = len(portfolio_weights)
    
    views_matrix = np.zeros((n_views, n_assets))
    views_returns = np.zeros(n_views)
    
    view_idx = 0
    cov_matrix = returns.cov().values
    
    for asset_idx, is_significant in enumerate(significant_deviations):
        if is_significant:
            views_matrix[view_idx, asset_idx] = 1.0
            # Implied return based on weight deviation
            implied_return = risk_aversion * weight_deviations[asset_idx] * cov_matrix[asset_idx, asset_idx]
            views_returns[view_idx] = implied_return
            view_idx += 1
            
    return views_matrix, views_returns