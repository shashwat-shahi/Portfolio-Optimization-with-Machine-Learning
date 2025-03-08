"""
Utilities for Portfolio Optimization

This module provides risk metrics and performance analytics for 
portfolio optimization and analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List, Tuple, Union
import warnings


class RiskMetrics:
    """
    Risk measurement and calculation utilities.
    """
    
    @staticmethod
    def value_at_risk(returns: pd.Series, 
                     confidence_level: float = 0.05,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        confidence_level : float
            Confidence level (e.g., 0.05 for 5% VaR)
        method : str
            Method to use ('historical', 'parametric', 'cornish_fisher')
            
        Returns:
        --------
        var : float
            Value at Risk
        """
        if method == 'historical':
            return np.percentile(returns.dropna(), confidence_level * 100)
        elif method == 'parametric':
            mu = returns.mean()
            sigma = returns.std()
            return stats.norm.ppf(confidence_level, mu, sigma)
        elif method == 'cornish_fisher':
            mu = returns.mean()
            sigma = returns.std()
            skew = stats.skew(returns.dropna())
            kurt = stats.kurtosis(returns.dropna())
            
            # Cornish-Fisher expansion
            z = stats.norm.ppf(confidence_level)
            cf_z = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * skew**2 / 36
            
            return mu + sigma * cf_z
        else:
            raise ValueError(f"Unknown VaR method: {method}")
            
    @staticmethod
    def conditional_value_at_risk(returns: pd.Series, 
                                confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        confidence_level : float
            Confidence level
            
        Returns:
        --------
        cvar : float
            Conditional Value at Risk
        """
        var = RiskMetrics.value_at_risk(returns, confidence_level, 'historical')
        return returns[returns <= var].mean()
        
    @staticmethod
    def maximum_drawdown(returns: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
            
        Returns:
        --------
        metrics : dict
            Dictionary with drawdown metrics
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        
        max_dd = drawdowns.min()
        max_dd_idx = drawdowns.idxmin()
        
        # Find recovery date
        recovery_date = None
        if max_dd_idx < len(drawdowns) - 1:
            post_dd = drawdowns.loc[max_dd_idx:]
            recovery_idx = post_dd[post_dd >= 0].index
            if len(recovery_idx) > 0:
                recovery_date = recovery_idx[0]
                
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_idx,
            'recovery_date': recovery_date,
            'drawdown_duration': None if recovery_date is None else (recovery_date - max_dd_idx).days,
            'current_drawdown': drawdowns.iloc[-1]
        }
        
    @staticmethod
    def downside_deviation(returns: pd.Series, 
                          target_return: float = 0.0) -> float:
        """
        Calculate downside deviation.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        target_return : float
            Target return threshold
            
        Returns:
        --------
        downside_dev : float
            Downside deviation
        """
        downside_returns = returns[returns < target_return] - target_return
        return np.sqrt(np.mean(downside_returns**2))
        
    @staticmethod
    def tracking_error(portfolio_returns: pd.Series, 
                      benchmark_returns: pd.Series) -> float:
        """
        Calculate tracking error.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Portfolio returns
        benchmark_returns : pd.Series
            Benchmark returns
            
        Returns:
        --------
        tracking_error : float
            Tracking error (annualized)
        """
        active_returns = portfolio_returns - benchmark_returns
        return active_returns.std() * np.sqrt(252)  # Annualized
        
    @staticmethod
    def information_ratio(portfolio_returns: pd.Series, 
                         benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Portfolio returns
        benchmark_returns : pd.Series
            Benchmark returns
            
        Returns:
        --------
        info_ratio : float
            Information ratio
        """
        active_returns = portfolio_returns - benchmark_returns
        active_return = active_returns.mean() * 252  # Annualized
        tracking_err = RiskMetrics.tracking_error(portfolio_returns, benchmark_returns)
        
        return active_return / tracking_err if tracking_err > 0 else 0
        
    @staticmethod
    def beta(portfolio_returns: pd.Series, 
            market_returns: pd.Series) -> float:
        """
        Calculate portfolio beta.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Portfolio returns
        market_returns : pd.Series
            Market returns
            
        Returns:
        --------
        beta : float
            Portfolio beta
        """
        return np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)
        
    @staticmethod
    def treynor_ratio(portfolio_returns: pd.Series, 
                     market_returns: pd.Series,
                     risk_free_rate: float = 0.0) -> float:
        """
        Calculate Treynor ratio.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Portfolio returns
        market_returns : pd.Series
            Market returns
        risk_free_rate : float
            Risk-free rate
            
        Returns:
        --------
        treynor_ratio : float
            Treynor ratio
        """
        portfolio_return = portfolio_returns.mean() * 252
        beta = RiskMetrics.beta(portfolio_returns, market_returns)
        
        return (portfolio_return - risk_free_rate) / beta if beta != 0 else 0


class PerformanceAnalytics:
    """
    Performance analysis and attribution utilities.
    """
    
    @staticmethod
    def calculate_returns_statistics(returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive return statistics.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
            
        Returns:
        --------
        stats : dict
            Dictionary of return statistics
        """
        clean_returns = returns.dropna()
        
        # Basic statistics
        annual_return = clean_returns.mean() * 252
        annual_volatility = clean_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Higher moments
        skewness = stats.skew(clean_returns)
        kurtosis = stats.kurtosis(clean_returns)
        
        # Risk metrics
        max_dd = RiskMetrics.maximum_drawdown(clean_returns)
        var_5 = RiskMetrics.value_at_risk(clean_returns, 0.05)
        cvar_5 = RiskMetrics.conditional_value_at_risk(clean_returns, 0.05)
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'max_drawdown': max_dd['max_drawdown'],
            'var_5': var_5,
            'cvar_5': cvar_5,
            'calmar_ratio': annual_return / abs(max_dd['max_drawdown']) if max_dd['max_drawdown'] != 0 else 0,
            'sortino_ratio': annual_return / (RiskMetrics.downside_deviation(clean_returns) * np.sqrt(252))
        }
        
    @staticmethod
    def performance_attribution(portfolio_weights: pd.DataFrame,
                              asset_returns: pd.DataFrame,
                              benchmark_weights: Optional[pd.DataFrame] = None) -> Dict:
        """
        Perform performance attribution analysis.
        
        Parameters:
        -----------
        portfolio_weights : pd.DataFrame
            Portfolio weights over time
        asset_returns : pd.DataFrame
            Asset returns over time
        benchmark_weights : pd.DataFrame, optional
            Benchmark weights over time
            
        Returns:
        --------
        attribution : dict
            Performance attribution results
        """
        # Align dates
        common_dates = portfolio_weights.index.intersection(asset_returns.index)
        port_weights = portfolio_weights.loc[common_dates]
        returns = asset_returns.loc[common_dates]
        
        # Portfolio returns
        portfolio_returns = (port_weights.shift(1) * returns).sum(axis=1)
        
        # Benchmark (equal weights if not provided)
        if benchmark_weights is None:
            n_assets = len(portfolio_weights.columns)
            bench_weights = pd.DataFrame(
                np.ones((len(common_dates), n_assets)) / n_assets,
                index=common_dates,
                columns=portfolio_weights.columns
            )
        else:
            bench_weights = benchmark_weights.loc[common_dates]
            
        benchmark_returns = (bench_weights.shift(1) * returns).sum(axis=1)
        
        # Attribution components
        active_weights = port_weights - bench_weights.shift(1)
        active_returns = returns.subtract(benchmark_returns, axis=0)
        
        # Selection effect: portfolio weights * active returns
        selection_effect = (port_weights.shift(1) * active_returns).sum(axis=1)
        
        # Allocation effect: active weights * benchmark returns
        allocation_effect = (active_weights.shift(1) * benchmark_returns.values.reshape(-1, 1)).sum(axis=1)
        
        # Interaction effect
        interaction_effect = (active_weights.shift(1) * active_returns).sum(axis=1)
        
        total_active_return = portfolio_returns - benchmark_returns
        
        return {
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'active_returns': total_active_return,
            'selection_effect': selection_effect,
            'allocation_effect': allocation_effect,
            'interaction_effect': interaction_effect,
            'total_attribution': selection_effect + allocation_effect + interaction_effect
        }
        
    @staticmethod
    def rolling_performance_metrics(returns: pd.Series, 
                                  window: int = 252,
                                  metrics: List[str] = None) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        window : int
            Rolling window size
        metrics : list, optional
            List of metrics to calculate
            
        Returns:
        --------
        rolling_metrics : pd.DataFrame
            Rolling performance metrics
        """
        if metrics is None:
            metrics = ['volatility', 'sharpe_ratio', 'max_drawdown', 'var_5']
            
        results = {}
        
        for metric in metrics:
            if metric == 'volatility':
                results[metric] = returns.rolling(window).std() * np.sqrt(252)
            elif metric == 'sharpe_ratio':
                annual_ret = returns.rolling(window).mean() * 252
                annual_vol = returns.rolling(window).std() * np.sqrt(252)
                results[metric] = annual_ret / annual_vol
            elif metric == 'max_drawdown':
                def rolling_max_dd(x):
                    if len(x) < 2:
                        return np.nan
                    cumulative = (1 + x).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdowns = (cumulative - running_max) / running_max
                    return drawdowns.min()
                results[metric] = returns.rolling(window).apply(rolling_max_dd)
            elif metric == 'var_5':
                results[metric] = returns.rolling(window).quantile(0.05)
                
        return pd.DataFrame(results, index=returns.index)
        
    @staticmethod
    def risk_adjusted_performance_comparison(portfolios: Dict[str, pd.Series],
                                           risk_free_rate: float = 0.0) -> pd.DataFrame:
        """
        Compare risk-adjusted performance across portfolios.
        
        Parameters:
        -----------
        portfolios : dict
            Dictionary of portfolio name -> returns series
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation
            
        Returns:
        --------
        comparison : pd.DataFrame
            Performance comparison table
        """
        results = {}
        
        for name, returns in portfolios.items():
            stats = PerformanceAnalytics.calculate_returns_statistics(returns)
            stats['excess_return'] = stats['annual_return'] - risk_free_rate
            results[name] = stats
            
        return pd.DataFrame(results).T
        
    @staticmethod
    def calculate_risk_adjusted_improvement(baseline_returns: pd.Series,
                                          improved_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate risk-adjusted performance improvement.
        
        Parameters:
        -----------
        baseline_returns : pd.Series
            Baseline portfolio returns
        improved_returns : pd.Series
            Improved portfolio returns
            
        Returns:
        --------
        improvement : dict
            Performance improvement metrics
        """
        baseline_stats = PerformanceAnalytics.calculate_returns_statistics(baseline_returns)
        improved_stats = PerformanceAnalytics.calculate_returns_statistics(improved_returns)
        
        # Calculate improvements
        return_improvement = (improved_stats['annual_return'] - baseline_stats['annual_return']) / abs(baseline_stats['annual_return']) if baseline_stats['annual_return'] != 0 else 0
        sharpe_improvement = (improved_stats['sharpe_ratio'] - baseline_stats['sharpe_ratio']) / abs(baseline_stats['sharpe_ratio']) if baseline_stats['sharpe_ratio'] != 0 else 0
        volatility_reduction = (baseline_stats['annual_volatility'] - improved_stats['annual_volatility']) / baseline_stats['annual_volatility'] if baseline_stats['annual_volatility'] != 0 else 0
        
        return {
            'return_improvement_pct': return_improvement * 100,
            'sharpe_improvement_pct': sharpe_improvement * 100,
            'volatility_reduction_pct': volatility_reduction * 100,
            'calmar_improvement_pct': (improved_stats['calmar_ratio'] - baseline_stats['calmar_ratio']) / abs(baseline_stats['calmar_ratio']) * 100 if baseline_stats['calmar_ratio'] != 0 else 0
        }


class PortfolioAnalyzer:
    """
    Comprehensive portfolio analysis tool.
    """
    
    def __init__(self, returns: pd.DataFrame, weights: pd.DataFrame):
        """
        Initialize portfolio analyzer.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns
        weights : pd.DataFrame
            Portfolio weights over time
        """
        self.returns = returns
        self.weights = weights
        self.portfolio_returns = None
        self._calculate_portfolio_returns()
        
    def _calculate_portfolio_returns(self):
        """Calculate portfolio returns from weights and asset returns."""
        # Align dates
        common_dates = self.weights.index.intersection(self.returns.index)
        aligned_weights = self.weights.loc[common_dates]
        aligned_returns = self.returns.loc[common_dates]
        
        # Calculate portfolio returns
        self.portfolio_returns = (aligned_weights.shift(1) * aligned_returns).sum(axis=1).dropna()
        
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive portfolio analysis report.
        
        Returns:
        --------
        report : dict
            Comprehensive analysis report
        """
        # Basic performance statistics
        perf_stats = PerformanceAnalytics.calculate_returns_statistics(self.portfolio_returns)
        
        # Risk metrics
        risk_metrics = {
            'var_1': RiskMetrics.value_at_risk(self.portfolio_returns, 0.01),
            'cvar_1': RiskMetrics.conditional_value_at_risk(self.portfolio_returns, 0.01),
            'downside_deviation': RiskMetrics.downside_deviation(self.portfolio_returns),
            'max_drawdown_details': RiskMetrics.maximum_drawdown(self.portfolio_returns)
        }
        
        # Portfolio characteristics
        avg_weights = self.weights.mean()
        weight_concentration = {
            'max_weight': avg_weights.max(),
            'min_weight': avg_weights.min(),
            'effective_assets': 1.0 / np.sum(avg_weights**2),
            'weight_entropy': -np.sum(avg_weights * np.log(avg_weights + 1e-8))
        }
        
        # Turnover analysis
        weight_changes = self.weights.diff().abs()
        turnover_stats = {
            'average_turnover': weight_changes.sum(axis=1).mean(),
            'max_turnover': weight_changes.sum(axis=1).max(),
            'turnover_volatility': weight_changes.sum(axis=1).std()
        }
        
        return {
            'performance_statistics': perf_stats,
            'risk_metrics': risk_metrics,
            'portfolio_characteristics': weight_concentration,
            'turnover_analysis': turnover_stats,
            'analysis_period': {
                'start_date': self.portfolio_returns.index[0],
                'end_date': self.portfolio_returns.index[-1],
                'total_observations': len(self.portfolio_returns)
            }
        }
        
    def compare_to_benchmark(self, benchmark_returns: pd.Series) -> Dict:
        """
        Compare portfolio performance to benchmark.
        
        Parameters:
        -----------
        benchmark_returns : pd.Series
            Benchmark returns
            
        Returns:
        --------
        comparison : dict
            Performance comparison results
        """
        # Align dates
        common_dates = self.portfolio_returns.index.intersection(benchmark_returns.index)
        port_rets = self.portfolio_returns.loc[common_dates]
        bench_rets = benchmark_returns.loc[common_dates]
        
        # Performance statistics
        port_stats = PerformanceAnalytics.calculate_returns_statistics(port_rets)
        bench_stats = PerformanceAnalytics.calculate_returns_statistics(bench_rets)
        
        # Relative metrics
        tracking_error = RiskMetrics.tracking_error(port_rets, bench_rets)
        information_ratio = RiskMetrics.information_ratio(port_rets, bench_rets)
        
        return {
            'portfolio_stats': port_stats,
            'benchmark_stats': bench_stats,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'active_return': port_stats['annual_return'] - bench_stats['annual_return'],
            'relative_sharpe': port_stats['sharpe_ratio'] - bench_stats['sharpe_ratio']
        }