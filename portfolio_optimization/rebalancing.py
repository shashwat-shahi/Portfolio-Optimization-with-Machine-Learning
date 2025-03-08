"""
Dynamic Rebalancing Algorithms with Transaction Cost Optimization

This module implements dynamic rebalancing strategies with transaction cost optimization
and risk budgeting constraints for portfolio management.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint
from typing import Optional, Dict, List, Tuple, Callable, Union
import warnings
from abc import ABC, abstractmethod


class TransactionCostModel:
    """
    Transaction cost model for portfolio rebalancing.
    """
    
    def __init__(self, 
                 linear_cost: float = 0.001,
                 fixed_cost: float = 0.0,
                 market_impact: float = 0.0001,
                 bid_ask_spread: float = 0.0005):
        """
        Initialize transaction cost model.
        
        Parameters:
        -----------
        linear_cost : float
            Linear transaction cost (e.g., 0.1% = 0.001)
        fixed_cost : float
            Fixed cost per transaction
        market_impact : float
            Market impact coefficient
        bid_ask_spread : float
            Bid-ask spread coefficient
        """
        self.linear_cost = linear_cost
        self.fixed_cost = fixed_cost
        self.market_impact = market_impact
        self.bid_ask_spread = bid_ask_spread
        
    def calculate_cost(self, 
                      trades: np.ndarray,
                      portfolio_value: float,
                      volumes: Optional[np.ndarray] = None) -> float:
        """
        Calculate total transaction cost.
        
        Parameters:
        -----------
        trades : np.ndarray
            Trade amounts (dollar values)
        portfolio_value : float
            Total portfolio value
        volumes : np.ndarray, optional
            Trading volumes for market impact calculation
            
        Returns:
        --------
        total_cost : float
            Total transaction cost
        """
        abs_trades = np.abs(trades)
        
        # Linear costs
        linear_costs = self.linear_cost * abs_trades
        
        # Fixed costs (for non-zero trades)
        fixed_costs = self.fixed_cost * np.sum(abs_trades > 0)
        
        # Market impact costs
        if volumes is not None:
            # Square-root market impact model
            market_impact_costs = self.market_impact * np.sum(
                abs_trades * np.sqrt(abs_trades / (volumes + 1e-8))
            )
        else:
            # Simplified quadratic impact
            market_impact_costs = self.market_impact * np.sum(abs_trades**2) / portfolio_value
            
        # Bid-ask spread costs
        spread_costs = self.bid_ask_spread * abs_trades
        
        return np.sum(linear_costs) + fixed_costs + market_impact_costs + np.sum(spread_costs)


class RiskBudgeter:
    """
    Risk budgeting for portfolio constraints.
    """
    
    def __init__(self, risk_budgets: Optional[Dict[str, float]] = None):
        """
        Initialize risk budgeter.
        
        Parameters:
        -----------
        risk_budgets : dict, optional
            Risk budget allocation by asset/sector
        """
        self.risk_budgets = risk_budgets or {}
        
    def calculate_risk_contributions(self, 
                                   weights: np.ndarray,
                                   covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate marginal risk contributions.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        covariance_matrix : np.ndarray
            Asset covariance matrix
            
        Returns:
        --------
        risk_contributions : np.ndarray
            Risk contributions by asset
        """
        portfolio_variance = weights @ covariance_matrix @ weights
        marginal_risk = covariance_matrix @ weights
        
        if portfolio_variance > 0:
            risk_contributions = weights * marginal_risk / portfolio_variance
        else:
            risk_contributions = np.zeros_like(weights)
            
        return risk_contributions
        
    def create_risk_budget_constraints(self, 
                                     asset_groups: Dict[str, List[int]],
                                     covariance_matrix: np.ndarray) -> List[Dict]:
        """
        Create risk budget constraints for optimization.
        
        Parameters:
        -----------
        asset_groups : dict
            Mapping of group names to asset indices
        covariance_matrix : np.ndarray
            Asset covariance matrix
            
        Returns:
        --------
        constraints : list
            List of constraint dictionaries
        """
        constraints = []
        
        def risk_budget_constraint(weights, group_indices, target_budget):
            risk_contribs = self.calculate_risk_contributions(weights, covariance_matrix)
            group_risk = np.sum(risk_contribs[group_indices])
            return group_risk - target_budget
            
        for group_name, indices in asset_groups.items():
            if group_name in self.risk_budgets:
                target = self.risk_budgets[group_name]
                constraint = {
                    'type': 'eq',
                    'fun': lambda w, idx=indices, tgt=target: risk_budget_constraint(w, idx, tgt)
                }
                constraints.append(constraint)
                
        return constraints


class DynamicRebalancer:
    """
    Dynamic portfolio rebalancer with transaction cost optimization.
    """
    
    def __init__(self,
                 transaction_cost_model: Optional[TransactionCostModel] = None,
                 risk_budgeter: Optional[RiskBudgeter] = None,
                 rebalancing_frequency: str = 'monthly',
                 min_trade_size: float = 0.001):
        """
        Initialize dynamic rebalancer.
        
        Parameters:
        -----------
        transaction_cost_model : TransactionCostModel, optional
            Transaction cost model
        risk_budgeter : RiskBudgeter, optional
            Risk budgeting system
        rebalancing_frequency : str
            Rebalancing frequency ('daily', 'weekly', 'monthly', 'quarterly')
        min_trade_size : float
            Minimum trade size threshold
        """
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel()
        self.risk_budgeter = risk_budgeter
        self.rebalancing_frequency = rebalancing_frequency
        self.min_trade_size = min_trade_size
        
        # State variables
        self.current_weights = None
        self.target_weights = None
        self.portfolio_value = None
        self.rebalancing_history = []
        
    def set_current_portfolio(self, 
                            weights: np.ndarray,
                            portfolio_value: float) -> 'DynamicRebalancer':
        """
        Set current portfolio state.
        
        Parameters:
        -----------
        weights : np.ndarray
            Current portfolio weights
        portfolio_value : float
            Current portfolio value
            
        Returns:
        --------
        self : DynamicRebalancer
            Self for method chaining
        """
        self.current_weights = weights.copy()
        self.portfolio_value = portfolio_value
        return self
        
    def set_target_portfolio(self, weights: np.ndarray) -> 'DynamicRebalancer':
        """
        Set target portfolio weights.
        
        Parameters:
        -----------
        weights : np.ndarray
            Target portfolio weights
            
        Returns:
        --------
        self : DynamicRebalancer
            Self for method chaining
        """
        self.target_weights = weights.copy()
        return self
        
    def should_rebalance(self, 
                        current_date: pd.Timestamp,
                        last_rebalance: Optional[pd.Timestamp] = None,
                        deviation_threshold: float = 0.05) -> bool:
        """
        Determine if rebalancing is needed.
        
        Parameters:
        -----------
        current_date : pd.Timestamp
            Current date
        last_rebalance : pd.Timestamp, optional
            Date of last rebalancing
        deviation_threshold : float
            Weight deviation threshold for triggering rebalancing
            
        Returns:
        --------
        should_rebalance : bool
            Whether to rebalance
        """
        # Time-based rebalancing
        if last_rebalance is not None:
            freq_map = {
                'daily': pd.Timedelta(days=1),
                'weekly': pd.Timedelta(weeks=1),
                'monthly': pd.DateOffset(months=1),
                'quarterly': pd.DateOffset(months=3)
            }
            
            if self.rebalancing_frequency in freq_map:
                time_threshold = freq_map[self.rebalancing_frequency]
                if current_date - last_rebalance >= time_threshold:
                    return True
                    
        # Deviation-based rebalancing
        if self.current_weights is not None and self.target_weights is not None:
            max_deviation = np.max(np.abs(self.current_weights - self.target_weights))
            if max_deviation > deviation_threshold:
                return True
                
        return False
        
    def optimize_rebalancing(self,
                           covariance_matrix: np.ndarray,
                           expected_returns: Optional[np.ndarray] = None,
                           constraints: Optional[List[Dict]] = None) -> Dict:
        """
        Optimize portfolio rebalancing considering transaction costs.
        
        Parameters:
        -----------
        covariance_matrix : np.ndarray
            Asset covariance matrix
        expected_returns : np.ndarray, optional
            Expected asset returns
        constraints : list, optional
            Additional optimization constraints
            
        Returns:
        --------
        result : dict
            Optimization result with new weights and costs
        """
        if self.current_weights is None or self.target_weights is None:
            raise ValueError("Must set current and target portfolios first")
            
        n_assets = len(self.current_weights)
        
        # Objective function: minimize cost + risk penalty
        def objective(new_weights):
            # Calculate trades
            trades = (new_weights - self.current_weights) * self.portfolio_value
            
            # Transaction costs
            transaction_cost = self.transaction_cost_model.calculate_cost(
                trades, self.portfolio_value
            )
            
            # Tracking error penalty (deviation from target)
            tracking_error = np.sum((new_weights - self.target_weights)**2)
            
            # Risk penalty (optional)
            risk_penalty = 0.0
            if expected_returns is not None:
                portfolio_return = new_weights @ expected_returns
                portfolio_risk = np.sqrt(new_weights @ covariance_matrix @ new_weights)
                # Penalize high risk relative to return
                if portfolio_risk > 0:
                    risk_penalty = -portfolio_return / portfolio_risk
                    
            return transaction_cost + 1000 * tracking_error + 100 * risk_penalty
            
        # Constraints
        constraint_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Budget constraint
        ]
        
        # Add risk budget constraints
        if self.risk_budgeter is not None:
            risk_constraints = self.risk_budgeter.create_risk_budget_constraints(
                {}, covariance_matrix  # Asset groups would be provided separately
            )
            constraint_list.extend(risk_constraints)
            
        # Add custom constraints
        if constraints is not None:
            constraint_list.extend(constraints)
            
        # Bounds (no short selling by default)
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # Initial guess
        x0 = self.current_weights.copy()
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                new_weights = result.x
                trades = (new_weights - self.current_weights) * self.portfolio_value
                total_cost = self.transaction_cost_model.calculate_cost(
                    trades, self.portfolio_value
                )
                
                return {
                    'success': True,
                    'new_weights': new_weights,
                    'trades': trades,
                    'transaction_cost': total_cost,
                    'turnover': np.sum(np.abs(trades)) / self.portfolio_value,
                    'optimization_result': result
                }
            else:
                return {
                    'success': False,
                    'message': result.message,
                    'optimization_result': result
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': str(e),
                'optimization_result': None
            }
            
    def execute_rebalancing(self, 
                          rebalancing_result: Dict,
                          timestamp: pd.Timestamp) -> 'DynamicRebalancer':
        """
        Execute rebalancing and update portfolio state.
        
        Parameters:
        -----------
        rebalancing_result : dict
            Result from optimize_rebalancing
        timestamp : pd.Timestamp
            Execution timestamp
            
        Returns:
        --------
        self : DynamicRebalancer
            Self for method chaining
        """
        if rebalancing_result['success']:
            # Update current weights
            self.current_weights = rebalancing_result['new_weights'].copy()
            
            # Record in history
            self.rebalancing_history.append({
                'timestamp': timestamp,
                'trades': rebalancing_result['trades'],
                'transaction_cost': rebalancing_result['transaction_cost'],
                'turnover': rebalancing_result['turnover'],
                'new_weights': rebalancing_result['new_weights'].copy()
            })
            
        return self
        
    def get_rebalancing_history(self) -> pd.DataFrame:
        """
        Get rebalancing history as DataFrame.
        
        Returns:
        --------
        history : pd.DataFrame
            Rebalancing history
        """
        if not self.rebalancing_history:
            return pd.DataFrame()
            
        records = []
        for record in self.rebalancing_history:
            flat_record = {
                'timestamp': record['timestamp'],
                'transaction_cost': record['transaction_cost'],
                'turnover': record['turnover']
            }
            records.append(flat_record)
            
        return pd.DataFrame(records)
        
    def calculate_rebalancing_metrics(self) -> Dict:
        """
        Calculate rebalancing performance metrics.
        
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        if not self.rebalancing_history:
            return {}
            
        costs = [r['transaction_cost'] for r in self.rebalancing_history]
        turnovers = [r['turnover'] for r in self.rebalancing_history]
        
        return {
            'total_rebalancing_events': len(self.rebalancing_history),
            'total_transaction_costs': np.sum(costs),
            'average_transaction_cost': np.mean(costs),
            'average_turnover': np.mean(turnovers),
            'max_turnover': np.max(turnovers) if turnovers else 0,
            'cost_to_portfolio_ratio': np.sum(costs) / self.portfolio_value if self.portfolio_value > 0 else 0
        }


class AdaptiveRebalancer(DynamicRebalancer):
    """
    Adaptive rebalancer that adjusts strategy based on market conditions.
    """
    
    def __init__(self, *args, volatility_threshold: float = 0.02, **kwargs):
        """
        Initialize adaptive rebalancer.
        
        Parameters:
        -----------
        volatility_threshold : float
            Volatility threshold for adjusting rebalancing frequency
        """
        super().__init__(*args, **kwargs)
        self.volatility_threshold = volatility_threshold
        self.market_volatility = None
        
    def update_market_conditions(self, 
                                returns: pd.DataFrame,
                                lookback_window: int = 22) -> 'AdaptiveRebalancer':
        """
        Update market condition assessment.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Recent asset returns
        lookback_window : int
            Lookback window for volatility calculation
            
        Returns:
        --------
        self : AdaptiveRebalancer
            Self for method chaining
        """
        # Calculate market volatility (equal-weighted portfolio)
        equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
        portfolio_returns = (returns * equal_weights).sum(axis=1)
        self.market_volatility = portfolio_returns.rolling(lookback_window).std().iloc[-1]
        
        return self
        
    def adapt_rebalancing_frequency(self) -> str:
        """
        Adapt rebalancing frequency based on market volatility.
        
        Returns:
        --------
        frequency : str
            Recommended rebalancing frequency
        """
        if self.market_volatility is None:
            return self.rebalancing_frequency
            
        if self.market_volatility > self.volatility_threshold:
            # High volatility: rebalance more frequently
            return 'weekly'
        else:
            # Low volatility: rebalance less frequently
            return 'monthly'


def backtest_rebalancing_strategy(
    returns: pd.DataFrame,
    target_weights_series: pd.DataFrame,
    rebalancer: DynamicRebalancer,
    initial_portfolio_value: float = 1000000,
    rebalancing_dates: Optional[List[pd.Timestamp]] = None
) -> Dict:
    """
    Backtest dynamic rebalancing strategy.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns with dates as index
    target_weights_series : pd.DataFrame
        Target weights over time
    rebalancer : DynamicRebalancer
        Rebalancing strategy
    initial_portfolio_value : float
        Initial portfolio value
    rebalancing_dates : list, optional
        Specific dates for rebalancing
        
    Returns:
    --------
    results : dict
        Backtesting results
    """
    portfolio_values = []
    portfolio_weights_history = []
    
    # Initialize
    current_weights = target_weights_series.iloc[0].values
    portfolio_value = initial_portfolio_value
    
    rebalancer.set_current_portfolio(current_weights, portfolio_value)
    
    for i, (date, target_weights) in enumerate(target_weights_series.iterrows()):
        # Update portfolio value based on returns
        if i > 0:
            period_returns = returns.loc[date]
            portfolio_return = current_weights @ period_returns
            portfolio_value *= (1 + portfolio_return)
            
        # Check if rebalancing is needed
        rebalancer.set_target_portfolio(target_weights.values)
        
        should_rebal = rebalancer.should_rebalance(
            date, 
            last_rebalance=rebalancing_dates[-1] if rebalancing_dates else None
        )
        
        if should_rebal or (rebalancing_dates and date in rebalancing_dates):
            # Perform rebalancing
            covariance_matrix = returns.rolling(window=22).cov().iloc[-1].values
            
            if not np.any(np.isnan(covariance_matrix)):
                rebalancing_result = rebalancer.optimize_rebalancing(covariance_matrix)
                rebalancer.execute_rebalancing(rebalancing_result, date)
                current_weights = rebalancer.current_weights.copy()
                
        portfolio_values.append(portfolio_value)
        portfolio_weights_history.append(current_weights.copy())
        
    return {
        'portfolio_values': pd.Series(portfolio_values, index=target_weights_series.index),
        'portfolio_weights': pd.DataFrame(portfolio_weights_history, 
                                        index=target_weights_series.index,
                                        columns=target_weights_series.columns),
        'rebalancing_metrics': rebalancer.calculate_rebalancing_metrics(),
        'rebalancing_history': rebalancer.get_rebalancing_history()
    }