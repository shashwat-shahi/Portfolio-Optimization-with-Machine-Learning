"""
Example: Portfolio Optimization with 500+ Assets

This example demonstrates the complete portfolio optimization workflow:
1. Multi-factor risk models using PCA
2. Black-Litterman optimization with market views
3. Dynamic rebalancing with transaction cost optimization

The example aims to show a 22% improvement in risk-adjusted returns.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import our portfolio optimization modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimization.factor_models import PCAFactorModel, estimate_factor_model_performance
from portfolio_optimization.black_litterman import BlackLittermanOptimizer, ViewsBuilder
from portfolio_optimization.rebalancing import DynamicRebalancer, TransactionCostModel, RiskBudgeter
from portfolio_optimization.utils import PerformanceAnalytics, RiskMetrics, PortfolioAnalyzer


def generate_synthetic_data(n_assets: int = 500, n_periods: int = 1000, seed: int = 42):
    """
    Generate synthetic asset return data for demonstration.
    
    In a real implementation, this would be replaced with actual market data.
    """
    np.random.seed(seed)
    
    # Create factor structure
    n_factors = 10
    factor_loadings = np.random.randn(n_assets, n_factors) * 0.3
    
    # Generate factor returns
    factor_returns = np.random.multivariate_normal(
        mean=np.zeros(n_factors),
        cov=np.eye(n_factors) * 0.02,
        size=n_periods
    )
    
    # Generate idiosyncratic returns
    idiosyncratic_vol = np.random.uniform(0.1, 0.5, n_assets)
    idiosyncratic_returns = np.random.randn(n_periods, n_assets) * idiosyncratic_vol
    
    # Combine to get asset returns
    systematic_returns = factor_returns @ factor_loadings.T
    asset_returns = systematic_returns + idiosyncratic_returns
    
    # Create DataFrame
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
    asset_names = [f'Asset_{i:03d}' for i in range(n_assets)]
    
    return pd.DataFrame(asset_returns, index=dates, columns=asset_names)


def download_real_market_data():
    """
    Download real market data for S&P 500 stocks.
    This provides more realistic data for the example.
    """
    try:
        # Get S&P 500 tickers (simplified list for demo)
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'LLY', 'ABBV', 'BAC',
            'PFE', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'DIS', 'ABT', 'DHR', 'VZ',
            # Add more tickers to reach 50+ for demonstration
            'ADBE', 'NFLX', 'CRM', 'XOM', 'NKE', 'CMCSA', 'CSCO', 'ACN', 'TXN', 'QCOM',
            'HON', 'IBM', 'INTC', 'AMD', 'COP', 'NOW', 'PM', 'UNP', 'RTX', 'SBUX'
        ]
        
        # Download 2 years of data
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Downloading data for {len(tickers)} stocks from {start_date} to {end_date}")
        
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            prices = data['Close']
            
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Remove any tickers with insufficient data
        returns = returns.dropna(axis=1)
        
        print(f"Successfully downloaded data for {len(returns.columns)} assets")
        print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
        print(f"Number of observations: {len(returns)}")
        
        return returns
        
    except Exception as e:
        print(f"Error downloading real data: {e}")
        print("Falling back to synthetic data...")
        return None


def main():
    """
    Main example demonstrating portfolio optimization workflow.
    """
    print("=== Portfolio Optimization with Machine Learning Example ===\n")
    
    # 1. Data Preparation
    print("1. Loading market data...")
    
    # Try to get real market data, fallback to synthetic
    returns_data = download_real_market_data()
    
    if returns_data is None or len(returns_data.columns) < 20:
        print("Using synthetic data for demonstration...")
        returns_data = generate_synthetic_data(n_assets=50, n_periods=500)
        
    print(f"Data shape: {returns_data.shape}")
    print(f"Assets: {len(returns_data.columns)}")
    print(f"Time period: {returns_data.index[0]} to {returns_data.index[-1]}\n")
    
    # 2. Factor Model Analysis
    print("2. Building multi-factor risk model using PCA...")
    
    # Split data for out-of-sample testing
    split_point = int(len(returns_data) * 0.7)
    train_returns = returns_data.iloc[:split_point]
    test_returns = returns_data.iloc[split_point:]
    
    # Fit PCA factor model
    factor_model = PCAFactorModel(n_factors=10, standardize=True)
    factor_model.fit(train_returns)
    
    # Get model summary
    model_summary = factor_model.get_factor_summary()
    print(f"Factor model summary:")
    print(f"- Number of factors: {model_summary['n_factors']}")
    print(f"- Total variance explained: {model_summary['total_variance_explained']:.2%}")
    print(f"- Average specific risk: {model_summary['average_specific_risk']:.4f}")
    
    # Evaluate model performance
    performance = estimate_factor_model_performance(train_returns, factor_model, test_size=0.3)
    print(f"- Out-of-sample relative error: {performance['relative_error']:.4f}\n")
    
    # 3. Black-Litterman Optimization
    print("3. Implementing Black-Litterman optimization...")
    
    # Use the most recent data for optimization
    recent_returns = returns_data.iloc[-252:]  # Last year of data
    
    # Initialize Black-Litterman optimizer
    bl_optimizer = BlackLittermanOptimizer(risk_aversion=3.0, tau=0.025)
    
    # Set market equilibrium (equal weights as proxy)
    n_assets = len(recent_returns.columns)
    market_weights = np.ones(n_assets) / n_assets
    bl_optimizer.set_market_equilibrium(recent_returns, market_weights)
    
    # Add some example views
    views_builder = ViewsBuilder()
    
    # Example: Top 5 assets by recent performance will outperform bottom 5
    recent_performance = recent_returns.mean().sort_values(ascending=False)
    top_assets = recent_performance.head(5)
    bottom_assets = recent_performance.tail(5)
    
    # Create relative views
    views_matrix_list = []
    views_returns_list = []
    
    for i, (top_asset, _) in enumerate(top_assets.items()):
        for j, (bottom_asset, _) in enumerate(bottom_assets.items()):
            if i == j:  # Pair assets
                top_idx = recent_returns.columns.get_loc(top_asset)
                bottom_idx = recent_returns.columns.get_loc(bottom_asset)
                view_vector = views_builder.relative_view(top_idx, bottom_idx, n_assets)
                views_matrix_list.append(view_vector)
                views_returns_list.append(0.02)  # 2% annual outperformance
                
    views_matrix = np.vstack(views_matrix_list)
    views_returns = np.array(views_returns_list)
    
    # Add views and optimize
    bl_optimizer.add_views(views_matrix, views_returns)
    bl_optimizer.optimize()
    
    # Get optimized weights
    optimal_weights = bl_optimizer.get_portfolio_weights()
    portfolio_metrics = bl_optimizer.calculate_portfolio_metrics()
    
    print(f"Black-Litterman optimization results:")
    print(f"- Expected return: {portfolio_metrics['expected_return']:.2%}")
    print(f"- Volatility: {portfolio_metrics['volatility']:.2%}")
    print(f"- Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.3f}")
    print(f"- Effective assets: {portfolio_metrics['effective_assets']:.1f}\n")
    
    # 4. Dynamic Rebalancing with Transaction Costs
    print("4. Implementing dynamic rebalancing...")
    
    # Setup transaction cost model
    cost_model = TransactionCostModel(
        linear_cost=0.001,  # 0.1% linear cost
        market_impact=0.0001,
        bid_ask_spread=0.0005
    )
    
    # Setup rebalancer
    rebalancer = DynamicRebalancer(
        transaction_cost_model=cost_model,
        rebalancing_frequency='monthly',
        min_trade_size=0.001
    )
    
    # Simulate rebalancing over test period
    portfolio_value = 1000000  # $1M initial portfolio
    current_weights = market_weights.copy()  # Start with market weights
    target_weights = optimal_weights.values
    
    rebalancer.set_current_portfolio(current_weights, portfolio_value)
    rebalancer.set_target_portfolio(target_weights)
    
    # Perform optimization
    covariance_matrix = recent_returns.cov().values
    rebalancing_result = rebalancer.optimize_rebalancing(covariance_matrix)
    
    if rebalancing_result['success']:
        print(f"Rebalancing optimization successful:")
        print(f"- Transaction cost: ${rebalancing_result['transaction_cost']:,.2f}")
        print(f"- Turnover: {rebalancing_result['turnover']:.2%}")
        print(f"- Cost as % of portfolio: {rebalancing_result['transaction_cost']/portfolio_value:.3%}\n")
    else:
        print("Rebalancing optimization failed\n")
    
    # 5. Performance Analysis and Backtesting
    print("5. Analyzing performance improvements...")
    
    # Create baseline portfolio (equal weights)
    baseline_weights = pd.DataFrame(
        np.tile(market_weights, (len(test_returns), 1)),
        index=test_returns.index,
        columns=test_returns.columns
    )
    
    # Create optimized portfolio weights
    optimized_weights = pd.DataFrame(
        np.tile(optimal_weights.values, (len(test_returns), 1)),
        index=test_returns.index,
        columns=test_returns.columns
    )
    
    # Calculate portfolio returns
    baseline_analyzer = PortfolioAnalyzer(test_returns, baseline_weights)
    optimized_analyzer = PortfolioAnalyzer(test_returns, optimized_weights)
    
    baseline_returns = baseline_analyzer.portfolio_returns
    optimized_returns = optimized_analyzer.portfolio_returns
    
    # Performance statistics
    baseline_stats = PerformanceAnalytics.calculate_returns_statistics(baseline_returns)
    optimized_stats = PerformanceAnalytics.calculate_returns_statistics(optimized_returns)
    
    # Calculate improvement
    improvement = PerformanceAnalytics.calculate_risk_adjusted_improvement(
        baseline_returns, optimized_returns
    )
    
    print("Performance Comparison:")
    print(f"{'Metric':<20} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12}")
    print("-" * 60)
    print(f"{'Annual Return':<20} {baseline_stats['annual_return']:<12.2%} {optimized_stats['annual_return']:<12.2%} {improvement['return_improvement_pct']:<12.1f}%")
    print(f"{'Volatility':<20} {baseline_stats['annual_volatility']:<12.2%} {optimized_stats['annual_volatility']:<12.2%} {improvement['volatility_reduction_pct']:<12.1f}%")
    print(f"{'Sharpe Ratio':<20} {baseline_stats['sharpe_ratio']:<12.3f} {optimized_stats['sharpe_ratio']:<12.3f} {improvement['sharpe_improvement_pct']:<12.1f}%")
    print(f"{'Max Drawdown':<20} {baseline_stats['max_drawdown']:<12.2%} {optimized_stats['max_drawdown']:<12.2%} {'N/A':<12}")
    print(f"{'Calmar Ratio':<20} {baseline_stats['calmar_ratio']:<12.3f} {optimized_stats['calmar_ratio']:<12.3f} {improvement['calmar_improvement_pct']:<12.1f}%")
    
    # Check if we achieved the target improvement
    if improvement['sharpe_improvement_pct'] >= 22:
        print(f"\n✅ SUCCESS: Achieved {improvement['sharpe_improvement_pct']:.1f}% improvement in risk-adjusted returns (target: 22%)")
    else:
        print(f"\n⚠️  Note: Achieved {improvement['sharpe_improvement_pct']:.1f}% improvement in risk-adjusted returns (target: 22%)")
        print("   Real-world results may vary depending on market conditions and data quality.")
    
    # 6. Generate comprehensive report
    print("\n6. Generating comprehensive analysis report...")
    
    baseline_report = baseline_analyzer.generate_comprehensive_report()
    optimized_report = optimized_analyzer.generate_comprehensive_report()
    
    print("\nBaseline Portfolio Analysis:")
    print(f"- Average turnover: {baseline_report['turnover_analysis']['average_turnover']:.2%}")
    print(f"- Effective assets: {baseline_report['portfolio_characteristics']['effective_assets']:.1f}")
    print(f"- Weight concentration: {baseline_report['portfolio_characteristics']['max_weight']:.2%}")
    
    print("\nOptimized Portfolio Analysis:")
    print(f"- Average turnover: {optimized_report['turnover_analysis']['average_turnover']:.2%}")
    print(f"- Effective assets: {optimized_report['portfolio_characteristics']['effective_assets']:.1f}")
    print(f"- Weight concentration: {optimized_report['portfolio_characteristics']['max_weight']:.2%}")
    
    print("\n=== Example completed successfully! ===")
    
    return {
        'factor_model': factor_model,
        'bl_optimizer': bl_optimizer,
        'rebalancer': rebalancer,
        'baseline_stats': baseline_stats,
        'optimized_stats': optimized_stats,
        'improvement': improvement
    }


if __name__ == "__main__":
    results = main()