"""
Multi-Factor Risk Models using PCA and Factor Analysis

This module implements factor models for portfolio construction with 500+ assets,
including PCA-based factor extraction and traditional factor analysis.
"""

from typing import Optional, Tuple, Dict, List
import warnings

try:
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    # Mock numpy for basic functionality testing
    class MockNumPy:
        @staticmethod
        def array(x): return x
        @staticmethod
        def zeros(shape): return [0] * (shape if isinstance(shape, int) else shape[0])
        @staticmethod
        def ones(shape): return [1] * (shape if isinstance(shape, int) else shape[0])
        @staticmethod
        def sum(x): return sum(x) if hasattr(x, '__iter__') else x
        @staticmethod
        def sqrt(x): return x**0.5 if isinstance(x, (int, float)) else [i**0.5 for i in x]
        @staticmethod
        def var(x, axis=None): return 0.1  # Mock variance
        @staticmethod
        def cov(x): return [[0.1, 0.01], [0.01, 0.1]]  # Mock covariance
        @staticmethod
        def linalg_norm(x, ord=None): return 1.0  # Mock norm
        @staticmethod
        def mean(x): return sum(x) / len(x) if hasattr(x, '__len__') else x
    
    np = MockNumPy()
    
    class MockPandas:
        class DataFrame:
            def __init__(self, data=None, index=None, columns=None):
                self.data = data or []
                self.index = index or []
                self.columns = columns or []
            def dropna(self): return self
            def iloc(self, idx): return self
            def __len__(self): return len(self.data) if self.data else 0
            def cov(self): return MockNumPy().cov([])
        class Series:
            def __init__(self, data=None, index=None):
                self.data = data or []
                self.index = index or []
            def dropna(self): return self
    
    pd = MockPandas()


class FactorModel:
    """
    Base class for factor models used in portfolio construction.
    """
    
    def __init__(self, n_factors: int = 5):
        """
        Initialize factor model.
        
        Parameters:
        -----------
        n_factors : int
            Number of factors to extract (default: 5)
        """
        self.n_factors = n_factors
        self.factor_loadings_ = None
        self.factor_returns_ = None
        self.specific_risk_ = None
        self.factor_cov_ = None
        self.is_fitted = False
        
    def fit(self, returns: pd.DataFrame) -> 'FactorModel':
        """
        Fit the factor model to return data.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns with shape (n_periods, n_assets)
            
        Returns:
        --------
        self : FactorModel
            Fitted model instance
        """
        raise NotImplementedError("Subclasses must implement fit method")
        
    def get_risk_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get factor-based risk model components.
        
        Returns:
        --------
        factor_cov : np.ndarray
            Factor covariance matrix (n_factors, n_factors)
        factor_loadings : np.ndarray
            Factor loadings matrix (n_assets, n_factors)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting risk model")
            
        return self.factor_cov_, self.factor_loadings_
        
    def predict_covariance(self) -> np.ndarray:
        """
        Predict asset covariance matrix using factor model.
        
        Returns:
        --------
        cov_matrix : np.ndarray
            Predicted covariance matrix (n_assets, n_assets)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Covariance = B * F * B' + D
        # where B = factor loadings, F = factor covariance, D = specific risk
        factor_cov = self.factor_loadings_ @ self.factor_cov_ @ self.factor_loadings_.T
        specific_cov = np.diag(self.specific_risk_)
        
        return factor_cov + specific_cov


class PCAFactorModel(FactorModel):
    """
    Factor model using Principal Component Analysis.
    
    This implementation extracts factors using PCA and estimates
    factor loadings and specific risks for portfolio optimization.
    """
    
    def __init__(self, n_factors: int = 5, standardize: bool = True):
        """
        Initialize PCA factor model.
        
        Parameters:
        -----------
        n_factors : int
            Number of principal components to extract
        standardize : bool
            Whether to standardize returns before PCA
        """
        super().__init__(n_factors)
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.pca = PCA(n_components=n_factors)
        self.explained_variance_ratio_ = None
        
    def fit(self, returns: pd.DataFrame) -> 'PCAFactorModel':
        """
        Fit PCA factor model to return data.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns with shape (n_periods, n_assets)
            
        Returns:
        --------
        self : PCAFactorModel
            Fitted model instance
        """
        if returns.isnull().any().any():
            warnings.warn("Returns contain NaN values. Consider cleaning data first.")
            returns = returns.dropna()
            
        # Store original data info
        self.asset_names = returns.columns.tolist()
        self.n_assets = len(self.asset_names)
        
        # Standardize if requested
        if self.standardize:
            returns_scaled = pd.DataFrame(
                self.scaler.fit_transform(returns),
                index=returns.index,
                columns=returns.columns
            )
        else:
            returns_scaled = returns.copy()
            
        # Fit PCA
        self.pca.fit(returns_scaled)
        
        # Extract factor returns (principal components)
        self.factor_returns_ = pd.DataFrame(
            self.pca.transform(returns_scaled),
            index=returns.index,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
        # Factor loadings are the components (transposed for our convention)
        self.factor_loadings_ = self.pca.components_.T
        
        # Calculate factor covariance matrix
        self.factor_cov_ = np.cov(self.factor_returns_.T)
        
        # Calculate specific risk (residual variance)
        predicted_returns = self.factor_returns_.values @ self.factor_loadings_.T
        residuals = returns_scaled.values - predicted_returns
        self.specific_risk_ = np.var(residuals, axis=0)
        
        # Store explained variance
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        
        self.is_fitted = True
        return self
        
    def get_factor_interpretation(self) -> pd.DataFrame:
        """
        Get factor loadings for interpretation.
        
        Returns:
        --------
        loadings_df : pd.DataFrame
            Factor loadings with asset names as index
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        return pd.DataFrame(
            self.factor_loadings_,
            index=self.asset_names,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
    def get_factor_summary(self) -> Dict:
        """
        Get summary statistics of the factor model.
        
        Returns:
        --------
        summary : dict
            Dictionary containing model summary statistics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        total_var_explained = np.sum(self.explained_variance_ratio_)
        
        return {
            'n_factors': self.n_factors,
            'n_assets': self.n_assets,
            'explained_variance_ratio': self.explained_variance_ratio_,
            'total_variance_explained': total_var_explained,
            'average_specific_risk': np.mean(self.specific_risk_),
            'max_specific_risk': np.max(self.specific_risk_),
            'min_specific_risk': np.min(self.specific_risk_)
        }


class StatisticalFactorModel(FactorModel):
    """
    Factor model using statistical factor analysis.
    
    This implementation uses maximum likelihood factor analysis
    to extract common factors from asset returns.
    """
    
    def __init__(self, n_factors: int = 5, max_iter: int = 1000, tol: float = 1e-6):
        """
        Initialize statistical factor model.
        
        Parameters:
        -----------
        n_factors : int
            Number of factors to extract
        max_iter : int
            Maximum iterations for EM algorithm
        tol : float
            Convergence tolerance
        """
        super().__init__(n_factors)
        self.max_iter = max_iter
        self.tol = tol
        self.fa = FactorAnalysis(
            n_components=n_factors,
            max_iter=max_iter,
            tol=tol,
            random_state=42
        )
        
    def fit(self, returns: pd.DataFrame) -> 'StatisticalFactorModel':
        """
        Fit statistical factor model to return data.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns with shape (n_periods, n_assets)
            
        Returns:
        --------
        self : StatisticalFactorModel
            Fitted model instance
        """
        if returns.isnull().any().any():
            warnings.warn("Returns contain NaN values. Consider cleaning data first.")
            returns = returns.dropna()
            
        # Store original data info
        self.asset_names = returns.columns.tolist()
        self.n_assets = len(self.asset_names)
        
        # Fit factor analysis
        self.fa.fit(returns)
        
        # Extract factor loadings
        self.factor_loadings_ = self.fa.components_.T
        
        # Transform data to get factor scores
        factor_scores = self.fa.transform(returns)
        self.factor_returns_ = pd.DataFrame(
            factor_scores,
            index=returns.index,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )
        
        # Calculate factor covariance matrix
        self.factor_cov_ = np.cov(self.factor_returns_.T)
        
        # Specific variances (uniquenesses)
        self.specific_risk_ = self.fa.noise_variance_
        
        self.is_fitted = True
        return self
        
    def get_factor_interpretation(self) -> pd.DataFrame:
        """
        Get factor loadings for interpretation.
        
        Returns:
        --------
        loadings_df : pd.DataFrame
            Factor loadings with asset names as index
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        return pd.DataFrame(
            self.factor_loadings_,
            index=self.asset_names,
            columns=[f'Factor_{i+1}' for i in range(self.n_factors)]
        )


def estimate_factor_model_performance(
    returns: pd.DataFrame,
    model: FactorModel,
    test_size: float = 0.2
) -> Dict:
    """
    Evaluate factor model performance using out-of-sample testing.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns data
    model : FactorModel
        Factor model to evaluate
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    performance : dict
        Dictionary containing performance metrics
    """
    n_periods = len(returns)
    split_point = int(n_periods * (1 - test_size))
    
    # Split data
    train_returns = returns.iloc[:split_point]
    test_returns = returns.iloc[split_point:]
    
    # Fit model on training data
    model.fit(train_returns)
    
    # Predict covariance for test period
    predicted_cov = model.predict_covariance()
    
    # Calculate actual covariance for test period
    actual_cov = test_returns.cov().values
    
    # Calculate metrics
    frobenius_error = np.linalg.norm(predicted_cov - actual_cov, 'fro')
    relative_error = frobenius_error / np.linalg.norm(actual_cov, 'fro')
    
    return {
        'frobenius_error': frobenius_error,
        'relative_error': relative_error,
        'predicted_cov_condition': np.linalg.cond(predicted_cov),
        'actual_cov_condition': np.linalg.cond(actual_cov)
    }