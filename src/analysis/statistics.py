"""
Statistical analysis module for loss aversion research.

This module provides statistical tools and tests specifically designed for
analyzing loss aversion in financial markets, including hypothesis testing,
effect size calculations, and power analysis.
"""

import logging
from typing import Tuple, List, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import pingouin as pg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StatisticalTests:
    """Class containing statistical tests for loss aversion analysis."""
    
    @staticmethod
    def compare_means(
        group1: np.ndarray,
        group2: np.ndarray,
        equal_var: bool = False
    ) -> Dict[str, float]:
        """
        Compare means between two groups with comprehensive statistics.
        
        Args:
            group1: First group's data
            group2: Second group's data
            equal_var: Whether to assume equal variances
            
        Returns:
            Dictionary of statistical results
        """
        results = {}
        
        # Basic t-test
        t_stat, p_value = stats.ttest_ind(
            group1,
            group2,
            equal_var=equal_var
        )
        results['t_statistic'] = t_stat
        results['p_value'] = p_value
        
        # Effect size (Cohen's d)
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohen_d = (np.mean(group1) - np.mean(group2)) / pooled_se
        results['cohens_d'] = cohen_d
        
        # Confidence intervals
        diff_mean = np.mean(group1) - np.mean(group2)
        diff_se = np.sqrt(var1/n1 + var2/n2)
        ci = stats.t.interval(
            0.95,
            n1 + n2 - 2,
            diff_mean,
            diff_se
        )
        results['ci_lower'] = ci[0]
        results['ci_upper'] = ci[1]
        
        # Non-parametric test (Mann-Whitney U)
        u_stat, u_p_value = stats.mannwhitneyu(
            group1,
            group2,
            alternative='two-sided'
        )
        results['u_statistic'] = u_stat
        results['u_p_value'] = u_p_value
        
        return results
        
    @staticmethod
    def regression_diagnostics(
        model: sm.regression.linear_model.RegressionResultsWrapper
    ) -> Dict[str, Union[float, Dict]]:
        """
        Perform comprehensive regression diagnostics.
        
        Args:
            model: Fitted statsmodels regression model
            
        Returns:
            Dictionary of diagnostic results
        """
        results = {}
        
        # R-squared and adjusted R-squared
        results['r_squared'] = model.rsquared
        results['adj_r_squared'] = model.rsquared_adj
        
        # Heteroskedasticity test
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        results['heteroskedasticity'] = {
            'statistic': bp_test[0],
            'p_value': bp_test[1]
        }
        
        # Autocorrelation test
        dw_stat = durbin_watson(model.resid)
        results['durbin_watson'] = dw_stat
        
        # Condition number (multicollinearity)
        results['condition_number'] = np.linalg.cond(model.model.exog)
        
        # VIF for each predictor
        X = pd.DataFrame(model.model.exog[:, 1:], columns=model.model.exog_names[1:])
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [
            sm.OLS(X[col], X.drop(col, axis=1)).fit().rsquared_adj
            for col in X.columns
        ]
        results['vif'] = vif_data.to_dict('records')
        
        return results
        
    @staticmethod
    def power_analysis(
        effect_size: float,
        n_samples: int,
        alpha: float = 0.05,
        test_type: str = 'two-sample'
    ) -> Dict[str, float]:
        """
        Calculate statistical power for different tests.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            n_samples: Sample size per group
            alpha: Significance level
            test_type: Type of test ('two-sample' or 'one-sample')
            
        Returns:
            Dictionary containing power analysis results
        """
        results = {}
        
        if test_type == 'two-sample':
            # Two-sample t-test power
            dof = 2 * n_samples - 2
            nc = effect_size * np.sqrt(n_samples / 2)
            crit_val = stats.t.ppf(1 - alpha/2, dof)
            power = 1 - stats.nct.cdf(crit_val, dof, nc)
            
        else:  # one-sample
            # One-sample t-test power
            dof = n_samples - 1
            nc = effect_size * np.sqrt(n_samples)
            crit_val = stats.t.ppf(1 - alpha/2, dof)
            power = 1 - stats.nct.cdf(crit_val, dof, nc)
            
        results['power'] = power
        results['effect_size'] = effect_size
        results['n_samples'] = n_samples
        results['alpha'] = alpha
        
        return results
        
    @staticmethod
    def calculate_effect_sizes(
        data: pd.DataFrame,
        group_col: str,
        measure_col: str
    ) -> Dict[str, float]:
        """
        Calculate multiple effect size measures.
        
        Args:
            data: DataFrame containing the data
            group_col: Column name for grouping variable
            measure_col: Column name for measure variable
            
        Returns:
            Dictionary of effect sizes
        """
        results = {}
        
        # Cohen's d
        results['cohens_d'] = pg.compute_effsize(
            data[measure_col][data[group_col] == data[group_col].unique()[0]],
            data[measure_col][data[group_col] == data[group_col].unique()[1]],
            eftype='cohen'
        )
        
        # Hedges' g
        results['hedges_g'] = pg.compute_effsize(
            data[measure_col][data[group_col] == data[group_col].unique()[0]],
            data[measure_col][data[group_col] == data[group_col].unique()[1]],
            eftype='hedges'
        )
        
        # Glass' delta
        control_std = data[measure_col][data[group_col] == data[group_col].unique()[0]].std()
        mean_diff = (
            data[measure_col][data[group_col] == data[group_col].unique()[1]].mean() -
            data[measure_col][data[group_col] == data[group_col].unique()[0]].mean()
        )
        results['glass_delta'] = mean_diff / control_std
        
        return results
        
    @staticmethod
    def multiple_hypothesis_correction(
        p_values: np.ndarray,
        method: str = 'fdr_bh'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple hypothesis testing correction.
        
        Args:
            p_values: Array of p-values
            method: Correction method ('fdr_bh', 'bonferroni', etc.)
            
        Returns:
            Tuple of (rejected null hypotheses, corrected p-values)
        """
        return multipletests(p_values, method=method)[:2]
        
    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray,
        statistic: callable,
        n_iterations: int = 1000,
        ci: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence intervals.
        
        Args:
            data: Input data
            statistic: Function to calculate statistic
            n_iterations: Number of bootstrap iterations
            ci: Confidence interval level
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        bootstrap_stats = []
        
        for _ in range(n_iterations):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic(sample))
            
        # Calculate percentile confidence intervals
        lower_percentile = (1 - ci) / 2
        upper_percentile = 1 - lower_percentile
        
        return (
            np.percentile(bootstrap_stats, lower_percentile * 100),
            np.percentile(bootstrap_stats, upper_percentile * 100)
        )
        
    @staticmethod
    def cross_sectional_regression(
        data: pd.DataFrame,
        y_col: str,
        x_cols: List[str],
        robust: bool = True
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Perform cross-sectional regression with robust standard errors.
        
        Args:
            data: DataFrame containing variables
            y_col: Dependent variable column
            x_cols: List of independent variable columns
            robust: Whether to use robust standard errors
            
        Returns:
            Fitted regression model
        """
        Y = data[y_col]
        X = sm.add_constant(data[x_cols])
        
        if robust:
            model = sm.RLM(Y, X).fit()
        else:
            model = sm.OLS(Y, X).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': 5}
            )
            
        return model
        
def main():
    """Example usage of statistical tests."""
    # Generate sample data
    np.random.seed(42)
    group1 = np.random.normal(0, 1, 100)
    group2 = np.random.normal(0.5, 1, 100)
    
    # Initialize tests
    stats_tests = StatisticalTests()
    
    # Run comparison
    results = stats_tests.compare_means(group1, group2)
    print("Mean Comparison Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
        
if __name__ == '__main__':
    main()