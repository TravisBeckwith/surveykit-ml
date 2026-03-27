"""
Statistical tests commonly used in survey research.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Optional
from survey_toolkit.utils import logger, timer


class SurveyStats:
    """Statistical analysis methods for survey data."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = {}

    # ================================================================
    # Reliability
    # ================================================================

    def cronbachs_alpha(self, columns: list[str]) -> dict:
        """
        Calculate Cronbach's alpha for internal consistency
        of a set of items (e.g., a Likert scale construct).

        Parameters
        ----------
        columns : list[str]
            Column names for scale items.

        Returns
        -------
        dict
            Alpha value, interpretation, and diagnostics.
        """
        subset = self.data[columns].dropna()
        n_items = len(columns)

        if n_items < 2:
            raise ValueError("Need at least 2 items for Cronbach's alpha.")

        item_variances = subset.var(axis=0, ddof=1).sum()
        total_variance = subset.sum(axis=1).var(ddof=1)

        alpha = (n_items / (n_items - 1)) * (
            1 - item_variances / total_variance
        )

        # Item-total correlations
        total = subset.sum(axis=1)
        item_total_corr = {
            col: round(subset[col].corr(total - subset[col]), 4)
            for col in columns
        }

        # Alpha if item deleted
        alpha_if_deleted = {}
        for col in columns:
            remaining = [c for c in columns if c != col]
            sub = subset[remaining]
            iv = sub.var(axis=0, ddof=1).sum()
            tv = sub.sum(axis=1).var(ddof=1)
            n = len(remaining)
            a = (n / (n - 1)) * (1 - iv / tv) if n > 1 else 0
            alpha_if_deleted[col] = round(a, 4)

        interpretation = (
            "Excellent" if alpha >= 0.9 else
            "Good" if alpha >= 0.8 else
            "Acceptable" if alpha >= 0.7 else
            "Questionable" if alpha >= 0.6 else
            "Poor" if alpha >= 0.5 else
            "Unacceptable"
        )

        result = {
            "alpha": round(alpha, 4),
            "n_items": n_items,
            "n_valid": len(subset),
            "interpretation": interpretation,
            "item_total_correlations": item_total_corr,
            "alpha_if_deleted": alpha_if_deleted,
        }
        self.results["cronbachs_alpha"] = result
        logger.info(f"Cronbach's Alpha: {alpha:.4f} ({interpretation})")
        return result

    # ================================================================
    # Comparison Tests
    # ================================================================

    @timer
    def compare_groups(
        self,
        variable: str,
        group_col: str,
        test: str = "auto",
    ) -> dict:
        """
        Compare a variable across groups.

        Automatically selects test based on:
        - 2 groups: t-test or Mann-Whitney U
        - 3+ groups: ANOVA or Kruskal-Wallis

        Parameters
        ----------
        variable : str
            Numeric variable to compare.
        group_col : str
            Categorical grouping variable.
        test : str
            Test to use: 'auto', 't-test', 'mann-whitney', 'anova', 'kruskal'.

        Returns
        -------
        dict
            Test results with statistic, p-value, effect size.
        """
        groups = [
            group[variable].dropna().values
            for _, group in self.data.groupby(group_col)
        ]
        group_names = list(self.data[group_col].dropna().unique())
        n_groups = len(groups)

        if n_groups < 2:
            raise ValueError(f"Need at least 2 groups, found {n_groups}.")

        # Check normality (Shapiro-Wilk)
        normality_tests = []
        for g in groups:
            if len(g) >= 8:
                _, p = stats.shapiro(g)
                normality_tests.append(p > 0.05)
            else:
                normality_tests.append(False)
        is_normal = all(normality_tests)

        # Check homogeneity of variance (Levene's)
        if n_groups >= 2:
            _, levene_p = stats.levene(*groups)
            equal_var = levene_p > 0.05
        else:
            equal_var = True

        # Auto-select test
        if test == "auto":
            if n_groups == 2:
                test = "t-test" if is_normal else "mann-whitney"
            else:
                test = "anova" if is_normal else "kruskal"

        # Run selected test
        if test == "t-test":
            stat, p = stats.ttest_ind(
                groups[0], groups[1], equal_var=equal_var
            )
            pooled_std = np.sqrt(
                (np.std(groups[0], ddof=1) ** 2
                 + np.std(groups[1], ddof=1) ** 2) / 2
            )
            effect_size = (
                (np.mean(groups[0]) - np.mean(groups[1])) / pooled_std
                if pooled_std > 0 else 0
            )
            effect_name = "Cohen's d"
            test_name = "Independent t-test" + (
                " (Welch's)" if not equal_var else ""
            )

        elif test == "mann-whitney":
            stat, p = stats.mannwhitneyu(
                groups[0], groups[1], alternative="two-sided"
            )
            n1, n2 = len(groups[0]), len(groups[1])
            effect_size = 1 - (2 * stat) / (n1 * n2)
            effect_name = "Rank-biserial r"
            test_name = "Mann-Whitney U"

        elif test == "anova":
            stat, p = stats.f_oneway(*groups)
            grand_mean = np.mean(np.concatenate(groups))
            ss_between = sum(
                len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups
            )
            ss_total = sum(
                np.sum((g - grand_mean) ** 2) for g in groups
            )
            effect_size = ss_between / ss_total if ss_total > 0 else 0
            effect_name = "Eta-squared"
            test_name = "One-way ANOVA"

        elif test == "kruskal":
            stat, p = stats.kruskal(*groups)
            n_total = sum(len(g) for g in groups)
            effect_size = (stat - n_groups + 1) / (n_total - n_groups)
            effect_name = "Epsilon-squared"
            test_name = "Kruskal-Wallis H"

        else:
            raise ValueError(f"Unknown test: {test}")

        result = {
            "test": test_name,
            "statistic": round(stat, 4),
            "p_value": round(p, 6),
            "effect_size": round(effect_size, 4),
            "effect_size_name": effect_name,
            "significant": p < 0.05,
            "groups": group_names,
            "group_stats": {
                name: {
                    "mean": round(np.mean(g), 4),
                    "median": round(np.median(g), 4),
                    "std": round(np.std(g, ddof=1), 4),
                    "n": len(g),
                }
                for name, g in zip(group_names, groups)
            },
            "normality_assumed": is_normal,
            "equal_variance": equal_var,
        }

        # Post-hoc for 3+ groups
        if n_groups > 2 and p < 0.05 and is_normal:
            melted = self.data[[variable, group_col]].dropna()
            posthoc = pairwise_tukeyhsd(
                melted[variable], melted[group_col], alpha=0.05
            )
            result["posthoc"] = str(posthoc)
            result["posthoc_summary"] = pd.DataFrame(
                posthoc._results_table.data[1:],
                columns=posthoc._results_table.data[0],
            ).to_dict("records")

        self.results["compare_groups"] = result
        return result

    # ================================================================
    # Correlation Analysis
    # ================================================================

    def correlation_matrix(
        self,
        columns: list[str],
        method: str = "pearson",
    ) -> dict:
        """
        Compute correlation matrix with p-values.

        Parameters
        ----------
        columns : list[str]
            Columns to correlate.
        method : str
            'pearson', 'spearman', or 'kendall'.

        Returns
        -------
        dict
            Correlation matrix and p-values.
        """
        subset = self.data[columns].dropna()
        n = len(subset)
        corr = subset.corr(method=method)

        # Calculate p-values
        p_values = pd.DataFrame(
            np.zeros((len(columns), len(columns))),
            columns=columns,
            index=columns,
        )
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i != j:
                    if method == "pearson":
                        _, p = stats.pearsonr(subset[col1], subset[col2])
                    elif method == "spearman":
                        _, p = stats.spearmanr(subset[col1], subset[col2])
                    elif method == "kendall":
                        _, p = stats.kendalltau(subset[col1], subset[col2])
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    p_values.iloc[i, j] = round(p, 6)

        # Significant correlations
        sig_pairs = []
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j and p_values.iloc[i, j] < 0.05:
                    sig_pairs.append({
                        "var1": col1,
                        "var2": col2,
                        "correlation": round(corr.iloc[i, j], 4),
                        "p_value": round(p_values.iloc[i, j], 6),
                    })

        result = {
            "correlation_matrix": corr,
            "p_values": p_values,
            "method": method,
            "n": n,
            "significant_pairs": sig_pairs,
        }
        self.results["correlations"] = result
        return result

    # ================================================================
    # Factor Analysis
    # ================================================================

    @timer
    def factor_analysis(
        self,
        columns: list[str],
        n_factors: Optional[int] = None,
        rotation: str = "varimax",
        method: str = "ml",
    ) -> dict:
        """
        Perform Exploratory Factor Analysis (EFA).

        Parameters
        ----------
        columns : list[str]
            Columns (items) to include.
        n_factors : int, optional
            Number of factors. Auto-detected if None.
        rotation : str
            Rotation method: 'varimax', 'promax', 'oblimin', None.
        method : str
            Extraction method: 'ml' (max likelihood) or 'minres'.

        Returns
        -------
        dict
            Loadings, variance explained, adequacy tests.
        """
        try:
            from factor_analyzer import FactorAnalyzer
            from factor_analyzer.factor_analyzer import (
                calculate_bartlett_sphericity,
                calculate_kmo,
            )
        except ImportError:
            raise ImportError(
                "factor-analyzer is required for factor analysis. "
                "Install with: pip install factor-analyzer"
            )

        subset = self.data[columns].dropna()

        if len(subset) < len(columns) * 5:
            logger.warning(
                f"Sample size ({len(subset)}) may be too small for "
                f"{len(columns)} items. Recommended: {len(columns) * 5}+"
            )

        # Adequacy tests
        chi_sq, p_bartlett = calculate_bartlett_sphericity(subset)
        kmo_per_item, kmo_overall = calculate_kmo(subset)

        # Determine number of factors
        if n_factors is None:
            eigenvalues = np.linalg.eigvals(subset.corr())
            eigenvalues = np.sort(eigenvalues.real)[::-1]
            n_factors = int(np.sum(eigenvalues > 1))  # Kaiser criterion
            logger.info(f"Auto-detected {n_factors} factors (Kaiser criterion)")

        # Fit
        fa = FactorAnalyzer(
            n_factors=n_factors, rotation=rotation, method=method
        )
        fa.fit(subset)

        loadings = pd.DataFrame(
            fa.loadings_,
            index=columns,
            columns=[f"Factor_{i + 1}" for i in range(n_factors)],
        )

        variance = fa.get_factor_variance()
        communalities = pd.Series(fa.get_communalities(), index=columns)

        # Assign items to factors based on highest loading
        item_assignments = {}
        for item in columns:
            max_factor = loadings.loc[item].abs().idxmax()
            item_assignments[item] = max_factor

        result = {
            "bartlett_chi_sq": round(chi_sq, 4),
            "bartlett_p": round(p_bartlett, 6),
            "bartlett_significant": p_bartlett < 0.05,
            "kmo": round(kmo_overall, 4),
            "kmo_adequate": kmo_overall >= 0.6,
            "kmo_interpretation": (
                "Marvelous" if kmo_overall >= 0.9 else
                "Meritorious" if kmo_overall >= 0.8 else
                "Middling" if kmo_overall >= 0.7 else
                "Mediocre" if kmo_overall >= 0.6 else
                "Miserable" if kmo_overall >= 0.5 else
                "Unacceptable"
            ),
            "n_factors": n_factors,
            "rotation": rotation,
            "method": method,
            "loadings": loadings,
            "communalities": communalities,
            "item_assignments": item_assignments,
            "variance_explained": {
                "per_factor": [round(v, 4) for v in variance[1].tolist()],
                "cumulative": [round(v, 4) for v in variance[2].tolist()],
                "eigenvalues": [round(v, 4) for v in variance[0].tolist()],
            },
            "n_observations": len(subset),
        }
        self.results["factor_analysis"] = result
        return result

    # ================================================================
    # Chi-Square Test
    # ================================================================

    def chi_square_test(self, col1: str, col2: str) -> dict:
        """
        Chi-square test of independence for two categorical variables.

        Parameters
        ----------
        col1, col2 : str
            Categorical column names.

        Returns
        -------
        dict
            Chi-square statistic, p-value, Cramér's V.
        """
        contingency = pd.crosstab(self.data[col1], self.data[col2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)

        # Cramér's V
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        result = {
            "test": "Chi-Square Test of Independence",
            "chi2": round(chi2, 4),
            "p_value": round(p, 6),
            "dof": dof,
            "cramers_v": round(cramers_v, 4),
            "effect_size_interpretation": (
                "Large" if cramers_v >= 0.5 else
                "Medium" if cramers_v >= 0.3 else
                "Small" if cramers_v >= 0.1 else
                "Negligible"
            ),
            "significant": p < 0.05,
            "contingency_table": contingency,
                        "expected_frequencies": pd.DataFrame(
                np.round(expected, 2),
                index=contingency.index,
                columns=contingency.columns,
            ),
            "n_observations": int(n),
        }

        # Check expected frequency assumption
        pct_below_5 = (expected < 5).sum() / expected.size * 100
        if pct_below_5 > 20:
            result["warning"] = (
                f"{pct_below_5:.0f}% of expected frequencies are < 5. "
                "Consider using Fisher's exact test."
            )

        self.results["chi_square"] = result
        logger.info(
            f"Chi-Square: χ²={chi2:.4f}, p={p:.6f}, "
            f"Cramér's V={cramers_v:.4f}"
        )
        return result

    # ================================================================
    # Proportion Tests
    # ================================================================

    def proportion_test(
        self,
        column: str,
        value: any,
        group_col: Optional[str] = None,
    ) -> dict:
        """
        Test proportions — one-sample or two-sample z-test.

        Parameters
        ----------
        column : str
            Column to test proportions on.
        value : any
            The value to count as a "success."
        group_col : str, optional
            If provided, compares proportions between two groups.

        Returns
        -------
        dict
            Test results.
        """
        from statsmodels.stats.proportion import proportions_ztest

        if group_col is None:
            # One-sample proportion test (against 0.5)
            n_total = self.data[column].notna().sum()
            n_success = (self.data[column] == value).sum()
            stat, p = proportions_ztest(n_success, n_total, value=0.5)
            result = {
                "test": "One-sample z-test for proportion",
                "proportion": round(n_success / n_total, 4),
                "n": int(n_total),
                "statistic": round(stat, 4),
                "p_value": round(p, 6),
                "significant": p < 0.05,
            }
        else:
            # Two-sample proportion test
            groups = self.data.groupby(group_col)
            group_names = list(groups.groups.keys())[:2]
            counts = []
            totals = []
            for name in group_names:
                grp = groups.get_group(name)
                totals.append(grp[column].notna().sum())
                counts.append((grp[column] == value).sum())

            stat, p = proportions_ztest(counts, totals)
            result = {
                "test": "Two-sample z-test for proportions",
                "groups": group_names,
                "proportions": {
                    str(name): round(c / t, 4)
                    for name, c, t in zip(group_names, counts, totals)
                },
                "statistic": round(stat, 4),
                "p_value": round(p, 6),
                "significant": p < 0.05,
            }

        self.results["proportion_test"] = result
        return result

    # ================================================================
    # Descriptive by Group
    # ================================================================

    def descriptives_by_group(
        self,
        variables: list[str],
        group_col: str,
    ) -> pd.DataFrame:
        """
        Compute descriptive statistics for variables by group.

        Parameters
        ----------
        variables : list[str]
            Numeric variables to summarize.
        group_col : str
            Grouping variable.

        Returns
        -------
        pd.DataFrame
            Multi-indexed descriptive statistics table.
        """
        result = self.data.groupby(group_col)[variables].agg(
            ["count", "mean", "median", "std", "min", "max"]
        ).round(4)
        self.results["descriptives_by_group"] = result
        return result

    # ================================================================
    # Summary Export
    # ================================================================

    def get_all_results(self) -> dict:
        """Return all stored results."""
        return self.results

    def __repr__(self) -> str:
        return (
            f"SurveyStats(rows={len(self.data)}, "
            f"cols={len(self.data.columns)}, "
            f"analyses_run={len(self.results)})"
        )