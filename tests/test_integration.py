"""
End-to-end integration tests for the Survey ML Toolkit.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")


class TestFullPipeline:
    """Test complete analysis pipelines from data loading to reporting."""

    @pytest.mark.slow
    def test_full_survey_pipeline(self, tmp_dir):
        """
        End-to-end test: generate → load → clean → EDA →
        stats → ML → segment → report
        """
        import matplotlib.pyplot as plt

        from survey_toolkit.utils import generate_sample_survey
        from survey_toolkit.loader import SurveyLoader
        from survey_toolkit.cleaner import SurveyCleaner
        from survey_toolkit.eda import SurveyEDA
        from survey_toolkit.stats import SurveyStats
        from survey_toolkit.ml_models import SurveyClassifier, SurveySegmentation
        from survey_toolkit.reporting import ReportGenerator

        # 1. Generate sample data
        csv_path = str(tmp_dir / "survey.csv")
        generate_sample_survey(
            n_respondents=200,
            n_likert_items=10,
            save_path=csv_path,
        )
        assert Path(csv_path).exists()

        # 2. Load
        loader = SurveyLoader(csv_path)
        df = loader.load()
        assert len(df) == 200
        assert loader.metadata["n_respondents"] == 200

        # 3. Clean
        likert_cols = [f"q{i}" for i in range(1, 11)]
        cleaner = SurveyCleaner(df)
        clean_df = (
            cleaner
            .remove_speeders("duration_seconds", min_seconds=60)
            .remove_straightliners(likert_cols, threshold=0.95)
            .handle_missing(strategy="median")
            .get_clean_data()
        )
        assert len(clean_df) > 0
        assert len(cleaner.get_log()) == 3

        # 4. EDA
        eda = SurveyEDA(clean_df, output_dir=str(tmp_dir / "figures"))
        summary = eda.response_summary()
        assert isinstance(summary, pd.DataFrame)

        eda.plot_likert_distribution(likert_cols[:3], save=True)
        eda.plot_correlation_heatmap(likert_cols[:5], save=True)
        eda.missing_data_report(save=True)
        plt.close("all")
        assert len(eda.figures) >= 2

        # 5. Statistics
        stats = SurveyStats(clean_df)

        alpha = stats.cronbachs_alpha(likert_cols)
        assert alpha["alpha"] > 0
        assert alpha["n_items"] == 10

        comparison = stats.compare_groups("q1", "age_group")
        assert "p_value" in comparison

        corr = stats.correlation_matrix(likert_cols[:5])
        assert corr["correlation_matrix"].shape == (5, 5)

        chi2 = stats.chi_square_test("age_group", "gender")
        assert "chi2" in chi2

        fa = stats.factor_analysis(likert_cols, n_factors=2)
        assert fa["n_factors"] == 2
        assert fa["loadings"].shape == (10, 2)

        # 6. Classification
        classifier = SurveyClassifier(clean_df)
        X, y = classifier.prepare_data(
            feature_cols=likert_cols,
            target_col="satisfaction_group",
        )
        results = classifier.run_model_comparison(cv_folds=3)
        assert len(results) >= 3
        assert classifier.best_model is not None

        report_dict = classifier.get_classification_report()
        assert "confusion_matrix" in report_dict

        preds = classifier.predict(clean_df[likert_cols].head(5))
        assert len(preds) == 5

        # 7. Segmentation
        seg = SurveySegmentation(clean_df)
        seg.prepare_data(likert_cols)
        optimal = seg.find_optimal_k(k_range=range(2, 5))
        assert optimal["optimal_k"] >= 2

        profiles = seg.fit_clusters(n_clusters=3)
        assert len(profiles) == 3

        labels = seg.get_cluster_labels()
        assert len(labels) > 0

        viz = seg.visualize_clusters()
        assert "PC1" in viz.columns

        demo_profiles = seg.profile_clusters_by_demographics(
            ["age_group", "gender"]
        )
        assert len(demo_profiles) > 0

        # 8. Report
        report = ReportGenerator(clean_df)
        report.set_metadata(
            title="Integration Test Report",
            author="Test Suite",
        )
        report.add_summary_statistics(likert_cols)
        report.add_stats_result("Reliability", alpha)
        report.add_stats_result("Group Comparison", comparison)
        report.add_dataframe("Correlations", corr["correlation_matrix"])
        report.add_dataframe("Model Comparison", results)
        report.add_dataframe("Cluster Profiles", profiles)

        for fig_path in eda.figures:
            if Path(fig_path).exists():
                report.add_figure(
                    f"Figure: {Path(fig_path).stem}", fig_path
                )

        report_path = str(tmp_dir / "full_report.html")
        output = report.generate(
            output_path=report_path, auto_sections=False
        )

        assert Path(output).exists()
        content = Path(output).read_text()
        assert "Integration Test Report" in content
        assert "Reliability" in content
        assert "Model Comparison" in content

        plt.close("all")

    def test_minimal_pipeline(self, tmp_dir):
        """Test minimum viable pipeline: load → clean → stats."""
        from survey_toolkit.utils import generate_sample_survey
        from survey_toolkit.loader import SurveyLoader
        from survey_toolkit.cleaner import SurveyCleaner
        from survey_toolkit.stats import SurveyStats

        csv_path = str(tmp_dir / "minimal.csv")
        generate_sample_survey(
            n_respondents=50,
            n_likert_items=5,
            save_path=csv_path,
        )

        df = SurveyLoader(csv_path).load()
        clean_df = (
            SurveyCleaner(df)
            .handle_missing(strategy="median")
            .get_clean_data()
        )

        stats = SurveyStats(clean_df)
        alpha = stats.cronbachs_alpha([f"q{i}" for i in range(1, 6)])
        assert alpha["alpha"] > 0

    def test_import_all_public_api(self):
        """Verify all public API imports work."""
        from survey_toolkit import (
            SurveyLoader,
            SurveyCleaner,
            SurveyEDA,
            SurveyStats,
            SurveyClassifier,
            SurveySegmentation,
            ReportGenerator,
            generate_sample_survey,
            detect_column_types,
            timer,
            logger,
        )

        assert callable(SurveyLoader)
        assert callable(SurveyCleaner)
        assert callable(SurveyEDA)
        assert callable(SurveyStats)
        assert callable(SurveyClassifier)
        assert callable(SurveySegmentation)
        assert callable(ReportGenerator)
        assert callable(generate_sample_survey)
        assert callable(detect_column_types)
        assert callable(timer)

    def test_version_info(self):
        """Verify version metadata is accessible."""
        import survey_toolkit

        assert hasattr(survey_toolkit, "__version__")
        assert hasattr(survey_toolkit, "__title__")
        assert hasattr(survey_toolkit, "__license__")
        assert survey_toolkit.__license__ == "MIT"

    def test_chaining_across_modules(self, tmp_dir):
        """Test that data flows cleanly between modules."""
        from survey_toolkit.utils import generate_sample_survey
        from survey_toolkit.cleaner import SurveyCleaner
        from survey_toolkit.stats import SurveyStats
        from survey_toolkit.utils import detect_column_types, compute_scale_scores

        df = generate_sample_survey(n_respondents=100, n_likert_items=6)

        # Detect types
        types = detect_column_types(df)
        likert_cols = types["likert"]
        assert len(likert_cols) == 6

        # Clean
        clean_df = (
            SurveyCleaner(df)
            .remove_speeders("duration_seconds")
            .handle_missing(strategy="median")
            .get_clean_data()
        )

        # Compute scale scores
        construct_map = {
            "construct_a": likert_cols[:3],
            "construct_b": likert_cols[3:],
        }
        scores = compute_scale_scores(clean_df, construct_map)
        assert "construct_a" in scores.columns
        assert "construct_b" in scores.columns

        # Run stats on scale scores
        merged = pd.concat([clean_df, scores], axis=1)
        stats = SurveyStats(merged)
        corr = stats.correlation_matrix(["construct_a", "construct_b"])
        assert corr["correlation_matrix"].shape == (2, 2)

    def test_edge_case_all_same_responses(self):
        """Test handling when all respondents give the same answer."""
        from survey_toolkit.cleaner import SurveyCleaner
        from survey_toolkit.stats import SurveyStats

        df = pd.DataFrame({
            "q1": [3] * 50,
            "q2": [3] * 50,
            "q3": [3] * 50,
            "group": ["A"] * 25 + ["B"] * 25,
        })

        clean = SurveyCleaner(df).get_clean_data()
        stats = SurveyStats(clean)

        # Alpha should be low/undefined for zero-variance items
        alpha = stats.cronbachs_alpha(["q1", "q2", "q3"])
        # Zero variance means alpha calculation may produce nan or 0
        assert isinstance(alpha["alpha"], float)

    def test_edge_case_many_missing(self, tmp_dir):
        """Test handling of heavily missing data."""
        from survey_toolkit.cleaner import SurveyCleaner
        from survey_toolkit.utils import validate_survey_data

        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame({
            f"q{i}": [
                rng.choice([1, 2, 3, 4, 5, np.nan], p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
                for _ in range(n)
            ]
            for i in range(1, 6)
        })

        # Validation should flag warnings
        validation = validate_survey_data(df, max_missing_pct=30)
        assert len(validation["warnings"]) > 0

        # Cleaning with median should handle it
        clean = (
            SurveyCleaner(df)
            .handle_missing(strategy="median")
            .get_clean_data()
        )
        for col in clean.columns:
            assert clean[col].isna().sum() == 0

    def test_edge_case_single_respondent(self):
        """Test with minimum data."""
        from survey_toolkit.utils import validate_survey_data

        df = pd.DataFrame({"q1": [3], "q2": [4]})
        result = validate_survey_data(df, min_respondents=30)
        assert result["valid"] is False