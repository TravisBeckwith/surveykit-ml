"""
Tests for survey_toolkit.ml_models module.
"""

import pytest
import pandas as pd
import numpy as np


class TestSurveyClassifier:
    """Tests for the SurveyClassifier class."""

    def test_prepare_data(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        X, y = clf.prepare_data(
            feature_cols=likert_columns,
            target_col="satisfaction_group",
        )

        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(y)
        assert len(X) > 0
        assert clf.feature_names is not None

    def test_prepare_data_encodes_target(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        X, y = clf.prepare_data(
            feature_cols=likert_columns,
            target_col="satisfaction_group",
        )
        assert y.dtype in [np.int64, np.int32, int]
        assert clf.label_encoder is not None

    def test_prepare_data_numeric_target(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveyClassifier

        # Use NPS score (numeric) as target
        df = clean_survey_df.copy()
        df["nps_binary"] = (df["nps_score"] >= 7).astype(int)

        clf = SurveyClassifier(df)
        X, y = clf.prepare_data(
            feature_cols=likert_columns,
            target_col="nps_binary",
        )
        assert clf.label_encoder is None  # Shouldn't encode numeric target

    def test_prepare_data_drops_na(self, sample_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(sample_survey_df)
        X, y = clf.prepare_data(
            feature_cols=likert_columns,
            target_col="satisfaction_group",
        )
        assert X.isna().sum().sum() == 0
        assert y.isna().sum() == 0

    @pytest.mark.slow
    def test_model_comparison(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        clf.prepare_data(
            feature_cols=likert_columns,
            target_col="satisfaction_group",
        )
        results = clf.run_model_comparison(cv_folds=3)

        assert isinstance(results, pd.DataFrame)
        assert "model" in results.columns
        assert "mean_score" in results.columns
        assert "std_score" in results.columns
        assert len(results) >= 3  # At least 3 models
        assert clf.best_model is not None

    @pytest.mark.slow
    def test_model_comparison_scores_valid(
        self, clean_survey_df, likert_columns
    ):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        clf.prepare_data(
            feature_cols=likert_columns,
            target_col="satisfaction_group",
        )
        results = clf.run_model_comparison(cv_folds=3)

        for _, row in results.iterrows():
            assert 0 <= row["mean_score"] <= 1
            assert row["std_score"] >= 0

    @pytest.mark.slow
    def test_classification_report(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        clf.prepare_data(
            feature_cols=likert_columns,
            target_col="satisfaction_group",
        )
        clf.run_model_comparison(cv_folds=3)
        report = clf.get_classification_report()

        assert "classification_report" in report
        assert "confusion_matrix" in report
        assert "target_names" in report
        assert isinstance(report["confusion_matrix"], list)

    @pytest.mark.slow
    def test_feature_importance(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        clf.prepare_data(
            feature_cols=likert_columns,
            target_col="satisfaction_group",
        )
        clf.run_model_comparison(cv_folds=3)

        try:
            importance = clf.feature_importance()
            assert isinstance(importance, pd.DataFrame)
            assert "feature" in importance.columns
            assert "mean_shap_value" in importance.columns
            assert "rank" in importance.columns
            assert len(importance) == len(clf.feature_names)
            assert importance["rank"].iloc[0] == 1
        except ImportError:
            pytest.skip("SHAP not installed")

    @pytest.mark.slow
    def test_hyperparameter_tune(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        clf.prepare_data(
            feature_cols=likert_columns,
            target_col="satisfaction_group",
        )
        clf.run_model_comparison(cv_folds=3)

        result = clf.hyperparameter_tune(
            model_name="Random Forest",
            param_grid={
                "model__n_estimators": [50, 100],
                "model__max_depth": [None, 10],
            },
            cv_folds=3,
        )

        assert "best_params" in result
        assert "best_score" in result
        assert result["best_score"] > 0

    @pytest.mark.slow
    def test_hyperparameter_tune_invalid_model(
        self, clean_survey_df, likert_columns
    ):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        clf.prepare_data(
            feature_cols=likert_columns,
            target_col="satisfaction_group",
        )
        clf.run_model_comparison(cv_folds=3)

        with pytest.raises(ValueError, match="not found"):
            clf.hyperparameter_tune(model_name="NonexistentModel")

    @pytest.mark.slow
    def test_predict(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        clf.prepare_data(
            feature_cols=likert_columns,
            target_col="satisfaction_group",
        )
        clf.run_model_comparison(cv_folds=3)

        # Predict on a subset
        new_data = clean_survey_df[likert_columns].head(10)
        preds = clf.predict(new_data)

        assert len(preds) == 10
        assert all(
            p in ["Dissatisfied", "Neutral", "Satisfied"] for p in preds
        )

    @pytest.mark.slow
    def test_predict_proba(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        clf.prepare_data(
            feature_cols=likert_columns,
            target_col="satisfaction_group",
        )
        clf.run_model_comparison(cv_folds=3)

        new_data = clean_survey_df[likert_columns].head(5)
        proba = clf.predict(new_data, return_proba=True)

        assert proba.shape[0] == 5
        assert proba.shape[1] >= 2  # At least 2 classes
        # Probabilities should sum to ~1
        for row in proba:
            assert abs(sum(row) - 1.0) < 1e-6

    def test_predict_before_training_raises(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        with pytest.raises(ValueError, match="No model trained"):
            clf.predict(clean_survey_df[likert_columns].head(5))

    def test_classification_report_before_training_raises(
        self, clean_survey_df
    ):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        with pytest.raises(ValueError, match="No model found"):
            clf.get_classification_report()

    def test_prepare_data_before_comparison_raises(self, clean_survey_df):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        with pytest.raises(ValueError, match="Call prepare_data"):
            clf.run_model_comparison()

    def test_repr(self, clean_survey_df):
        from survey_toolkit.ml_models import SurveyClassifier

        clf = SurveyClassifier(clean_survey_df)
        repr_str = repr(clf)
        assert "SurveyClassifier" in repr_str


class TestSurveySegmentation:
    """Tests for the SurveySegmentation class."""

    def test_prepare_data(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        X = seg.prepare_data(likert_columns)

        assert X.shape[0] > 0
        assert X.shape[1] == len(likert_columns)
        assert seg.feature_names == likert_columns

    def test_prepare_data_scaled(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        X = seg.prepare_data(likert_columns, scale=True)

        # Scaled data should have mean ~0, std ~1
        assert abs(X.mean()) < 0.5
        assert seg.scaler is not None

    def test_prepare_data_unscaled(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        X = seg.prepare_data(likert_columns, scale=False)
        assert seg.scaler is None

    @pytest.mark.slow
    def test_find_optimal_k(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        seg.prepare_data(likert_columns)
        result = seg.find_optimal_k(k_range=range(2, 6))

        assert "optimal_k" in result
        assert "silhouette_scores" in result
        assert "inertias" in result
        assert "calinski_harabasz_scores" in result
        assert 2 <= result["optimal_k"] <= 5
        assert len(result["silhouette_scores"]) == 4

    def test_find_optimal_k_before_prepare_raises(self, clean_survey_df):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        with pytest.raises(ValueError, match="Call prepare_data"):
            seg.find_optimal_k()

    @pytest.mark.slow
    def test_fit_clusters(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        seg.prepare_data(likert_columns)
        profiles = seg.fit_clusters(n_clusters=3)

        assert isinstance(profiles, pd.DataFrame)
        assert len(profiles) == 3
        assert list(profiles.columns) == likert_columns
        assert seg.labels is not None
        assert "cluster_sizes" in seg.results
        assert "silhouette" in seg.results
        assert sum(seg.results["cluster_sizes"].values()) == seg.X.shape[0]

    def test_fit_clusters_before_prepare_raises(self, clean_survey_df):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        with pytest.raises(ValueError, match="Call prepare_data"):
            seg.fit_clusters(n_clusters=3)

    @pytest.mark.slow
    def test_get_cluster_labels(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        seg.prepare_data(likert_columns)
        seg.fit_clusters(n_clusters=3)

        labels = seg.get_cluster_labels()
        assert isinstance(labels, pd.Series)
        assert labels.name == "cluster"
        assert set(labels.unique()).issubset({0, 1, 2})

    def test_get_cluster_labels_before_fit_raises(
        self, clean_survey_df, likert_columns
    ):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        seg.prepare_data(likert_columns)
        with pytest.raises(ValueError, match="Call fit_clusters"):
            seg.get_cluster_labels()

    @pytest.mark.slow
    def test_visualize_clusters(self, clean_survey_df, likert_columns):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        seg.prepare_data(likert_columns)
        seg.fit_clusters(n_clusters=3)
        viz = seg.visualize_clusters()

        assert isinstance(viz, pd.DataFrame)
        assert "PC1" in viz.columns
        assert "PC2" in viz.columns
        assert "cluster" in viz.columns
        assert len(viz) == seg.X.shape[0]
        assert "pca_variance" in seg.results
        assert len(seg.results["pca_variance"]) == 2

    def test_visualize_before_fit_raises(
        self, clean_survey_df, likert_columns
    ):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        seg.prepare_data(likert_columns)
        with pytest.raises(ValueError, match="Call fit_clusters"):
            seg.visualize_clusters()

    @pytest.mark.slow
    def test_profile_by_demographics(
        self, clean_survey_df, likert_columns, demographic_columns
    ):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        seg.prepare_data(likert_columns)
        seg.fit_clusters(n_clusters=3)

        profiles = seg.profile_clusters_by_demographics(demographic_columns)

        assert isinstance(profiles, dict)
        for col in demographic_columns:
            if col in clean_survey_df.columns:
                assert col in profiles
                assert isinstance(profiles[col], pd.DataFrame)
                # Rows should be normalized (sum to ~1)
                row_sums = profiles[col].sum(axis=1)
                for s in row_sums:
                    assert abs(s - 1.0) < 0.01

    def test_profile_before_fit_raises(
        self, clean_survey_df, likert_columns
    ):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        seg.prepare_data(likert_columns)
        with pytest.raises(ValueError, match="Call fit_clusters"):
            seg.profile_clusters_by_demographics(["age_group"])

    def test_repr(self, clean_survey_df):
        from survey_toolkit.ml_models import SurveySegmentation

        seg = SurveySegmentation(clean_survey_df)
        repr_str = repr(seg)
        assert "SurveySegmentation" in repr_str