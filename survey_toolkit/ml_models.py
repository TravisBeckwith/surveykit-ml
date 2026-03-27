"""
Machine Learning models tailored for survey data analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report,
    silhouette_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from typing import Optional
from survey_toolkit.utils import logger, timer


class SurveyClassifier:
    """
    Classification models for predicting survey outcomes
    (e.g., churn, satisfaction level, segment membership).
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = None
        self.scaler = None
        self.feature_names = None
        self.X = None
        self.y = None
        self.results = {}

    def prepare_data(
        self,
        feature_cols: list[str],
        target_col: str,
        scale: bool = True,
    ) -> tuple:
        """
        Prepare features and target for modeling.

        Parameters
        ----------
        feature_cols : list[str]
            Feature column names.
        target_col : str
            Target variable column name.
        scale : bool
            Whether to scale features.

        Returns
        -------
        tuple
            (X, y) DataFrames.
        """
        subset = self.data[feature_cols + [target_col]].dropna()
        X = subset[feature_cols]
        y = subset[target_col]

        # Encode target if categorical
        if y.dtype == "object" or y.dtype.name == "category":
            self.label_encoder = LabelEncoder()
            y = pd.Series(
                self.label_encoder.fit_transform(y),
                name=target_col,
                index=y.index,
            )
            logger.info(
                f"Encoded target classes: "
                f"{dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}"
            )

        # Encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        self.feature_names = X.columns.tolist()
        self.X = X
        self.y = y
        self.scaler = StandardScaler() if scale else None

        logger.info(
            f"Prepared data: {X.shape[0]} samples, "
            f"{X.shape[1]} features, "
            f"{y.nunique()} classes"
        )
        return X, y

    @timer
    def run_model_comparison(
        self,
        cv_folds: int = 5,
        scoring: str = "accuracy",
    ) -> pd.DataFrame:
        """
        Compare multiple classifiers using cross-validation.

        Parameters
        ----------
        cv_folds : int
            Number of cross-validation folds.
        scoring : str
            Scoring metric (e.g., 'accuracy', 'f1_macro', 'roc_auc').

        Returns
        -------
        pd.DataFrame
            Model comparison results sorted by performance.
        """
        if self.X is None or self.y is None:
            raise ValueError("Call prepare_data() first.")

        # Try to import xgboost, skip if not available
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
        }

        try:
            import xgboost as xgb
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric="mlogloss",
            )
        except ImportError:
            logger.warning(
                "XGBoost not installed. Skipping. "
                "Install with: pip install survey-ml-toolkit[ml]"
            )

        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=42
        )
        results = []

        for name, model in models.items():
            logger.info(f"Training {name}...")
            pipeline_steps = []
            if self.scaler:
                pipeline_steps.append(("scaler", StandardScaler()))
            pipeline_steps.append(("model", model))
            pipeline = Pipeline(pipeline_steps)

            try:
                scores = cross_val_score(
                    pipeline, self.X, self.y, cv=cv, scoring=scoring
                )
                results.append({
                    "model": name,
                    "mean_score": round(scores.mean(), 4),
                    "std_score": round(scores.std(), 4),
                    "min_score": round(scores.min(), 4),
                    "max_score": round(scores.max(), 4),
                    "scores": scores.tolist(),
                })
                self.models[name] = pipeline
                logger.info(
                    f"  {name}: {scores.mean():.4f} "
                    f"(+/- {scores.std():.4f})"
                )
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")

        results_df = pd.DataFrame(results).sort_values(
            "mean_score", ascending=False
        )
        self.results["model_comparison"] = results_df

        # Set best model
        if not results_df.empty:
            best_name = results_df.iloc[0]["model"]
            self.best_model_name = best_name
            self.best_model = self.models[best_name]
            self.best_model.fit(self.X, self.y)
            logger.info(f"Best model: {best_name}")

        return results_df

    def get_classification_report(
        self,
        model_name: Optional[str] = None,
    ) -> dict:
        """
        Generate a full classification report for a fitted model.

        Parameters
        ----------
        model_name : str, optional
            Model name. Uses best model if None.

        Returns
        -------
        dict
            Classification report with confusion matrix.
        """
        model = (
            self.models.get(model_name)
            if model_name
            else self.best_model
        )
        if model is None:
            raise ValueError("No model found. Run run_model_comparison() first.")

        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)

        report = classification_report(self.y, y_pred, output_dict=True)
        cm = confusion_matrix(self.y, y_pred)

        # Add class labels if available
        if self.label_encoder:
            target_names = list(self.label_encoder.classes_)
        else:
            target_names = [str(c) for c in sorted(self.y.unique())]

        result = {
            "model": model_name or self.best_model_name,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "target_names": target_names,
        }
        self.results["classification_report"] = result
        return result

    @timer
    def feature_importance(
        self,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get feature importance using SHAP values.

        Parameters
        ----------
        model_name : str, optional
            Model to explain. Uses best model if None.

        Returns
        -------
        pd.DataFrame
            Features ranked by importance.
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is required for feature importance. "
                "Install with: pip install survey-ml-toolkit[ml]"
            )

        model = (
            self.models.get(model_name)
            if model_name
            else self.best_model
        )
        if model is None:
            raise ValueError("No model found. Run run_model_comparison() first.")

        model.fit(self.X, self.y)
        actual_model = model.named_steps["model"]

        # Prepare transformed data
        if self.scaler and "scaler" in model.named_steps:
            X_transformed = model.named_steps["scaler"].transform(self.X)
        else:
            X_transformed = self.X.values

        # Select appropriate SHAP explainer
        if hasattr(actual_model, "feature_importances_"):
            explainer = shap.TreeExplainer(actual_model)
        else:
            explainer = shap.LinearExplainer(actual_model, X_transformed)

        shap_values = explainer.shap_values(X_transformed)

        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            mean_shap = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_shap_value": mean_shap,
        }).sort_values("mean_shap_value", ascending=False).reset_index(drop=True)

        # Add rank
        importance_df["rank"] = range(1, len(importance_df) + 1)

        self.results["feature_importance"] = importance_df
        logger.info(
            f"Top 5 features: "
            f"{importance_df['feature'].head().tolist()}"
        )
        return importance_df

    @timer
    def hyperparameter_tune(
        self,
        model_name: str = "Random Forest",
        param_grid: Optional[dict] = None,
        cv_folds: int = 5,
        scoring: str = "accuracy",
    ) -> dict:
        """
        Grid search hyperparameter tuning.

        Parameters
        ----------
        model_name : str
            Name of the model to tune.
        param_grid : dict, optional
            Parameter grid. Uses defaults if None.
        cv_folds : int
            Number of CV folds.
        scoring : str
            Scoring metric.

        Returns
        -------
        dict
            Best parameters and score.
        """
        default_grids = {
            "Logistic Regression": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l1", "l2"],
                "model__solver": ["saga"],
            },
            "Random Forest": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            },
            "Gradient Boosting": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.2],
            },
            "XGBoost": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.2],
            },
        }

        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(self.models.keys())}"
            )

        param_grid = param_grid or default_grids.get(model_name, {})
        pipeline = self.models[model_name]

        logger.info(f"Tuning {model_name} with {len(param_grid)} param sets...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(self.X, self.y)

        result = {
            "model": model_name,
            "best_params": grid_search.best_params_,
            "best_score": round(grid_search.best_score_, 4),
            "cv_results_summary": pd.DataFrame(
                grid_search.cv_results_
            )[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
            .sort_values("rank_test_score")
            .head(10)
            .to_dict("records"),
        }

        self.best_model = grid_search.best_estimator_
        self.best_model_name = f"{model_name} (tuned)"
        self.models[self.best_model_name] = self.best_model

        logger.info(
            f"Best score: {result['best_score']}, "
            f"Params: {result['best_params']}"
        )
        self.results["hyperparameter_tuning"] = result
        return result

    def predict(
        self,
        new_data: pd.DataFrame,
        return_proba: bool = False,
    ) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        new_data : pd.DataFrame
            New survey data with same feature columns.
        return_proba : bool
            Return class probabilities instead of labels.

        Returns
        -------
        np.ndarray
            Predictions or probabilities.
        """
        if self.best_model is None:
            raise ValueError("No model trained. Run run_model_comparison() first.")

        # Match feature columns
        X_new = pd.get_dummies(new_data, drop_first=True)
        missing_cols = set(self.feature_names) - set(X_new.columns)
        for col in missing_cols:
            X_new[col] = 0
        X_new = X_new[self.feature_names]

        if return_proba:
            proba = self.best_model.predict_proba(X_new)
            return proba
        else:
            preds = self.best_model.predict(X_new)
            if self.label_encoder:
                preds = self.label_encoder.inverse_transform(preds)
            return preds

    def __repr__(self) -> str:
        return (
            f"SurveyClassifier(samples={len(self.data)}, "
            f"models_trained={len(self.models)}, "
            f"best={self.best_model_name})"
        )


class SurveySegmentation:
    """
    Unsupervised segmentation / clustering of survey respondents.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = None
        self.feature_names = None
        self.subset_index = None
        self.X = None
        self.labels = None
        self.results = {}

    def prepare_data(
        self,
        feature_cols: list[str],
        scale: bool = True,
    ) -> np.ndarray:
        """
        Prepare and scale features for clustering.

        Parameters
        ----------
        feature_cols : list[str]
            Feature columns to cluster on.
        scale : bool
            Whether to standardize features.

        Returns
        -------
        np.ndarray
            Prepared feature matrix.
        """
        subset = self.data[feature_cols].dropna()
        self.feature_names = feature_cols
        self.subset_index = subset.index

        if scale:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(subset)
        else:
            self.X = subset.values

        logger.info(
            f"Prepared {self.X.shape[0]} respondents, "
            f"{self.X.shape[1]} features for clustering"
        )
        return self.X

    @timer
    def find_optimal_k(
        self,
        k_range: range = range(2, 11),
    ) -> dict:
        """
        Find optimal number of clusters using
        silhouette score and inertia (elbow method).

        Parameters
        ----------
        k_range : range
            Range of k values to test.

        Returns
        -------
        dict
            Optimal k, silhouette scores, inertias.
        """
        if self.X is None:
            raise ValueError("Call prepare_data() first.")

        silhouette_scores = []
        inertias = []
        calinski_scores = []

        from sklearn.metrics import calinski_harabasz_score

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X)
            sil = silhouette_score(self.X, labels)
            cal = calinski_harabasz_score(self.X, labels)
            silhouette_scores.append(round(sil, 4))
            inertias.append(round(kmeans.inertia_, 2))
            calinski_scores.append(round(cal, 2))
            logger.info(
                f"  k={k}: silhouette={sil:.4f}, "
                f"inertia={kmeans.inertia_:.0f}"
            )

        optimal_k = list(k_range)[np.argmax(silhouette_scores)]

        result = {
            "k_range": list(k_range),
            "silhouette_scores": silhouette_scores,
            "inertias": inertias,
            "calinski_harabasz_scores": calinski_scores,
            "optimal_k": optimal_k,
            "best_silhouette": max(silhouette_scores),
        }
        self.results["optimal_k"] = result
        logger.info(f"Optimal k = {optimal_k} (silhouette = {max(silhouette_scores):.4f})")
        return result

    @timer
    def fit_clusters(
        self,
        n_clusters: int,
    ) -> pd.DataFrame:
        """
        Fit KMeans and return cluster profiles.

        Parameters
        ----------
        n_clusters : int
            Number of clusters.

        Returns
        -------
        pd.DataFrame
            Mean feature values per cluster.
        """
        if self.X is None:
            raise ValueError("Call prepare_data() first.")

        kmeans = KMeans(
            n_clusters=n_clusters, random_state=42, n_init=10
        )
        self.labels = kmeans.fit_predict(self.X)

        # Build cluster profiles
        profile_df = self.data.loc[self.subset_index].copy()
        profile_df["cluster"] = self.labels

        # Mean profiles
        cluster_means = profile_df.groupby("cluster")[
            self.feature_names
        ].mean().round(4)

        # Standard deviation profiles
        cluster_stds = profile_df.groupby("cluster")[
            self.feature_names
        ].std().round(4)

        # Cluster sizes
        sizes = pd.Series(self.labels).value_counts().sort_index()

        self.results["cluster_profiles"] = cluster_means
        self.results["cluster_stds"] = cluster_stds
        self.results["cluster_sizes"] = sizes.to_dict()
        self.results["n_clusters"] = n_clusters
        self.results["silhouette"] = round(
            silhouette_score(self.X, self.labels), 4
        )

        logger.info(
            f"Fit {n_clusters} clusters. "
            f"Silhouette: {self.results['silhouette']:.4f}. "
            f"Sizes: {sizes.to_dict()}"
        )
        return cluster_means

    def get_cluster_labels(self) -> pd.Series:
        """Return cluster labels aligned with the original data index."""
        if self.labels is None:
            raise ValueError("Call fit_clusters() first.")
        return pd.Series(
            self.labels, index=self.subset_index, name="cluster"
        )

    def visualize_clusters(self) -> pd.DataFrame:
        """
        Reduce to 2D with PCA for visualization.

        Returns
        -------
        pd.DataFrame
            DataFrame with PC1, PC2, and cluster labels.
        """
        if self.labels is None:
            raise ValueError("Call fit_clusters() first.")

        pca = PCA(n_components=2)
        coords = pca.fit_transform(self.X)

        viz_df = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "cluster": self.labels,
        })

        self.results["pca_variance"] = [
            round(v, 4) for v in pca.explained_variance_ratio_.tolist()
        ]
        logger.info(
            f"PCA variance explained: "
            f"PC1={self.results['pca_variance'][0]:.1%}, "
            f"PC2={self.results['pca_variance'][1]:.1%}"
        )
        return viz_df

    def profile_clusters_by_demographics(
        self,
        demographic_cols: list[str],
    ) -> dict:
        """
        Profile clusters by demographic variables.

        Parameters
        ----------
        demographic_cols : list[str]
            Demographic columns to cross-tabulate.

        Returns
        -------
        dict
            Cross-tabulations of clusters by demographics.
        """
        if self.labels is None:
            raise ValueError("Call fit_clusters() first.")

        profile_df = self.data.loc[self.subset_index].copy()
        profile_df["cluster"] = self.labels

        profiles = {}
        for col in demographic_cols:
            if col in profile_df.columns:
                ct = pd.crosstab(
                    profile_df["cluster"],
                    profile_df[col],
                    normalize="index",
                ).round(4)
                profiles[col] = ct

        self.results["demographic_profiles"] = profiles
        return profiles

    def __repr__(self) -> str:
        n_clusters = self.results.get("n_clusters", 0)
        return (
            f"SurveySegmentation(respondents={len(self.data)}, "
            f"clusters={n_clusters})"
        )