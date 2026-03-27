"""
Exploratory Data Analysis for survey data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from survey_toolkit.utils import logger, timer


class SurveyEDA:
    """Automated EDA for survey datasets."""

    def __init__(
        self,
        data: pd.DataFrame,
        output_dir: str = "outputs/figures",
        style: str = "whitegrid",
    ):
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style(style)
        self.figures = []

    def _save_figure(self, fig, name: str, save: bool = True):
        """Save figure and track it."""
        if save:
            path = self.output_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            self.figures.append(str(path))
            logger.info(f"Figure saved: {path}")

    def response_summary(self) -> pd.DataFrame:
        """Generate summary stats for all columns."""
        summary = []
        for col in self.data.columns:
            info = {
                "column": col,
                "dtype": str(self.data[col].dtype),
                "n_valid": int(self.data[col].notna().sum()),
                "n_missing": int(self.data[col].isna().sum()),
                "pct_missing": round(self.data[col].isna().mean() * 100, 2),
                "n_unique": int(self.data[col].nunique()),
            }
            if pd.api.types.is_numeric_dtype(self.data[col]):
                info.update({
                    "mean": round(self.data[col].mean(), 2),
                    "median": round(self.data[col].median(), 2),
                    "std": round(self.data[col].std(), 2),
                    "min": self.data[col].min(),
                    "max": self.data[col].max(),
                    "skew": round(self.data[col].skew(), 2),
                    "kurtosis": round(self.data[col].kurtosis(), 2),
                })
            summary.append(info)
        return pd.DataFrame(summary)

    def plot_likert_distribution(
        self,
        columns: list[str],
        labels: Optional[dict[str, str]] = None,
        colors: Optional[list[str]] = None,
        save: bool = True,
    ):
        """Plot horizontal stacked bar chart for Likert items."""
        colors = colors or [
            "#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"
        ]

        fig, axes = plt.subplots(
            len(columns), 1,
            figsize=(10, len(columns) * 1.2),
        )
        if len(columns) == 1:
            axes = [axes]

        for ax, col in zip(axes, columns):
            counts = self.data[col].value_counts().sort_index()
            pcts = counts / counts.sum() * 100
            left = 0
            for i, (val, pct) in enumerate(pcts.items()):
                ax.barh(
                    0, pct, left=left,
                    color=colors[i % len(colors)],
                    edgecolor="white",
                    label=f"{int(val)}" if ax == axes[0] else "",
                )
                if pct > 5:
                    ax.text(
                        left + pct / 2, 0, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=9,
                    )
                left += pct
            display_name = labels.get(col, col) if labels else col
            ax.set_xlim(0, 100)
            ax.set_yticks([0])
            ax.set_yticklabels([display_name[:50]])
            ax.set_xlabel("")

        if axes[0].get_legend_handles_labels()[1]:
            axes[0].legend(
                loc="upper right", title="Response",
                ncol=len(colors), fontsize=8,
            )

        plt.suptitle("Likert Scale Response Distribution", fontsize=14)
        plt.tight_layout()
        self._save_figure(fig, "likert_distribution", save)
        plt.show()

    def plot_correlation_heatmap(
        self,
        columns: list[str],
        method: str = "pearson",
        save: bool = True,
    ):
        """Generate a correlation heatmap."""
        corr = self.data[columns].corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
        )
        ax.set_title(f"{method.title()} Correlation Matrix", fontsize=14)
        plt.tight_layout()
        self._save_figure(fig, "correlation_heatmap", save)
        plt.show()

    def plot_demographic_breakdown(
        self,
        column: str,
        plot_type: str = "bar",
        save: bool = True,
    ):
        """Plot distribution of a demographic variable."""
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = self.data[column].value_counts()

        if plot_type == "bar":
            counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
            ax.set_ylabel("Count")
            # Add percentage labels
            total = counts.sum()
            for i, (val, count) in enumerate(counts.items()):
                ax.text(
                    i, count + total * 0.01,
                    f"{count / total * 100:.1f}%",
                    ha="center", fontsize=9,
                )
        elif plot_type == "pie":
            counts.plot(
                kind="pie", ax=ax, autopct="%1.1f%%",
                startangle=90, colors=sns.color_palette("Set2"),
            )
            ax.set_ylabel("")

        ax.set_title(f"Distribution: {column}", fontsize=14)
        plt.tight_layout()
        self._save_figure(fig, f"demographic_{column}", save)
        plt.show()

    def plot_response_by_group(
        self,
        value_col: str,
        group_col: str,
        plot_type: str = "box",
        save: bool = True,
    ):
        """Plot a response variable broken down by a grouping variable."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "box":
            sns.boxplot(
                data=self.data, x=group_col, y=value_col,
                ax=ax, palette="Set2",
            )
        elif plot_type == "violin":
            sns.violinplot(
                data=self.data, x=group_col, y=value_col,
                ax=ax, palette="Set2", inner="quartile",
            )
        elif plot_type == "strip":
            sns.stripplot(
                data=self.data, x=group_col, y=value_col,
                ax=ax, palette="Set2", alpha=0.5, jitter=True,
            )

        ax.set_title(f"{value_col} by {group_col}", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save_figure(fig, f"{value_col}_by_{group_col}", save)
        plt.show()

    def missing_data_report(self, save: bool = True):
        """Visualize missing data patterns."""
        missing = self.data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)

        if len(missing) == 0:
            logger.info("No missing data found.")
            print("✅ No missing data found.")
            return

        fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * 0.4)))
        missing.plot(kind="barh", ax=ax, color="coral", edgecolor="white")
        ax.set_xlabel("Number of Missing Values")
        ax.set_title("Missing Data by Column", fontsize=14)

        for i, v in enumerate(missing.values):
            pct = v / len(self.data) * 100
            ax.text(v + 0.5, i, f"{pct:.1f}%", va="center", fontsize=9)

        plt.tight_layout()
        self._save_figure(fig, "missing_data", save)
        plt.show()

    @timer
    def full_eda_report(
        self,
        likert_cols: Optional[list[str]] = None,
        demographic_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Run a full automated EDA pipeline."""
        logger.info("Running full EDA report...")

        # Summary table
        summary = self.response_summary()

        # Missing data
        self.missing_data_report()

        # Likert distributions
        if likert_cols:
            self.plot_likert_distribution(likert_cols)
            self.plot_correlation_heatmap(likert_cols)

        # Demographics
        if demographic_cols:
            for col in demographic_cols:
                self.plot_demographic_breakdown(col)

        logger.info(f"EDA complete. {len(self.figures)} figures generated.")
        return summary

    def __repr__(self) -> str:
        return (
            f"SurveyEDA(rows={len(self.data)}, "
            f"cols={len(self.data.columns)}, "
            f"figures={len(self.figures)})"
        )