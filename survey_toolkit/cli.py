"""
Command-line interface for the Survey ML Toolkit.
"""

import argparse
import sys
import json
from pathlib import Path


def main():
    """Main CLI entry point for survey analysis."""
    parser = argparse.ArgumentParser(
        prog="survey-analyze",
        description="Analyze survey data from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  survey-analyze data/survey.csv --eda
  survey-analyze data/survey.csv --alpha q1 q2 q3 q4 q5 --format json
  survey-analyze data/survey.csv --cluster q1 q2 q3 --format csv
  survey-analyze data/survey.csv --compare q1 --group age_group
  survey-analyze data/survey.csv --classify q1 q2 q3 --target satisfaction
        """,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the survey data file (CSV, XLSX, SPSS, Stata).",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run exploratory data analysis.",
    )
    parser.add_argument(
        "--stats",
        nargs="+",
        metavar="COL",
        help="Run statistical summary on specified columns.",
    )
    parser.add_argument(
        "--alpha",
        nargs="+",
        metavar="COL",
        help="Compute Cronbach's alpha for specified columns.",
    )
    parser.add_argument(
        "--compare",
        type=str,
        metavar="VAR",
        help="Variable to compare across groups.",
    )
    parser.add_argument(
        "--group",
        type=str,
        metavar="COL",
        help="Grouping variable for --compare.",
    )
    parser.add_argument(
        "--chi-square",
        nargs=2,
        metavar=("COL1", "COL2"),
        help="Run chi-square test on two categorical columns.",
    )
    parser.add_argument(
        "--correlations",
        nargs="+",
        metavar="COL",
        help="Compute correlation matrix for specified columns.",
    )
    parser.add_argument(
        "--factor",
        nargs="+",
        metavar="COL",
        help="Run factor analysis on specified columns.",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=None,
        help="Number of factors for factor analysis (auto if not specified).",
    )
    parser.add_argument(
        "--cluster",
        nargs="+",
        metavar="COL",
        help="Run respondent segmentation on specified columns.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters (auto-detected if not specified).",
    )
    parser.add_argument(
        "--classify",
        nargs="+",
        metavar="COL",
        help="Feature columns for classification.",
    )
    parser.add_argument(
        "--target",
        type=str,
        metavar="COL",
        help="Target column for classification.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output files (default: outputs/).",
    )
    parser.add_argument(
        "--missing-strategy",
        type=str,
        choices=["drop", "median", "mode", "fill", "interpolate"],
        default="median",
        help="Missing data handling strategy (default: median).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "print"],
        default="print",
        help="Output format (default: print).",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a sample survey dataset and exit.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of respondents for sample generation (default: 500).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    # Lazy imports so CLI starts fast
    from survey_toolkit.utils import generate_sample_survey, logger

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Generate sample data ----
    if args.generate_sample:
        logger.info(f"Generating sample survey with {args.sample_size} respondents...")
        save_path = str(output_dir / "sample_survey.csv")
        df = generate_sample_survey(
            n_respondents=args.sample_size,
            save_path=save_path,
        )
        print(f"✅ Sample survey saved to {save_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        return

    # ---- Load data ----
    from survey_toolkit.loader import SurveyLoader
    from survey_toolkit.cleaner import SurveyCleaner

    print(f"📥 Loading data from {args.input_file}...")
    loader = SurveyLoader(args.input_file)
    df = loader.load()
    print(
        f"   Loaded {loader.metadata['n_respondents']} respondents, "
        f"{loader.metadata['n_questions']} columns."
    )

    # ---- Clean ----
    print(f"🧹 Cleaning data (strategy: {args.missing_strategy})...")
    cleaner = SurveyCleaner(df)
    clean_df = (
        cleaner
        .handle_missing(strategy=args.missing_strategy)
        .get_clean_data()
    )
    print(f"   {len(clean_df)} respondents after cleaning.")

    # ---- EDA ----
    if args.eda:
        from survey_toolkit.eda import SurveyEDA

        print("📊 Running EDA...")
        eda = SurveyEDA(clean_df, output_dir=str(output_dir / "figures"))
        summary = eda.response_summary()
        _output(summary, args.format, output_dir / "eda_summary")
        eda.missing_data_report(save=True)
        print("   EDA complete. Figures saved.")

    # ---- Cronbach's Alpha ----
    if args.alpha:
        from survey_toolkit.stats import SurveyStats

        print(f"🔬 Computing Cronbach's Alpha for {args.alpha}...")
        survey_stats = SurveyStats(clean_df)
        result = survey_stats.cronbachs_alpha(args.alpha)
        _output(result, args.format, output_dir / "cronbachs_alpha")

    # ---- Group Comparison ----
    if args.compare:
        from survey_toolkit.stats import SurveyStats

        if not args.group:
            print("❌ --group is required with --compare")
            sys.exit(1)

        print(f"📈 Comparing {args.compare} by {args.group}...")
        survey_stats = SurveyStats(clean_df)
        result = survey_stats.compare_groups(args.compare, args.group)
        _output(result, args.format, output_dir / "group_comparison")

    # ---- Chi-Square ----
    if args.chi_square:
        from survey_toolkit.stats import SurveyStats

        col1, col2 = args.chi_square
        print(f"📊 Chi-square test: {col1} × {col2}...")
        survey_stats = SurveyStats(clean_df)
        result = survey_stats.chi_square_test(col1, col2)
        _output(result, args.format, output_dir / "chi_square")

    # ---- Correlations ----
    if args.correlations:
        from survey_toolkit.stats import SurveyStats

        print(f"🔗 Computing correlations for {args.correlations}...")
        survey_stats = SurveyStats(clean_df)
        result = survey_stats.correlation_matrix(args.correlations)
        _output(
            result["correlation_matrix"],
            args.format,
            output_dir / "correlations",
        )

    # ---- Factor Analysis ----
    if args.factor:
        from survey_toolkit.stats import SurveyStats

        print(f"🔬 Running factor analysis on {args.factor}...")
        survey_stats = SurveyStats(clean_df)
        result = survey_stats.factor_analysis(
            args.factor, n_factors=args.n_factors
        )
        _output(result["loadings"], args.format, output_dir / "factor_loadings")
        print(f"   KMO: {result['kmo']}")
        print(f"   Factors: {result['n_factors']}")

    # ---- Stats Summary ----
    if args.stats:
        print(f"📈 Statistical summary for {args.stats}...")
        summary = clean_df[args.stats].describe()
        _output(summary, args.format, output_dir / "stats_summary")

    # ---- Clustering ----
    if args.cluster:
        from survey_toolkit.ml_models import SurveySegmentation

        print(f"👥 Running segmentation on {args.cluster}...")
        seg = SurveySegmentation(clean_df)
        seg.prepare_data(args.cluster)

        if args.n_clusters is None:
            optimal = seg.find_optimal_k()
            n_clusters = optimal["optimal_k"]
            print(f"   Auto-detected optimal k = {n_clusters}")
        else:
            n_clusters = args.n_clusters

        profiles = seg.fit_clusters(n_clusters=n_clusters)
        _output(profiles, args.format, output_dir / "cluster_profiles")
        print(f"   Cluster sizes: {seg.results['cluster_sizes']}")

    # ---- Classification ----
    if args.classify:
        from survey_toolkit.ml_models import SurveyClassifier

        if not args.target:
            print("❌ --target is required with --classify")
            sys.exit(1)

        print(f"🤖 Running classification: {args.classify} → {args.target}...")
        classifier = SurveyClassifier(clean_df)
        classifier.prepare_data(
            feature_cols=args.classify, target_col=args.target
        )
        results = classifier.run_model_comparison()
        _output(results, args.format, output_dir / "model_comparison")

    print("✅ Done!")


def _output(data, fmt: str, filepath: Path):
    """Output results in the requested format."""
    if fmt == "print":
        if isinstance(data, dict):
            for k, v in data.items():
                if not isinstance(v, (pd.DataFrame, np.ndarray)):
                    print(f"  {k}: {v}")
        else:
            print(data)
    elif fmt == "json":
        outpath = filepath.with_suffix(".json")
        if hasattr(data, "to_json"):
            data.to_json(outpath, indent=2)
        else:
            with open(outpath, "w") as f:
                json.dump(data, f, indent=2, default=str)
        print(f"   💾 Saved to {outpath}")
    elif fmt == "csv":
        outpath = filepath.with_suffix(".csv")
        if hasattr(data, "to_csv"):
            data.to_csv(outpath)
        else:
            print(data)
        print(f"   💾 Saved to {outpath}")


# Need these imports for _output function
import pandas as pd
import numpy as np


def generate_report():
    """CLI entry point for report generation."""
    parser = argparse.ArgumentParser(
        prog="survey-report",
        description="Generate an HTML/PDF report from survey analysis.",
    )
    parser.add_argument(
        "input_file", type=str, help="Survey data file."
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="Columns to analyze.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/report.html",
        help="Output report path (default: outputs/report.html).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Survey Analysis Report",
        help="Report title.",
    )
    parser.add_argument(
        "--author",
        type=str,
        default="",
        help="Report author.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also generate a PDF version.",
    )
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="Run full analysis pipeline and include all results.",
    )

    args = parser.parse_args()

    from survey_toolkit.loader import SurveyLoader
    from survey_toolkit.cleaner import SurveyCleaner
    from survey_toolkit.reporting import ReportGenerator
    from survey_toolkit.stats import SurveyStats
    from survey_toolkit.eda import SurveyEDA

    # Load & Clean
    print(f"📥 Loading {args.input_file}...")
    df = SurveyLoader(args.input_file).load()
    clean_df = (
        SurveyCleaner(df)
        .handle_missing(strategy="median")
        .get_clean_data()
    )

    # Build report
    report = ReportGenerator(clean_df)
    report.set_metadata(
        title=args.title,
        author=args.author,
        description=f"Analysis of {len(clean_df)} survey responses.",
    )

    # Add summary stats
    report.add_summary_statistics(args.columns)

    # Full analysis
    if args.full_analysis:
        # EDA figures
        eda = SurveyEDA(clean_df, output_dir="outputs/figures")
        eda.plot_correlation_heatmap(args.columns, save=True)
        eda.plot_likert_distribution(args.columns, save=True)

        for fig_path in eda.figures:
            report.add_figure(
                f"Figure: {Path(fig_path).stem}",
                fig_path,
            )

        # Stats
        survey_stats = SurveyStats(clean_df)

        # Cronbach's alpha
        if len(args.columns) >= 2:
            alpha = survey_stats.cronbachs_alpha(args.columns)
            report.add_stats_result("Reliability Analysis", alpha)

        # Correlations
        corr_result = survey_stats.correlation_matrix(args.columns)
        report.add_dataframe(
            "Correlation Matrix",
            corr_result["correlation_matrix"],
            description=f"Method: {corr_result['method']}",
        )

    # Generate
    output_path = report.generate(
        columns=args.columns,
        title=args.title,
        output_path=args.output,
    )
    print(f"📄 Report saved to {output_path}")

    # PDF
    if args.pdf:
        pdf_path = str(Path(args.output).with_suffix(".pdf"))
        report.generate_pdf(output_path=pdf_path)
        print(f"📄 PDF saved to {pdf_path}")


if __name__ == "__main__":
    main()