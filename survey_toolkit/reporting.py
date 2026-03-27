"""
Report generation module for survey analysis results.
Generates HTML and PDF reports from analysis outputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from survey_toolkit.utils import logger, timer


class ReportGenerator:
    """
    Generate structured HTML/PDF reports from survey analysis results.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.sections = []
        self.title = "Survey Analysis Report"
        self.author = ""
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def set_metadata(
        self,
        title: str = "Survey Analysis Report",
        author: str = "",
        description: str = "",
    ) -> "ReportGenerator":
        """Set report metadata."""
        self.title = title
        self.author = author
        self.description = description
        return self

    def add_section(
        self,
        title: str,
        content: str,
        section_type: str = "text",
    ) -> "ReportGenerator":
        """
        Add a section to the report.

        Parameters
        ----------
        title : str
            Section heading.
        content : str
            HTML content or text.
        section_type : str
            'text', 'table', 'figure', 'stats'.
        """
        self.sections.append({
            "title": title,
            "content": content,
            "type": section_type,
        })
        return self

    def add_dataframe(
        self,
        title: str,
        df: pd.DataFrame,
        description: str = "",
        max_rows: int = 50,
    ) -> "ReportGenerator":
        """Add a DataFrame as a formatted table section."""
        table_html = df.head(max_rows).to_html(
            classes="table table-striped table-hover",
            border=0,
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        )
        content = ""
        if description:
            content += f"<p class='description'>{description}</p>"
        content += table_html
        if len(df) > max_rows:
            content += f"<p class='note'>Showing {max_rows} of {len(df)} rows.</p>"
        self.add_section(title, content, section_type="table")
        return self

    def add_stats_result(
        self,
        title: str,
        result: dict,
    ) -> "ReportGenerator":
        """Add a statistical test result as a formatted section."""
        content = "<div class='stats-result'>"
        content += "<table class='table table-bordered'>"
        for key, value in result.items():
            if isinstance(value, pd.DataFrame):
                content += f"<tr><td><strong>{key}</strong></td>"
                content += f"<td>{value.to_html(classes='table table-sm', border=0)}</td></tr>"
            elif isinstance(value, dict):
                formatted = "<ul>"
                for k, v in value.items():
                    formatted += f"<li><strong>{k}:</strong> {v}</li>"
                formatted += "</ul>"
                content += f"<tr><td><strong>{key}</strong></td><td>{formatted}</td></tr>"
            elif isinstance(value, (list, np.ndarray)):
                content += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
            else:
                # Highlight significance
                display_val = str(value)
                if key == "significant":
                    color = "green" if value else "red"
                    display_val = f"<span style='color:{color};font-weight:bold'>{value}</span>"
                elif key == "p_value" and isinstance(value, (int, float)):
                    color = "green" if value < 0.05 else "red"
                    display_val = f"<span style='color:{color}'>{value}</span>"
                content += f"<tr><td><strong>{key}</strong></td><td>{display_val}</td></tr>"
        content += "</table></div>"
        self.add_section(title, content, section_type="stats")
        return self

    def add_figure(
        self,
        title: str,
        figure_path: str,
        description: str = "",
    ) -> "ReportGenerator":
        """Add a figure/image to the report."""
        import base64

        path = Path(figure_path)
        if path.exists():
            # Embed image as base64
            with open(path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            content = f"<img src='data:image/png;base64,{img_data}' "
            content += "class='report-figure' style='max-width:100%;'>"
        else:
            content = f"<p class='error'>Figure not found: {figure_path}</p>"

        if description:
            content += f"<p class='figure-caption'>{description}</p>"

        self.add_section(title, content, section_type="figure")
        return self

    def add_summary_statistics(
        self,
        columns: Optional[list[str]] = None,
    ) -> "ReportGenerator":
        """Add auto-generated summary statistics section."""
        cols = columns or self.data.select_dtypes(include=[np.number]).columns.tolist()
        summary = self.data[cols].describe().round(4)
        self.add_dataframe(
            "Summary Statistics",
            summary,
            description=f"Descriptive statistics for {len(cols)} numeric variables.",
        )
        return self

    def _generate_css(self) -> str:
        """Generate report CSS styles."""
        return """
        <style>
            * { box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1100px;
                margin: 0 auto;
                padding: 40px 20px;
                color: #333;
                background: #fafafa;
                line-height: 1.6;
            }
            .report-header {
                background: linear-gradient(135deg, #1a365d, #2563eb);
                color: white;
                padding: 40px;
                border-radius: 10px;
                margin-bottom: 30px;
            }
            .report-header h1 {
                margin: 0 0 10px 0;
                font-size: 2em;
            }
            .report-header .meta {
                opacity: 0.85;
                font-size: 0.95em;
            }
            .section {
                background: white;
                padding: 30px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .section h2 {
                color: #1a365d;
                border-bottom: 2px solid #e2e8f0;
                padding-bottom: 10px;
                margin-top: 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 0.9em;
            }
            table th {
                background: #f1f5f9;
                padding: 10px 12px;
                text-align: left;
                font-weight: 600;
                border-bottom: 2px solid #cbd5e1;
            }
            table td {
                padding: 8px 12px;
                border-bottom: 1px solid #e2e8f0;
            }
            table tr:hover { background: #f8fafc; }
            .description {
                color: #64748b;
                font-style: italic;
                margin-bottom: 15px;
            }
            .note {
                color: #94a3b8;
                font-size: 0.85em;
                margin-top: 10px;
            }
            .error { color: #ef4444; }
            .figure-caption {
                color: #64748b;
                font-size: 0.9em;
                text-align: center;
                margin-top: 10px;
            }
            .report-figure {
                display: block;
                margin: 0 auto;
                border: 1px solid #e2e8f0;
                border-radius: 4px;
            }
            .stats-result table td { vertical-align: top; }
            .footer {
                text-align: center;
                color: #94a3b8;
                font-size: 0.8em;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e2e8f0;
            }
            @media print {
                body { background: white; }
                .section { box-shadow: none; break-inside: avoid; }
            }
        </style>
        """

    def _generate_html(self) -> str:
        """Generate the full HTML report."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    {self._generate_css()}
</head>
<body>
    <div class="report-header">
        <h1>📊 {self.title}</h1>
        <div class="meta">
"""
        if self.author:
            html += f"            <p>Author: {self.author}</p>\n"
        html += f"            <p>Generated: {self.created_at}</p>\n"
        html += f"            <p>Dataset: {len(self.data)} respondents, {len(self.data.columns)} variables</p>\n"
        if hasattr(self, "description") and self.description:
            html += f"            <p>{self.description}</p>\n"
        html += """        </div>
    </div>
"""

        # Table of contents
        if len(self.sections) > 3:
            html += "    <div class='section'>\n"
            html += "        <h2>📑 Table of Contents</h2>\n"
            html += "        <ol>\n"
            for i, section in enumerate(self.sections):
                html += f"            <li><a href='#section-{i}'>{section['title']}</a></li>\n"
            html += "        </ol>\n"
            html += "    </div>\n"

        # Sections
        for i, section in enumerate(self.sections):
            icon = {
                "text": "📝",
                "table": "📋",
                "figure": "📈",
                "stats": "🔬",
            }.get(section["type"], "📌")

            html += f"""
    <div class="section" id="section-{i}">
        <h2>{icon} {section['title']}</h2>
        {section['content']}
    </div>
"""

        html += f"""
    <div class="footer">
        <p>Generated by Survey ML Toolkit v0.1.0 | {self.created_at}</p>
    </div>
</body>
</html>"""
        return html

    @timer
    def generate(
        self,
        columns: Optional[list[str]] = None,
        title: Optional[str] = None,
        output_path: str = "outputs/report.html",
        auto_sections: bool = True,
    ) -> str:
        """
        Generate and save the HTML report.

        Parameters
        ----------
        columns : list[str], optional
            Columns to focus on in auto-generated sections.
        title : str, optional
            Override report title.
        output_path : str
            Path to save the HTML report.
        auto_sections : bool
            Automatically add summary statistics if no sections exist.

        Returns
        -------
        str
            Path to the generated report.
        """
        if title:
            self.title = title

        # Auto-add sections if empty
        if auto_sections and not self.sections:
            self.add_summary_statistics(columns)

            # Missing data summary
            missing = self.data.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                missing_df = pd.DataFrame({
                    "Column": missing.index,
                    "Missing Count": missing.values,
                    "Missing %": (missing.values / len(self.data) * 100).round(2),
                }).sort_values("Missing %", ascending=False)
                self.add_dataframe(
                    "Missing Data Summary",
                    missing_df,
                    description="Columns with missing values.",
                )

        # Generate HTML
        html = self._generate_html()

        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html, encoding="utf-8")
        logger.info(f"Report saved to {output_file}")

        return str(output_file)

    def generate_pdf(
        self,
        output_path: str = "outputs/report.pdf",
        **kwargs,
    ) -> str:
        """
        Generate a PDF report (requires weasyprint).

        Parameters
        ----------
        output_path : str
            Path to save the PDF.

        Returns
        -------
        str
            Path to the generated PDF.
        """
        try:
            from weasyprint import HTML
        except ImportError:
            raise ImportError(
                "weasyprint is required for PDF generation. "
                "Install with: pip install survey-ml-toolkit[reporting]"
            )

        # First generate HTML
        html_content = self._generate_html()

        # Convert to PDF
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html_content).write_pdf(str(output_file))
        logger.info(f"PDF report saved to {output_file}")

        return str(output_file)

    def __repr__(self) -> str:
        return (
            f"ReportGenerator(title='{self.title}', "
            f"sections={len(self.sections)})"
        )