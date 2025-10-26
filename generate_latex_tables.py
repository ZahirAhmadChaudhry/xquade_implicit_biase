"""
LaTeX table generator for publication-ready results.
Creates formatted tables with metrics, statistics, and significance markers.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import math


class LaTeXTableGenerator:
    """Generates LaTeX tables from experimental results."""
    
    def __init__(self, results_path: str, comparison_table_path: str = None):
        self.results_path = Path(results_path)
        self.output_dir = self.results_path / "latex_tables"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load complete results
        all_results_file = self.results_path / "all_results.json"
        with open(all_results_file, 'r', encoding='utf-8') as f:
            self.all_results = json.load(f)
        
        # Load comparison table
        self.comparison_table = None
        if comparison_table_path:
            comp_file = self.results_path / comparison_table_path
            if comp_file.exists():
                with open(comp_file, 'r', encoding='utf-8') as f:
                    self.comparison_table = json.load(f)
        
        print(f"✅ LaTeX generator initialized. Output directory: {self.output_dir}")
    
    def _format_value(self, value: float, precision: int = 3) -> str:
        """Format a numeric value for LaTeX."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "---"
        return f"{value:.{precision}f}"
    
    def _format_pvalue(self, p: float) -> str:
        """Format p-value with proper notation."""
        if p is None or math.isnan(p):
            return "---"
        if p < 0.001:
            return "< 0.001"
        return f"{p:.3f}"
    
    def _significance_marker(self, p: float) -> str:
        """Return LaTeX significance marker."""
        if p is None or math.isnan(p):
            return ""
        if p < 0.001:
            return "$^{***}$"
        if p < 0.01:
            return "$^{**}$"
        if p < 0.05:
            return "$^{*}$"
        return ""
    
    def generate_main_results_table(
        self,
        languages: List[str],
        metric: str = "avg_semantic_similarity",
        filename: str = None
    ) -> str:
        """
        Generate main results table showing all configs across languages.
        
        Format:
        Configuration | EN | ES | TH | AR
        """
        if self.comparison_table is None:
            print("⚠️ No comparison table loaded. Cannot generate results table.")
            return ""
        
        configs = list(self.comparison_table.keys())
        
        # Find baseline
        baseline_config = None
        for c in self.all_results.keys():
            if "no_tag" in c.lower() or "baseline" in c.lower():
                baseline_config = c
                break
        
        # Start LaTeX table
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{" + f"Results for {metric.replace('_', ' ').title()}" + "}")
        latex.append("\\label{tab:main_results}")
        
        # Column specification
        n_cols = len(languages) + 1
        latex.append(f"\\begin{{tabular}}{{l{'c' * len(languages)}}}")
        latex.append("\\toprule")
        
        # Header
        header = "\\textbf{Configuration}"
        for lang in languages:
            header += f" & \\textbf{{{lang.upper()}}}"
        header += " \\\\"
        latex.append(header)
        latex.append("\\midrule")
        
        # Baseline row
        if baseline_config:
            row = self._escape_latex(baseline_config)
            for lang in languages:
                stats = self._get_stats(baseline_config, lang, metric)
                row += f" & {self._format_value(stats['mean'])} $\\pm$ {self._format_value(stats['sem'], 4)}"
            row += " \\\\"
            latex.append(row)
            latex.append("\\midrule")
        
        # Other configurations
        for config in configs:
            row = self._escape_latex(config)
            
            for lang in languages:
                if lang in self.comparison_table[config]:
                    comparison = self.comparison_table[config][lang][metric]["comparison"]
                    stats = self.comparison_table[config][lang][metric]["config_stats"]
                    
                    mean = stats['mean']
                    sem = stats['sem']
                    p_val = comparison['p_value']
                    sig = self._significance_marker(p_val)
                    
                    row += f" & {self._format_value(mean)}{sig}"
                else:
                    row += " & ---"
            
            row += " \\\\"
            latex.append(row)
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\\\[0.5em]")
        latex.append("\\footnotesize")
        latex.append("Values shown as mean. Significance vs baseline: $^{*}$p < 0.05, $^{**}$p < 0.01, $^{***}$p < 0.001.")
        latex.append("\\end{table}")
        
        # Save to file
        if filename is None:
            filename = f"main_results_{metric}.tex"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex))
        
        print(f"📝 Saved LaTeX table: {filename}")
        
        return '\n'.join(latex)
    
    def generate_statistical_comparison_table(
        self,
        languages: List[str],
        metric: str = "avg_semantic_similarity",
        filename: str = None
    ) -> str:
        """
        Generate detailed statistical comparison table.
        
        Shows: Config | Mean ± SEM | Δ from baseline | p-value | Cohen's d
        """
        if self.comparison_table is None:
            print("⚠️ No comparison table loaded.")
            return ""
        
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{" + f"Statistical Comparison - {metric.replace('_', ' ').title()}" + "}")
        latex.append("\\label{tab:statistical_comparison}")
        latex.append("\\resizebox{\\textwidth}{!}{%")
        latex.append("\\begin{tabular}{llcccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Language} & \\textbf{Configuration} & \\textbf{Mean $\\pm$ SEM} & \\textbf{$\\Delta$ (\%)} & \\textbf{p-value} & \\textbf{Cohen's d} \\\\")
        latex.append("\\midrule")
        
        for lang in languages:
            configs = list(self.comparison_table.keys())
            
            for i, config in enumerate(configs):
                if lang not in self.comparison_table[config]:
                    continue
                
                comparison = self.comparison_table[config][lang][metric]["comparison"]
                stats = self.comparison_table[config][lang][metric]["config_stats"]
                
                # First column: language (merged cells)
                if i == 0:
                    lang_cell = f"\\multirow{{{len(configs)}}}{{*}}{{\\textbf{{{lang.upper()}}}}}"
                else:
                    lang_cell = ""
                
                mean = stats['mean']
                sem = stats['sem']
                delta = comparison['percent_change']
                p_val = comparison['p_value']
                cohens_d = comparison['cohens_d']
                sig = self._significance_marker(p_val)
                
                row = f"{lang_cell} & {self._escape_latex(config)}"
                row += f" & {self._format_value(mean)} $\\pm$ {self._format_value(sem, 4)}"
                row += f" & {self._format_value(delta, 1)}\\%{sig}"
                row += f" & {self._format_pvalue(p_val)}"
                row += f" & {self._format_value(cohens_d, 2)}"
                row += " \\\\"
                
                latex.append(row)
            
            if lang != languages[-1]:
                latex.append("\\midrule")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}%")
        latex.append("}")
        latex.append("\\\\[0.5em]")
        latex.append("\\footnotesize")
        latex.append("Significance: $^{*}$p < 0.05, $^{**}$p < 0.01, $^{***}$p < 0.001. ")
        latex.append("Effect sizes: small (0.2), medium (0.5), large (0.8).")
        latex.append("\\end{table}")
        
        if filename is None:
            filename = f"statistical_comparison_{metric}.tex"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex))
        
        print(f"📝 Saved LaTeX table: {filename}")
        
        return '\n'.join(latex)
    
    def generate_multi_metric_table(
        self,
        languages: List[str],
        metrics: List[str] = None,
        config: str = None,
        filename: str = None
    ) -> str:
        """
        Generate table showing multiple metrics for a specific configuration.
        """
        if metrics is None:
            metrics = [
                "avg_semantic_similarity",
                "avg_bertscore_f1",
                "avg_token_f1",
                "avg_exact_match"
            ]
        
        if config is None:
            # Choose best performing config
            config = list(self.comparison_table.keys())[0] if self.comparison_table else list(self.all_results.keys())[0]
        
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append(f"\\caption{{Multi-Metric Results - {self._escape_latex(config)}}}")
        latex.append("\\label{tab:multi_metric}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        
        # Header
        header = "\\textbf{Metric}"
        for lang in languages:
            header += f" & \\textbf{{{lang.upper()}}}"
        header += " \\\\"
        latex.append(header)
        latex.append("\\midrule")
        
        for metric in metrics:
            metric_name = metric.replace("avg_", "").replace("_", " ").title()
            row = metric_name
            
            for lang in languages:
                stats = self._get_stats(config, lang, metric)
                row += f" & {self._format_value(stats['mean'])}"
            
            row += " \\\\"
            latex.append(row)
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        if filename is None:
            filename = f"multi_metric_{config}.tex"
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex))
        
        print(f"📝 Saved LaTeX table: {filename}")
        
        return '\n'.join(latex)
    
    def _get_stats(self, config: str, language: str, metric: str) -> Dict:
        """Helper to extract statistics."""
        values = []
        
        if config in self.all_results and language in self.all_results[config]:
            for seed_key, result in self.all_results[config][language].items():
                val = result.get(metric)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    values.append(val)
        
        if not values:
            return {'mean': float('nan'), 'sem': float('nan')}
        
        return {
            'mean': float(np.mean(values)),
            'sem': float(np.std(values, ddof=1) / np.sqrt(len(values)))
        }
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = {
            '_': '\\_',
            '%': '\\%',
            '$': '\\$',
            '&': '\\&',
            '#': '\\#',
            '{': '\\{',
            '}': '\\}',
        }
        
        for char, escaped in replacements.items():
            text = text.replace(char, escaped)
        
        return text
    
    def generate_all_tables(self, languages: List[str]):
        """Generate complete set of LaTeX tables."""
        print("\n📝 Generating LaTeX tables...")
        print("="*60)
        
        metrics = [
            "avg_semantic_similarity",
            "avg_bertscore_f1",
            "avg_token_f1",
            "avg_exact_match"
        ]
        
        # Main results tables
        print("\n📊 Creating main results tables...")
        for metric in metrics:
            self.generate_main_results_table(languages, metric)
        
        # Statistical comparison tables
        if self.comparison_table:
            print("\n📊 Creating statistical comparison tables...")
            for metric in metrics:
                self.generate_statistical_comparison_table(languages, metric)
        
        # Multi-metric tables for top configs
        if self.comparison_table:
            print("\n📊 Creating multi-metric tables...")
            top_configs = list(self.comparison_table.keys())[:3]
            for config in top_configs:
                self.generate_multi_metric_table(languages, metrics, config)
        
        print(f"\n✅ All LaTeX tables saved to {self.output_dir}")
        print("="*60)


def main():
    """Generate LaTeX tables from results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for experimental results")
    parser.add_argument("--results-dir", type=str, default="comprehensive_results", help="Results directory")
    parser.add_argument("--comparison-table", type=str, default="comparison_table.json", help="Comparison table file")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "th", "ar"], help="Languages to include")
    
    args = parser.parse_args()
    
    generator = LaTeXTableGenerator(
        args.results_dir,
        comparison_table_path=args.comparison_table
    )
    
    generator.generate_all_tables(args.languages)
    
    print("\n✅ LaTeX table generation complete!")


if __name__ == "__main__":
    main()
