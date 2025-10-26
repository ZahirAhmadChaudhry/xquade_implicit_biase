"""
Visualization suite for experimental results.
Generates heatmaps, bar charts, effect size plots, and language comparisons.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import math


class ResultVisualizer:
    """Creates publication-quality visualizations of experimental results."""
    
    def __init__(self, results_path: str, comparison_table_path: str = None):
        self.results_path = Path(results_path)
        self.output_dir = self.results_path / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load complete results
        all_results_file = self.results_path / "all_results.json"
        with open(all_results_file, 'r', encoding='utf-8') as f:
            self.all_results = json.load(f)
        
        # Load comparison table if available
        self.comparison_table = None
        if comparison_table_path:
            comp_file = self.results_path / comparison_table_path
            if comp_file.exists():
                with open(comp_file, 'r', encoding='utf-8') as f:
                    self.comparison_table = json.load(f)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        
        print(f"✅ Visualizer initialized. Output directory: {self.output_dir}")
    
    def plot_metric_heatmap(
        self,
        languages: List[str],
        metric: str = "avg_semantic_similarity",
        filename: str = None
    ):
        """
        Create heatmap showing metric values across configs and languages.
        """
        configs = list(self.all_results.keys())
        
        # Build matrix: rows=configs, cols=languages
        matrix = []
        for config in configs:
            row = []
            for lang in languages:
                # Average across seeds
                values = []
                if lang in self.all_results[config]:
                    for seed_key, result in self.all_results[config][lang].items():
                        val = result.get(metric)
                        if val is not None and not (isinstance(val, float) and math.isnan(val)):
                            values.append(val)
                
                row.append(np.mean(values) if values else np.nan)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, max(6, len(configs) * 0.4)))
        
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=[l.upper() for l in languages],
            yticklabels=configs,
            cbar_kws={'label': metric.replace('_', ' ').title()},
            ax=ax
        )
        
        ax.set_title(f'{metric.replace("_", " ").title()} Across Configurations and Languages', 
                     fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel('Language', fontsize=11, fontweight='bold')
        ax.set_ylabel('Configuration', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if filename is None:
            filename = f"heatmap_{metric}.png"
        plt.savefig(self.output_dir / filename, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved heatmap: {filename}")
    
    def plot_language_comparison(
        self,
        languages: List[str],
        metric: str = "avg_semantic_similarity",
        filename: str = None
    ):
        """
        Bar chart comparing configurations for each language.
        """
        configs = list(self.all_results.keys())
        
        x = np.arange(len(configs))
        width = 0.8 / len(languages)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, lang in enumerate(languages):
            means = []
            stds = []
            
            for config in configs:
                values = []
                if lang in self.all_results[config]:
                    for seed_key, result in self.all_results[config][lang].items():
                        val = result.get(metric)
                        if val is not None and not (isinstance(val, float) and math.isnan(val)):
                            values.append(val)
                
                means.append(np.mean(values) if values else 0)
                stds.append(np.std(values) if len(values) > 1 else 0)
            
            ax.bar(
                x + i * width,
                means,
                width,
                label=lang.upper(),
                yerr=stds,
                capsize=3,
                alpha=0.8
            )
        
        ax.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} by Configuration and Language',
                     fontsize=12, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(languages) - 1) / 2)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend(title='Language', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if filename is None:
            filename = f"language_comparison_{metric}.png"
        plt.savefig(self.output_dir / filename, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved language comparison: {filename}")
    
    def plot_effect_sizes(
        self,
        languages: List[str],
        metric: str = "avg_semantic_similarity",
        filename: str = None
    ):
        """
        Plot Cohen's d effect sizes for each config vs baseline.
        Requires comparison_table to be loaded.
        """
        if self.comparison_table is None:
            print("⚠️ No comparison table loaded. Skipping effect size plot.")
            return
        
        configs = list(self.comparison_table.keys())
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(configs) * 0.5)))
        
        y_positions = np.arange(len(configs))
        
        for i, lang in enumerate(languages):
            effect_sizes = []
            for config in configs:
                if lang in self.comparison_table[config]:
                    comparison = self.comparison_table[config][lang][metric]["comparison"]
                    d = comparison.get("cohens_d", 0)
                    if isinstance(d, float) and not math.isnan(d):
                        effect_sizes.append(d)
                    else:
                        effect_sizes.append(0)
                else:
                    effect_sizes.append(0)
            
            ax.barh(
                y_positions + i * 0.2,
                effect_sizes,
                0.18,
                label=lang.upper(),
                alpha=0.8
            )
        
        # Add reference lines for effect size interpretation
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
        ax.axvline(0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(0.8, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(-0.2, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        
        ax.set_yticks(y_positions + 0.2)
        ax.set_yticklabels(configs)
        ax.set_xlabel("Cohen's d (Effect Size)", fontsize=11, fontweight='bold')
        ax.set_title(f"Effect Sizes vs Baseline - {metric.replace('_', ' ').title()}",
                     fontsize=12, fontweight='bold', pad=20)
        ax.legend(title='Language', loc='best', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        
        # Add effect size interpretation text
        ax.text(0.2, -0.5, 'Small', ha='center', va='top', fontsize=8, alpha=0.6)
        ax.text(0.5, -0.5, 'Medium', ha='center', va='top', fontsize=8, alpha=0.6)
        ax.text(0.8, -0.5, 'Large', ha='center', va='top', fontsize=8, alpha=0.6)
        
        plt.tight_layout()
        
        if filename is None:
            filename = f"effect_sizes_{metric}.png"
        plt.savefig(self.output_dir / filename, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved effect size plot: {filename}")
    
    def plot_pvalue_heatmap(
        self,
        languages: List[str],
        metric: str = "avg_semantic_similarity",
        filename: str = None
    ):
        """
        Heatmap of p-values showing statistical significance.
        """
        if self.comparison_table is None:
            print("⚠️ No comparison table loaded. Skipping p-value heatmap.")
            return
        
        configs = list(self.comparison_table.keys())
        
        # Build matrix
        matrix = []
        for config in configs:
            row = []
            for lang in languages:
                if lang in self.comparison_table[config]:
                    comparison = self.comparison_table[config][lang][metric]["comparison"]
                    p_val = comparison.get("p_value")
                    if p_val is not None and not math.isnan(p_val):
                        # Use -log10(p) for better visualization
                        row.append(-np.log10(p_val) if p_val > 0 else 10)
                    else:
                        row.append(0)
                else:
                    row.append(0)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        fig, ax = plt.subplots(figsize=(8, max(6, len(configs) * 0.4)))
        
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            xticklabels=[l.upper() for l in languages],
            yticklabels=configs,
            cbar_kws={'label': '-log₁₀(p-value)'},
            ax=ax,
            vmin=0,
            vmax=5
        )
        
        ax.set_title(f'Statistical Significance - {metric.replace("_", " ").title()}',
                     fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel('Language', fontsize=11, fontweight='bold')
        ax.set_ylabel('Configuration', fontsize=11, fontweight='bold')
        
        # Add significance threshold lines
        # p=0.05 → -log10(0.05) ≈ 1.3
        # p=0.01 → -log10(0.01) = 2.0
        # p=0.001 → -log10(0.001) = 3.0
        
        plt.tight_layout()
        
        if filename is None:
            filename = f"pvalue_heatmap_{metric}.png"
        plt.savefig(self.output_dir / filename, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved p-value heatmap: {filename}")
    
    def plot_metric_distribution(
        self,
        language: str,
        config: str,
        metric: str = "avg_semantic_similarity",
        filename: str = None
    ):
        """
        Box plot showing distribution of metric across seeds.
        """
        values = []
        
        if config in self.all_results and language in self.all_results[config]:
            for seed_key, result in self.all_results[config][language].items():
                val = result.get(metric)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    values.append(val)
        
        if not values:
            print(f"⚠️ No data for {config}/{language}/{metric}")
            return
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.boxplot([values], labels=[config], showmeans=True)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution - {language.upper()} - {config}',
                     fontsize=11, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if filename is None:
            filename = f"distribution_{config}_{language}_{metric}.png"
        plt.savefig(self.output_dir / filename, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved distribution plot: {filename}")
    
    def generate_all_visualizations(self, languages: List[str]):
        """Generate complete visualization suite."""
        print("\n🎨 Generating all visualizations...")
        print("="*60)
        
        metrics = [
            "avg_semantic_similarity",
            "avg_bertscore_f1",
            "avg_token_f1",
            "avg_exact_match"
        ]
        
        # Heatmaps
        print("\n📊 Creating heatmaps...")
        for metric in metrics:
            self.plot_metric_heatmap(languages, metric)
        
        # Language comparisons
        print("\n📊 Creating language comparison charts...")
        for metric in metrics:
            self.plot_language_comparison(languages, metric)
        
        # Effect sizes (if comparison table available)
        if self.comparison_table:
            print("\n📊 Creating effect size plots...")
            for metric in metrics:
                self.plot_effect_sizes(languages, metric)
            
            print("\n📊 Creating p-value heatmaps...")
            for metric in metrics:
                self.plot_pvalue_heatmap(languages, metric)
        
        print(f"\n✅ All visualizations saved to {self.output_dir}")
        print("="*60)


def main():
    """Generate visualizations from results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualizations for experimental results")
    parser.add_argument("--results-dir", type=str, default="comprehensive_results", help="Results directory")
    parser.add_argument("--comparison-table", type=str, default="comparison_table.json", help="Comparison table file")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "th", "ar"], help="Languages to visualize")
    
    args = parser.parse_args()
    
    visualizer = ResultVisualizer(
        args.results_dir,
        comparison_table_path=args.comparison_table
    )
    
    visualizer.generate_all_visualizations(args.languages)
    
    print("\n✅ Visualization generation complete!")


if __name__ == "__main__":
    main()
