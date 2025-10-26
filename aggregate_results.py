"""
Result aggregation and statistical analysis across experimental conditions.
Combines results from multiple seeds and computes significance tests.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from stats_analysis import paired_summary
import math


class ResultAggregator:
    """Aggregates and analyzes results across experimental conditions."""
    
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        
        # Load complete results
        all_results_file = self.results_path / "all_results.json"
        with open(all_results_file, 'r', encoding='utf-8') as f:
            self.all_results = json.load(f)
        
        # Extract structure
        self.configs = list(self.all_results.keys())
        
        # Find baseline (no-tag condition)
        self.baseline_config = None
        for config in self.configs:
            if "no_tag" in config.lower() or "baseline" in config.lower():
                self.baseline_config = config
                break
        
        if not self.baseline_config and self.configs:
            self.baseline_config = self.configs[0]
            print(f"⚠️ No explicit baseline found, using {self.baseline_config}")
        
        print(f"✅ Loaded results for {len(self.configs)} configurations")
        print(f"📊 Baseline: {self.baseline_config}")
    
    def get_metric_values(
        self,
        config: str,
        language: str,
        metric: str = "avg_semantic_similarity"
    ) -> List[float]:
        """Extract metric values across all seeds for a given config and language."""
        values = []
        
        if config not in self.all_results:
            return values
        
        if language not in self.all_results[config]:
            return values
        
        for seed_key, result in self.all_results[config][language].items():
            value = result.get(metric)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                values.append(value)
        
        return values
    
    def compute_cross_seed_statistics(
        self,
        config: str,
        language: str,
        metric: str = "avg_semantic_similarity"
    ) -> Dict:
        """Compute mean and std across seeds for a config/language/metric."""
        values = self.get_metric_values(config, language, metric)
        
        if not values:
            return {
                "mean": float('nan'),
                "std": float('nan'),
                "sem": float('nan'),
                "n_seeds": 0,
                "values": []
            }
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))),
            "n_seeds": len(values),
            "values": values
        }
    
    def compare_to_baseline(
        self,
        config: str,
        language: str,
        metric: str = "avg_semantic_similarity",
        alpha: float = 0.05
    ) -> Dict:
        """
        Compare a configuration to baseline with paired t-test.
        
        Args:
            config: Configuration name
            language: Language code
            metric: Metric to compare
            alpha: Significance level (will apply Bonferroni correction)
        
        Returns:
            Dictionary with statistical comparison
        """
        baseline_values = self.get_metric_values(self.baseline_config, language, metric)
        config_values = self.get_metric_values(config, language, metric)
        
        if not baseline_values or not config_values:
            return {
                "comparison": f"{config} vs {self.baseline_config}",
                "language": language,
                "metric": metric,
                "status": "insufficient_data",
                "baseline_mean": float('nan'),
                "config_mean": float('nan'),
                "mean_difference": float('nan'),
                "p_value": float('nan'),
                "significant": False,
            }
        
        # Ensure equal length (take minimum)
        min_len = min(len(baseline_values), len(config_values))
        baseline_values = baseline_values[:min_len]
        config_values = config_values[:min_len]
        
        # Compute statistics
        baseline_mean = np.mean(baseline_values)
        config_mean = np.mean(config_values)
        mean_diff = config_mean - baseline_mean
        
        # Paired t-test
        if len(baseline_values) > 1:
            t_stat, p_value = stats.ttest_rel(config_values, baseline_values)
            
            # Cohen's d for paired samples
            diff = np.array(config_values) - np.array(baseline_values)
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            
        else:
            p_value = float('nan')
            cohens_d = float('nan')
        
        # Apply Bonferroni correction (will be set by caller)
        # For now, just report raw p-value
        
        return {
            "comparison": f"{config} vs {self.baseline_config}",
            "language": language,
            "metric": metric,
            "baseline_mean": float(baseline_mean),
            "baseline_std": float(np.std(baseline_values, ddof=1)),
            "config_mean": float(config_mean),
            "config_std": float(np.std(config_values, ddof=1)),
            "mean_difference": float(mean_diff),
            "percent_change": float((mean_diff / baseline_mean * 100) if baseline_mean != 0 else 0),
            "p_value": float(p_value) if not math.isnan(p_value) else float('nan'),
            "cohens_d": float(cohens_d) if not math.isnan(cohens_d) else float('nan'),
            "n_seeds": min_len,
            "significant": p_value < alpha if not math.isnan(p_value) else False,
        }
    
    def generate_comparison_table(
        self,
        languages: List[str],
        metrics: List[str] = None,
        output_file: str = None
    ) -> Dict:
        """
        Generate comprehensive comparison table across all configs and languages.
        
        Args:
            languages: List of language codes
            metrics: List of metrics to compare (defaults to key metrics)
            output_file: Optional file path to save JSON results
        
        Returns:
            Nested dictionary: config → language → metric → statistics
        """
        if metrics is None:
            metrics = [
                "avg_semantic_similarity",
                "avg_bertscore_f1",
                "avg_token_f1",
                "avg_exact_match"
            ]
        
        # Calculate Bonferroni correction
        num_comparisons = (len(self.configs) - 1) * len(languages) * len(metrics)
        bonferroni_alpha = 0.05 / num_comparisons if num_comparisons > 0 else 0.05
        
        print(f"🔬 Bonferroni correction: α = 0.05 / {num_comparisons} = {bonferroni_alpha:.6f}")
        
        comparison_table = {}
        
        for config in self.configs:
            if config == self.baseline_config:
                continue  # Skip baseline comparing to itself
            
            config_results = {}
            
            for lang in languages:
                lang_results = {}
                
                for metric in metrics:
                    # Get cross-seed statistics
                    config_stats = self.compute_cross_seed_statistics(config, lang, metric)
                    baseline_stats = self.compute_cross_seed_statistics(self.baseline_config, lang, metric)
                    
                    # Compare to baseline
                    comparison = self.compare_to_baseline(config, lang, metric, bonferroni_alpha)
                    
                    lang_results[metric] = {
                        "config_stats": config_stats,
                        "baseline_stats": baseline_stats,
                        "comparison": comparison,
                        "bonferroni_significant": bool(comparison["p_value"] < bonferroni_alpha if not math.isnan(comparison["p_value"]) else False),
                    }
                
                config_results[lang] = lang_results
            
            comparison_table[config] = config_results
        
        # Save if requested
        if output_file:
            output_path = self.results_path / output_file
            # Convert to JSON-serializable format
            serializable = json.loads(json.dumps(comparison_table, default=str))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved comparison table to {output_path}")
        
        return comparison_table
    
    def print_summary_table(self, languages: List[str], metric: str = "avg_semantic_similarity"):
        """Print a human-readable summary table."""
        print(f"\n{'='*100}")
        print(f"📊 SUMMARY: {metric}")
        print(f"{'='*100}")
        
        # Header
        header = f"{'Configuration':<35}"
        for lang in languages:
            header += f"{lang.upper():^20}"
        print(header)
        print("-" * 100)
        
        # Baseline row
        baseline_row = f"{self.baseline_config:<35}"
        for lang in languages:
            stats = self.compute_cross_seed_statistics(self.baseline_config, lang, metric)
            baseline_row += f"{stats['mean']:>8.4f} ± {stats['sem']:<8.4f}"
        print(baseline_row)
        print("-" * 100)
        
        # Config rows
        for config in self.configs:
            if config == self.baseline_config:
                continue
            
            row = f"{config:<35}"
            for lang in languages:
                stats = self.compute_cross_seed_statistics(config, lang, metric)
                comparison = self.compare_to_baseline(config, lang, metric)
                
                # Format with significance marker
                sig_marker = "***" if comparison["p_value"] < 0.001 else \
                            "**" if comparison["p_value"] < 0.01 else \
                            "*" if comparison["p_value"] < 0.05 else ""
                
                row += f"{stats['mean']:>8.4f}{sig_marker:<3} "
                row += f"({comparison['percent_change']:>+6.1f}%)"
            
            print(row)
        
        print(f"{'='*100}")
        print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
        print(f"{'='*100}\n")
    
    def get_best_configs_per_language(
        self,
        languages: List[str],
        metric: str = "avg_semantic_similarity",
        top_k: int = 3
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        """
        Find top-k best performing configs for each language.
        
        Returns:
            Dictionary: language → [(config, mean_value, p_value), ...]
        """
        best_configs = {}
        
        for lang in languages:
            # Get all configs with their scores
            config_scores = []
            for config in self.configs:
                stats = self.compute_cross_seed_statistics(config, lang, metric)
                comparison = self.compare_to_baseline(config, lang, metric)
                
                config_scores.append((
                    config,
                    stats['mean'],
                    comparison['p_value']
                ))
            
            # Sort by mean value (descending)
            config_scores.sort(key=lambda x: x[1] if not math.isnan(x[1]) else -float('inf'), reverse=True)
            
            best_configs[lang] = config_scores[:top_k]
        
        return best_configs


def main():
    """Run aggregation and analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate and analyze experimental results")
    parser.add_argument("--results-dir", type=str, default="comprehensive_results", help="Results directory")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "th", "ar"], help="Languages to analyze")
    parser.add_argument("--output", type=str, default="comparison_table.json", help="Output file name")
    
    args = parser.parse_args()
    
    # Initialize aggregator
    aggregator = ResultAggregator(args.results_dir)
    
    # Generate comparison table
    print("\n🔬 Generating comprehensive comparison table...")
    comparison_table = aggregator.generate_comparison_table(
        args.languages,
        output_file=args.output
    )
    
    # Print summary tables
    metrics = [
        "avg_semantic_similarity",
        "avg_bertscore_f1",
        "avg_token_f1",
        "avg_exact_match"
    ]
    
    for metric in metrics:
        aggregator.print_summary_table(args.languages, metric)
    
    # Find best configs
    print("\n🏆 Best performing configurations per language:")
    print("="*80)
    for metric in ["avg_semantic_similarity", "avg_bertscore_f1"]:
        print(f"\n📊 Metric: {metric}")
        print("-"*80)
        best = aggregator.get_best_configs_per_language(args.languages, metric, top_k=3)
        
        for lang, configs in best.items():
            print(f"\n  {lang.upper()}:")
            for i, (config, mean_val, p_val) in enumerate(configs, 1):
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"    {i}. {config:<30} | Mean: {mean_val:.4f} | p: {p_val:.4f} {sig}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
