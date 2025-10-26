"""
Error analysis pipeline for identifying failure patterns and diagnostic insights.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import math


class ErrorAnalyzer:
    """Analyzes errors and failure patterns in experimental results."""
    
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.samples_dir = self.results_path / "samples"
        
        if not self.samples_dir.exists():
            print("⚠️ No sample-level results found. Run experiments with --save-samples.")
            self.has_samples = False
        else:
            self.has_samples = True
            print(f"✅ Found sample results in {self.samples_dir}")
    
    def load_samples(self, config: str, language: str, seed: int) -> List[Dict]:
        """Load sample-level results for a specific condition."""
        sample_file = self.samples_dir / config / language / f"seed_{seed}.json"
        
        if not sample_file.exists():
            return []
        
        with open(sample_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def identify_low_performers(
        self,
        config: str,
        language: str,
        seed: int,
        metric: str = "semantic_similarity_to_answer",
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Find samples with low performance.
        
        Returns:
            List of samples with metric below threshold
        """
        samples = self.load_samples(config, language, seed)
        
        low_performers = []
        for sample in samples:
            value = sample.get(metric)
            if value is not None and not math.isnan(value) and value < threshold:
                low_performers.append({
                    "sample_id": sample.get("sample_id"),
                    "metric_value": value,
                    "output": sample.get("output"),
                    "expected": sample.get("expected_answers"),
                    "prompt": sample.get("prompt")
                })
        
        return low_performers
    
    def categorize_errors(
        self,
        config: str,
        language: str,
        seed: int
    ) -> Dict[str, List[Dict]]:
        """
        Categorize errors into types:
        - Empty/null outputs
        - Very short outputs (< 3 tokens)
        - Very long outputs (> 50 tokens)
        - Low semantic similarity
        - Zero exact match
        """
        samples = self.load_samples(config, language, seed)
        
        categories = {
            "empty_output": [],
            "too_short": [],
            "too_long": [],
            "low_semantic": [],
            "zero_exact_match": [],
            "low_bertscore": []
        }
        
        for sample in samples:
            output = sample.get("output", "")
            sem_sim = sample.get("semantic_similarity_to_answer")
            exact_match = sample.get("exact_match")
            bertscore = sample.get("bertscore_f1")
            
            # Empty output
            if not output or output.strip() == "":
                categories["empty_output"].append(sample)
                continue
            
            # Token count
            tokens = output.split()
            if len(tokens) < 3:
                categories["too_short"].append(sample)
            elif len(tokens) > 50:
                categories["too_long"].append(sample)
            
            # Low semantic similarity
            if sem_sim is not None and not math.isnan(sem_sim) and sem_sim < 0.5:
                categories["low_semantic"].append(sample)
            
            # Zero exact match
            if exact_match is not None and not math.isnan(exact_match) and exact_match == 0:
                categories["zero_exact_match"].append(sample)
            
            # Low BERTScore
            if bertscore is not None and not math.isnan(bertscore) and bertscore < 0.5:
                categories["low_bertscore"].append(sample)
        
        return categories
    
    def generate_error_report(
        self,
        config: str,
        language: str,
        seed: int,
        output_file: str = None
    ) -> str:
        """Generate comprehensive error analysis report."""
        
        if not self.has_samples:
            return "No sample data available for error analysis."
        
        samples = self.load_samples(config, language, seed)
        
        if not samples:
            return f"No samples found for {config}/{language}/seed_{seed}"
        
        # Categorize errors
        categories = self.categorize_errors(config, language, seed)
        
        # Build report
        report = []
        report.append(f"{'='*80}")
        report.append(f"ERROR ANALYSIS REPORT")
        report.append(f"Configuration: {config}")
        report.append(f"Language: {language.upper()}")
        report.append(f"Seed: {seed}")
        report.append(f"Total Samples: {len(samples)}")
        report.append(f"{'='*80}\n")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)
        
        metrics = ["semantic_similarity_to_answer", "bertscore_f1", "token_f1", "exact_match"]
        for metric in metrics:
            values = []
            for sample in samples:
                val = sample.get(metric)
                if val is not None and not math.isnan(val):
                    values.append(val)
            
            if values:
                report.append(f"{metric:<35}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}, Min={np.min(values):.4f}, Max={np.max(values):.4f}")
        
        report.append("\n" + "ERROR CATEGORIES")
        report.append("-" * 80)
        
        for category, samples_in_cat in categories.items():
            count = len(samples_in_cat)
            percentage = (count / len(samples)) * 100 if samples else 0
            report.append(f"{category.replace('_', ' ').title():<30}: {count:>5} ({percentage:>5.1f}%)")
        
        # Detailed examples
        report.append("\n" + "FAILURE EXAMPLES")
        report.append("-" * 80)
        
        for category, samples_in_cat in categories.items():
            if samples_in_cat:
                report.append(f"\n{category.replace('_', ' ').title()}:")
                
                # Show up to 3 examples
                for i, sample in enumerate(samples_in_cat[:3], 1):
                    report.append(f"\n  Example {i}:")
                    report.append(f"    Sample ID: {sample.get('sample_id')}")
                    report.append(f"    Output: {sample.get('output', 'N/A')[:100]}...")
                    report.append(f"    Expected: {sample.get('expected_answers', ['N/A'])[0][:100]}...")
                    report.append(f"    Semantic Similarity: {sample.get('semantic_similarity_to_answer', 'N/A')}")
        
        report_text = "\n".join(report)
        
        # Save if requested
        if output_file:
            output_dir = self.results_path / "error_analysis"
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"📋 Saved error report: {output_path}")
        
        return report_text
    
    def compare_error_patterns(
        self,
        configs: List[str],
        language: str,
        seed: int
    ) -> Dict:
        """
        Compare error patterns across multiple configurations.
        
        Returns:
            Dictionary with error category counts for each config
        """
        comparison = {}
        
        for config in configs:
            categories = self.categorize_errors(config, language, seed)
            comparison[config] = {
                cat: len(samples) for cat, samples in categories.items()
            }
        
        return comparison
    
    def generate_all_reports(
        self,
        languages: List[str],
        seed: int = 42
    ):
        """Generate error reports for all configs and languages."""
        
        if not self.has_samples:
            print("⚠️ No sample data available.")
            return
        
        print("\n📋 Generating error analysis reports...")
        print("="*60)
        
        # Find all configs
        configs = [d.name for d in self.samples_dir.iterdir() if d.is_dir()]
        
        for config in configs:
            for lang in languages:
                report = self.generate_error_report(
                    config,
                    lang,
                    seed,
                    output_file=f"error_report_{config}_{lang}_seed{seed}.txt"
                )
                
                print(f"✅ Generated report for {config}/{lang}")
        
        # Generate comparison summary
        self._generate_comparison_summary(configs, languages, seed)
        
        print(f"\n✅ All error reports saved to {self.results_path / 'error_analysis'}")
        print("="*60)
    
    def _generate_comparison_summary(
        self,
        configs: List[str],
        languages: List[str],
        seed: int
    ):
        """Generate summary comparing error rates across configs."""
        
        output_dir = self.results_path / "error_analysis"
        output_dir.mkdir(exist_ok=True)
        
        summary = []
        summary.append("ERROR RATE COMPARISON ACROSS CONFIGURATIONS")
        summary.append("="*100)
        summary.append(f"Seed: {seed}\n")
        
        for lang in languages:
            summary.append(f"\nLanguage: {lang.upper()}")
            summary.append("-"*100)
            
            comparison = self.compare_error_patterns(configs, lang, seed)
            
            # Header
            header = f"{'Category':<25}"
            for config in configs:
                header += f"{config[:15]:<17}"
            summary.append(header)
            summary.append("-"*100)
            
            # Categories
            all_categories = set()
            for cat_dict in comparison.values():
                all_categories.update(cat_dict.keys())
            
            for category in sorted(all_categories):
                row = f"{category.replace('_', ' ').title():<25}"
                for config in configs:
                    count = comparison.get(config, {}).get(category, 0)
                    row += f"{count:<17}"
                summary.append(row)
        
        summary_text = "\n".join(summary)
        
        output_path = output_dir / f"error_comparison_seed{seed}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"📋 Saved error comparison: {output_path}")


def main():
    """Run error analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze errors in experimental results")
    parser.add_argument("--results-dir", type=str, default="comprehensive_results", help="Results directory")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "th", "ar"], help="Languages to analyze")
    parser.add_argument("--seed", type=int, default=42, help="Seed to analyze")
    
    args = parser.parse_args()
    
    analyzer = ErrorAnalyzer(args.results_dir)
    analyzer.generate_all_reports(args.languages, args.seed)
    
    print("\n✅ Error analysis complete!")


if __name__ == "__main__":
    main()
