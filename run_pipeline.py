"""
Master script to run complete experimental pipeline.
Coordinates execution, aggregation, visualization, and reporting.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list, description: str):
    """Execute a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        print(f"   Return code: {result.returncode}")
        return False
    else:
        print(f"\n✅ Completed: {description}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run complete experimental pipeline")
    parser.add_argument("--skip-experiments", action="store_true", help="Skip running experiments (use existing results)")
    parser.add_argument("--full-matrix", action="store_true", help="Run full experiment matrix")
    parser.add_argument("--sample-size", type=int, default=240, help="Samples per language per seed")
    parser.add_argument("--num-seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "th", "ar"], help="Languages to evaluate")
    parser.add_argument("--results-dir", type=str, default="comprehensive_results", help="Results directory")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔬 COMPREHENSIVE EXPERIMENT PIPELINE")
    print("="*80)
    print(f"Sample size: {args.sample_size}")
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Languages: {', '.join(args.languages)}")
    print(f"Results directory: {args.results_dir}")
    print(f"Full matrix: {args.full_matrix}")
    print("="*80)
    
    # Step 1: Run experiments
    if not args.skip_experiments:
        cmd = [
            sys.executable,
            "run_comprehensive_experiments.py",
            "--sample-size", str(args.sample_size),
            "--num-seeds", str(args.num_seeds),
            "--languages", *args.languages,
            "--results-dir", args.results_dir
        ]
        
        if args.full_matrix:
            cmd.append("--full-matrix")
        
        if not run_command(cmd, "Running experiments"):
            print("\n❌ Pipeline failed at experiment stage")
            return 1
    else:
        print("\n⏭️  Skipping experiments (using existing results)")
    
    # Step 2: Aggregate results
    cmd = [
        sys.executable,
        "aggregate_results.py",
        "--results-dir", args.results_dir,
        "--languages", *args.languages,
        "--output", "comparison_table.json"
    ]
    
    if not run_command(cmd, "Aggregating results"):
        print("\n⚠️  Aggregation failed, but continuing...")
    
    # Step 3: Generate visualizations
    cmd = [
        sys.executable,
        "visualize_results.py",
        "--results-dir", args.results_dir,
        "--comparison-table", "comparison_table.json",
        "--languages", *args.languages
    ]
    
    if not run_command(cmd, "Generating visualizations"):
        print("\n⚠️  Visualization failed, but continuing...")
    
    # Step 4: Generate LaTeX tables
    cmd = [
        sys.executable,
        "generate_latex_tables.py",
        "--results-dir", args.results_dir,
        "--comparison-table", "comparison_table.json",
        "--languages", *args.languages
    ]
    
    if not run_command(cmd, "Generating LaTeX tables"):
        print("\n⚠️  LaTeX generation failed, but continuing...")
    
    # Step 5: Error analysis
    cmd = [
        sys.executable,
        "error_analysis.py",
        "--results-dir", args.results_dir,
        "--languages", *args.languages,
        "--seed", "42"
    ]
    
    if not run_command(cmd, "Running error analysis"):
        print("\n⚠️  Error analysis failed, but continuing...")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print(f"\n📁 All results saved to: {args.results_dir}/")
    print(f"   - Raw results: all_results.json")
    print(f"   - Aggregated: comparison_table.json")
    print(f"   - Visualizations: visualizations/")
    print(f"   - LaTeX tables: latex_tables/")
    print(f"   - Error analysis: error_analysis/")
    print(f"   - Sample results: samples/")
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
