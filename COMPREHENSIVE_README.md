# Comprehensive Language Tag Experiment Pipeline

## Overview

This pipeline provides a complete framework for evaluating the impact of language tags on multilingual question-answering performance using the XQuAD dataset. It implements systematic experiments across multiple tag formats, placements, few-shot configurations, and languages.

## Features

### Experimental Configurations
- **Tag Formats**: 5 types (bracket, natural, XML, instruction, ISO)
- **Tag Placements**: 4 positions (prefix, suffix, both, none/baseline)
- **Few-Shot Learning**: 0-3 example configurations
- **Ablation Studies**: Wrong-tag experiments, multilingual contexts
- **Statistical Rigor**: Bonferroni-corrected significance testing, effect sizes, confidence intervals

### Metrics
- **Semantic Similarity**: sentence-transformers (paraphrase-multilingual-mpnet-base-v2)
- **BERTScore**: xlm-roberta-large
- **Token F1**: Token overlap
- **ROUGE-L**: Lexical similarity
- **Exact Match**: Binary correctness

### Output
- Comprehensive JSON results with per-sample detail
- Publication-ready LaTeX tables
- High-quality visualizations (heatmaps, bar charts, effect size plots)
- Error analysis reports with failure categorization
- Statistical comparison tables with Bonferroni correction

## Installation

```cmd
pip install -r requirements.txt
```

### API Key Setup
Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Quick Start

### Run Complete Pipeline

```cmd
python run_pipeline.py
```

This will:
1. Execute all experimental conditions (240 samples × 3 seeds × ~15 configs × 4 languages)
2. Aggregate results with statistical testing
3. Generate all visualizations
4. Create LaTeX tables
5. Perform error analysis

### Options

```cmd
python run_pipeline.py --sample-size 240 --num-seeds 3 --languages en es th ar
```

- `--skip-experiments`: Use existing results (skip re-running experiments)
- `--full-matrix`: Run full experiment matrix (~30 configs instead of core ~15)
- `--sample-size N`: Samples per language per seed (default: 240)
- `--num-seeds N`: Number of random seeds (default: 3)
- `--languages L1 L2...`: Languages to evaluate (default: en es th ar)
- `--results-dir DIR`: Output directory (default: comprehensive_results)

## Individual Components

### 1. Run Experiments

```cmd
python run_comprehensive_experiments.py --sample-size 240 --num-seeds 3
```

Creates:
- `comprehensive_results/all_results.json`: Complete aggregated results
- `comprehensive_results/checkpoint.json`: Incremental checkpoint (resume support)
- `comprehensive_results/samples/`: Per-sample detailed results

### 2. Aggregate Results

```cmd
python aggregate_results.py --results-dir comprehensive_results
```

Creates:
- `comparison_table.json`: Statistical comparisons with Bonferroni correction
- Console output with significance tables

### 3. Generate Visualizations

```cmd
python visualize_results.py --results-dir comprehensive_results
```

Creates in `visualizations/`:
- Metric heatmaps (config × language)
- Language comparison bar charts
- Effect size plots
- P-value heatmaps

### 4. Generate LaTeX Tables

```cmd
python generate_latex_tables.py --results-dir comprehensive_results
```

Creates in `latex_tables/`:
- Main results tables
- Statistical comparison tables
- Multi-metric summaries

### 5. Error Analysis

```cmd
python error_analysis.py --results-dir comprehensive_results --seed 42
```

Creates in `error_analysis/`:
- Per-configuration error reports
- Error category comparisons
- Failure pattern analysis

## Experimental Design

### Core Configurations (~15)

1. **Baseline**: No language tags
2. **Tag Format Variations**:
   - Bracket: `[EN]`, `[ES]`, `[TH]`, `[AR]`
   - Natural: `This is English text...`
   - XML: `<lang>en</lang>`
   - Instruction: `Language: English`
   - ISO: `ISO-639: en`
3. **Tag Placement**:
   - Prefix only
   - Suffix only
   - Both prefix and suffix
4. **Few-Shot**:
   - 1-shot examples
   - 3-shot examples
5. **Ablations**:
   - Wrong language tags (e.g., `[ES]` for English text)

### Full Matrix (~30+)

Includes all combinations of:
- Tag formats × placements
- Few-shot variations (0, 1, 2, 3 examples)
- Multilingual contexts (parallel, interleaved)

## Output Structure

```
comprehensive_results/
├── all_results.json              # Complete results
├── checkpoint.json               # Resume checkpoint
├── comparison_table.json         # Statistical comparisons
├── samples/                      # Per-sample results
│   ├── config_1/
│   │   ├── en/
│   │   │   ├── seed_42.json
│   │   │   ├── seed_123.json
│   │   │   └── seed_456.json
│   │   ├── es/
│   │   ├── th/
│   │   └── ar/
├── visualizations/               # All plots
│   ├── heatmap_avg_semantic_similarity.png
│   ├── language_comparison_avg_bertscore_f1.png
│   ├── effect_sizes_avg_token_f1.png
│   └── ...
├── latex_tables/                 # Publication tables
│   ├── main_results_avg_semantic_similarity.tex
│   ├── statistical_comparison_avg_bertscore_f1.tex
│   └── ...
└── error_analysis/               # Error reports
    ├── error_report_config_1_en_seed42.txt
    ├── error_comparison_seed42.txt
    └── ...
```

## Statistical Methods

### Multiple Comparison Correction
- **Bonferroni Correction**: α_adjusted = 0.05 / n_comparisons
- Applied to all pairwise comparisons vs baseline

### Effect Sizes
- **Cohen's d**: Standardized mean difference for paired samples
- Interpretation: 0.2 (small), 0.5 (medium), 0.8 (large)

### Significance Testing
- **Paired t-test**: Primary test for matched samples across seeds
- **Wilcoxon signed-rank**: Non-parametric alternative
- **95% Confidence Intervals**: Bootstrap or t-distribution based

## Configuration Files

### experiment_config.py
Defines all experimental conditions:
- `TagFormat`: Enum for tag types
- `TagPlacement`: Enum for tag positions
- `PromptConfig`: Dataclass for experiment configuration
- `get_core_experiment_configs()`: Returns core experimental matrix
- `get_all_experiment_configs()`: Returns full experimental matrix

### prompt_builder.py
Handles prompt generation:
- `PromptBuilder`: Builds prompts with tags, few-shot examples, multilingual contexts
- `FewShotExampleSelector`: Selects random examples with exclusion logic

## Performance Expectations

### Timing (approximate)
- **Core configs** (~15): ~6-8 hours (240 samples × 3 seeds × 4 languages)
- **Full matrix** (~30): ~12-16 hours
- **API rate limits**: Gemini 2.0 Flash typically allows 60 requests/minute

### Resource Requirements
- **RAM**: 4-8 GB (for embeddings and BERTScore)
- **Storage**: ~500 MB - 2 GB (depending on sample saving)
- **GPU**: Optional (speeds up BERTScore but not required)

## Troubleshooting

### API Rate Limits
The runner includes automatic rate limiting. If you hit limits:
- Reduce `--sample-size`
- Use `--skip-experiments` to work with existing results
- Add delays in `run_comprehensive_experiments.py` (search for `time.sleep`)

### Memory Issues
- Disable sample saving: modify `ExperimentSettings` to set `save_samples=False`
- Process languages sequentially instead of loading all upfront

### Resume from Checkpoint
If experiments are interrupted, results are checkpointed after each run. Simply re-run:
```cmd
python run_comprehensive_experiments.py
```
It will skip completed runs (check `checkpoint.json`).

## Citation

If you use this pipeline, please cite:

```bibtex
@misc{language_tag_experiments_2024,
  title={Comprehensive Evaluation of Language Tags in Multilingual Question Answering},
  author={Your Name},
  year={2024},
  howpublished={\\url{https://github.com/yourusername/repo}}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
