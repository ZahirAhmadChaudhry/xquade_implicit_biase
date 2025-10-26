# Quick Start Guide - Comprehensive Experiments

## ⚡ Fastest Path to Results

### 1. Install Dependencies
```cmd
pip install -r requirements.txt
```

### 2. Set API Key
Create `.env` file:
```
GEMINI_API_KEY=your_key_here
```

### 3. Run Everything
```cmd
python run_pipeline.py
```

**Expected duration**: 6-8 hours for core configs (240 samples × 3 seeds × 4 languages × ~15 configs)

---

## 📊 What You'll Get

After completion, check `comprehensive_results/`:

### Key Files
- **all_results.json**: Complete raw results
- **comparison_table.json**: Statistical analysis with p-values, effect sizes
- **visualizations/**: All plots (heatmaps, bar charts, effect sizes)
- **latex_tables/**: Publication-ready tables
- **error_analysis/**: Detailed error reports

### Quick Preview Commands

**View statistical summary**:
```cmd
python aggregate_results.py --results-dir comprehensive_results
```

**Generate visualizations**:
```cmd
python visualize_results.py --results-dir comprehensive_results
```

---

## 🎯 Common Use Cases

### Small Test Run (Quick Validation)
```cmd
python run_pipeline.py --sample-size 50 --num-seeds 1 --languages en es
```
**Duration**: ~30 minutes

### Full Publication Run
```cmd
python run_pipeline.py --sample-size 240 --num-seeds 3 --languages en es th ar
```
**Duration**: ~6-8 hours

### Full Matrix (All Variations)
```cmd
python run_pipeline.py --full-matrix --sample-size 240 --num-seeds 3
```
**Duration**: ~12-16 hours

---

## 🔄 Resume Interrupted Run

If experiments are interrupted, just re-run:
```cmd
python run_comprehensive_experiments.py
```

It automatically resumes from `checkpoint.json`.

---

## 📈 Understanding Results

### Key Metrics (in order of importance)
1. **Semantic Similarity**: Main metric (sentence-transformers embeddings)
2. **BERTScore F1**: Contextual quality (xlm-roberta-large)
3. **Token F1**: Lexical overlap
4. **Exact Match**: Binary correctness

### Interpreting Statistical Tables

**Significance markers**:
- `***` = p < 0.001 (highly significant)
- `**` = p < 0.01 (very significant)
- `*` = p < 0.05 (significant)
- No marker = not significant

**Effect sizes (Cohen's d)**:
- 0.2 = small effect
- 0.5 = medium effect
- 0.8 = large effect

---

## 🐛 Troubleshooting

### "API rate limit exceeded"
Add this to `run_comprehensive_experiments.py` after line 141:
```python
time.sleep(1)  # Add 1 second delay between requests
```

### "Out of memory"
Reduce sample size:
```cmd
python run_pipeline.py --sample-size 100
```

### "Module not found"
```cmd
pip install -r requirements.txt
```

---

## 📂 File Structure Reference

```
exploring_idea/
├── run_pipeline.py                      # 🚀 RUN THIS (master script)
├── run_comprehensive_experiments.py     # Experiment executor
├── aggregate_results.py                 # Statistical analysis
├── visualize_results.py                 # Visualization generator
├── generate_latex_tables.py             # LaTeX table creator
├── error_analysis.py                    # Error pattern analyzer
├── experiment_config.py                 # Configuration definitions
├── prompt_builder.py                    # Prompt generator
├── experiment.py                        # Original experiment code
├── stats_analysis.py                    # Statistical utilities
├── requirements.txt                     # Dependencies
├── .env                                 # API key (create this)
├── COMPREHENSIVE_README.md              # Full documentation
└── QUICK_START.md                       # This file
```

---

## ✅ Next Steps After Running

1. **Check statistical significance**:
   - Look at `comparison_table.json`
   - Review console output from `aggregate_results.py`

2. **Review visualizations**:
   - Heatmaps show overall patterns
   - Effect size plots show practical significance
   - Bar charts show language-specific differences

3. **Examine failures**:
   - Check `error_analysis/` for failure patterns
   - Low-performing samples in `samples/` directories

4. **Prepare publication**:
   - Use LaTeX tables from `latex_tables/`
   - Include key visualizations
   - Summarize statistical findings from `comparison_table.json`

---

## 🎓 Publication Checklist

- [ ] Run with ≥240 samples per language
- [ ] Use ≥3 random seeds
- [ ] Include 4+ languages (EN, ES, TH, AR recommended)
- [ ] Report Bonferroni-corrected p-values
- [ ] Include effect sizes (Cohen's d)
- [ ] Show error analysis
- [ ] Present visualizations (heatmap + effect sizes minimum)
- [ ] Include LaTeX tables in manuscript

---

## 💡 Tips

- **Start small**: Test with `--sample-size 50 --num-seeds 1` first
- **Use checkpoints**: Don't worry about interruptions
- **Monitor progress**: Watch console output for per-run updates
- **Skip re-runs**: Use `--skip-experiments` to regenerate visualizations without re-running

---

## 📧 Need Help?

Check:
1. `COMPREHENSIVE_README.md` for detailed documentation
2. Console error messages for specific issues
3. `checkpoint.json` to see progress

Happy experimenting! 🚀
