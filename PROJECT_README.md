# Multilingual LLM Bias Evaluation: XQuAD Benchmarking

## 🎯 Project Overview

This project investigates **implicit translation bias** in Large Language Models (LLMs) and evaluates whether explicit language tags can improve semantic consistency and translation accuracy in multilingual question-answering tasks.

### Core Hypothesis
**Explicit language tags (e.g., `[lang=es]`) will improve semantic consistency and reduce translation drift when LLMs process multilingual content, leading to more accurate and consistent outputs across different languages.**

---

## 🔬 Research Methodology

### Problem Statement
Large Language Models often exhibit implicit biases when processing multilingual content, leading to:
- **Semantic drift** between raw and language-tagged prompts
- **Inconsistent performance** across high-resource vs. low-resource languages
- **Translation artifacts** that affect answer quality

### Experimental Design

#### 1. **Dataset Selection: XQuAD**
- **Source**: Cross-lingual Question Answering Dataset
- **Coverage**: 1,190 question-answer pairs across 11 languages
- **Languages Tested**: 
  - **High-resource**: English (`en`), Spanish (`es`)
  - **Low-resource**: Thai (`th`), Arabic (`ar`)
- **Format**: Professional translations of SQuAD v1.1 questions

#### 2. **Evaluation Metrics**

##### Primary Metrics:
- **Sentence-Embedding Similarity**: Cosine similarity between raw and tagged generations
- **BERTScore F1**: Semantic consistency and answer quality against references
- **Exact Match & Token F1**: SQuAD-style overlap with gold answers
- **ROUGE-L**: Recall-focused overlap for longer responses

##### Baseline Comparison:
- **mBERT F1 scores** (multilingual BERT)
- **XLM-R Large F1 scores** (cross-lingual RoBERTa)

#### 3. **Experimental Setup**

```python
# Raw Prompt
"Context: [context_text]
Question: [question_text]
Answer:"

# Tagged Prompt  
"[lang=es] Context: [context_text]
Question: [question_text]
Answer:"
```

---

## 🛠️ Technical Implementation

### Architecture

```
experiment.py
├── Data Loading (XQuAD Dataset)
├── Model Configuration (Gemini 2.0 Flash)
├── Prompt Generation (Raw vs Tagged)
├── Evaluation Pipeline
│   ├── Sentence-embedding semantic alignment
│   ├── BERTScore, Exact Match, Token F1, ROUGE-L
│   ├── Paired statistical analysis utilities
│   └── Baseline Comparison
└── Results Analysis & Visualization
```

### Key Components

#### 1. **Dataset Handling**
```python
def download_xquad_dataset(languages: List[str]) -> Dict[str, str]:
    """Downloads XQuAD files for specified languages"""
    
def load_xquad_data(file_path: str, sample_size: int) -> List[Dict]:
    """Loads and preprocesses XQuAD samples"""
```

#### 2. **Model Integration**
- **Model**: Google Gemini 2.0 Flash
- **Safety Settings**: Configured to allow academic research content
- **Temperature**: 0.1 (for consistent results)
- **Max Tokens**: 200 (for concise answers)

#### 3. **Evaluation Framework**
```python
def evaluate_xquad_sample(sample: Dict, lang: str) -> Dict:
    """
    Evaluates semantic consistency between raw and tagged outputs
    Returns: embedding-based semantic similarity plus BERTScore, overlap, and quality metrics
    """
```

---

## 📊 Results & Analysis

> **Note**: The previous KL-divergence-based results were retired on 2025-10-25. Re-run `experiment.py` to regenerate metrics with the updated evaluation stack.

### Baseline Performance (XQuAD Paper)

| Model         | EN   | ES   | TH   | AR   | Avg  |
|---------------|------|------|------|------|------|
| mBERT         | 83.5 | 75.5 | 42.7 | 61.5 | 65.8 |
| XLM-R Large   | 86.5 | 82.0 | 74.2 | 68.6 | 77.8 |

### Experimental Findings

#### Language Performance Analysis
- **High-resource languages** (EN, ES) show different consistency patterns than **low-resource languages** (TH, AR)
- **Thai** performance significantly lower than other languages (baseline F1: 42.7)
- **Arabic** shows moderate performance (baseline F1: 61.5)

#### Semantic Consistency Insights
1. **Embedding-space similarity** captures language-specific semantic drift
2. **BERTScore F1** consistency varies by language resource level
3. **Token-level and ROUGE metrics** surface complementary quality signals across language families

---

## 🔧 Setup & Usage

### Prerequisites
```bash
pip install torch numpy bert-score google-generativeai python-dotenv requests sentence-transformers rouge-score scipy
```

> **Windows note**: Hugging Face caches work best when symlinks are enabled. Either run Python as an administrator, enable Windows Developer Mode, or install `huggingface_hub[hf_xet]` to silence degraded-cache warnings.

### Environment Configuration
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_api_key_here
```

### Running the Experiment
```bash
cd exploring_idea
python experiment.py --sample-size 50 --seed 42 --save-samples-dir per_sample_outputs
```

- `--sample-size`: controls how many XQuAD items per language are evaluated (defaults to 50).
- `--seed`: shuffles the dataset deterministically so additional runs explore different samples.
- `--runs` and `--seed-step`: repeat the full evaluation multiple times, incrementing the seed each run.
- `--save-samples-dir`: persist per-sample JSON dumps (including raw/tagged answers) for deeper error analysis.

### Output Files
- `xquad_evaluation_results.json`: Aggregated metrics per language (or `xquad_evaluation_results_run_XX.json` when using `--runs`)
- `per_sample_outputs/<lang>_samples.json`: Optional per-sample metrics and generations (enabled via `--save-samples-dir`)
- Console output: Real-time evaluation progress and summaries

---

## 📁 Project Structure

```
exploring_idea/
├── experiment.py              # Main evaluation script
├── test_simple.py            # API testing utilities
├── requirements.txt          # Python dependencies
├── PROJECT_README.md         # This documentation
├── XQuAD_ReadMe.md          # XQuAD dataset documentation
├── xquad_data/              # Downloaded dataset files
│   ├── xquad.en.json
│   ├── xquad.es.json
│   ├── xquad.th.json
│   └── xquad.ar.json
└── xquad_evaluation_results.json  # Experiment results
```

---

## 🎯 Key Research Questions Addressed

### 1. **Does Language Tagging Reduce Semantic Drift?**
- **Method**: Sentence-embedding cosine similarity plus BERTScore F1 comparisons between raw and tagged generations
- **Expectation**: Positive tagged-minus-raw deltas indicate improved semantic alignment

### 2. **Do High-Resource Languages Benefit More from Tagging?**
- **Method**: Comparing EN/ES vs TH/AR performance improvements
- **Hypothesis**: High-resource languages may show more stable patterns

### 3. **How Does Tagged Performance Compare to SOTA Baselines?**
- **Method**: BERTScore F1 comparison with mBERT and XLM-R
- **Goal**: Understand if language tagging bridges performance gaps

---

## 🔍 Technical Challenges & Solutions

### Challenge 1: API Safety Filters
**Problem**: Gemini API blocking legitimate academic content
**Solution**: Implemented comprehensive safety setting configuration

### Challenge 2: Logit Access Limitations
**Problem**: Gemini API doesn't expose token-level logits
**Solution**: Replaced KL-divergence estimates with embedding-based semantic similarity and BERTScore comparisons

### Challenge 3: Multilingual BERTScore Evaluation
**Problem**: Language-specific tokenization and scoring
**Solution**: Used XLM-RoBERTa-large for cross-lingual semantic evaluation

---

## 📈 Expected Impact & Applications

### Academic Contributions
- **Bias Detection**: Quantitative framework for measuring multilingual LLM bias
- **Evaluation Methodology**: Reusable pipeline for cross-lingual consistency testing
- **Language Resource Analysis**: Insights into high vs. low resource language performance

### Practical Applications
- **Multilingual Chatbots**: Improved consistency across languages
- **Translation Quality**: Better semantic preservation in automated translation
- **Cross-lingual Information Retrieval**: More accurate multilingual search systems

---

## 🚀 Future Work

### Short-term Extensions
1. **Expand Language Coverage**: Test additional language families (African, Asian, European)
2. **Domain Variation**: Evaluate on different question types (factual, reasoning, opinion)
3. **Model Comparison**: Test with other LLMs (GPT-4, Claude, etc.)

### Long-term Research Directions
1. **Fine-tuning Experiments**: Train models with language tag awareness
2. **Prompt Engineering**: Develop more sophisticated tagging strategies
3. **Real-world Deployment**: Test in production multilingual applications

---

## 📚 References & Related Work

### Primary Dataset
- **XQuAD**: Artetxe, M., Ruder, S., & Yogatama, D. (2019). *On the cross-lingual transferability of monolingual representations*. arXiv preprint arXiv:1910.11856.

### Baseline Models
- **mBERT**: Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
- **XLM-RoBERTa**: Conneau, A., et al. (2019). *Unsupervised Cross-lingual Representation Learning at Scale*.

### Evaluation Framework
- **BERTScore**: Zhang, T., et al. (2019). *BERTScore: Evaluating Text Generation with BERT*.

---

## 👥 Contributing

This research is part of an ongoing investigation into multilingual LLM behavior. Contributions and suggestions are welcome through:

1. **Issue Reports**: Document any bugs or unexpected behavior
2. **Method Improvements**: Suggest better evaluation metrics or experimental designs
3. **Language Extensions**: Add support for additional languages or datasets

---

## 📄 License

This project is released under the MIT License. The XQuAD dataset is distributed under the CC BY-SA 4.0 license.

---

## 🔗 Contact & Collaboration

For questions, collaborations, or access to detailed results, please reach out through the project repository or academic channels.

**Note**: This project represents ongoing research and findings should be interpreted within the context of the experimental limitations and scope described above.
