import argparse
import os
import random
import numpy as np
import requests # Keeping in case for other API calls, though not directly used for Gemini
import json
from bert_score import score as bert_score
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv # For loading .env file
import urllib.request
from pathlib import Path
import math # For math.isnan, alternative to np.isnan for single floats
import re
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from stats_analysis import paired_summary

# Cell 3: Configure Gemini API and Load Environment Variables
# Load environment variables from .env file.
# Assuming .env is in the parent directory of where the script is run, or current directory.
# Adjust the path as needed for your project structure.
# For a typical setup where script is in 'src/' and .env is in project root:
# load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
# For simplicity, if running directly from a cell, assumes .env in current or immediate parent.
load_dotenv() # This loads from .env in current or parent directory by default

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ Error: Please set the GEMINI_API_KEY environment variable in your .env file or system.")
genai.configure(api_key=GEMINI_API_KEY)

# --- Dynamic Model Selection ---
SELECTED_MODEL = None
try:
    print("⏳ Listing available Gemini models...")
    # List models that support text generation (generateContent)
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

    if available_models:
        # Force use of gemini-2.0-flash specifically
        preferred_model = 'gemini-2.0-flash'
        
        # Check if the full model name exists (it might be models/gemini-2.0-flash)
        if preferred_model in available_models:
            SELECTED_MODEL = preferred_model
        elif f'models/{preferred_model}' in available_models:
            SELECTED_MODEL = f'models/{preferred_model}'
        else:
            # Search for any model containing "gemini-2.0-flash"
            for model in available_models:
                if 'gemini-2.0-flash' in model:
                    SELECTED_MODEL = model
                    break
        
        if SELECTED_MODEL:
            print(f"✅ Selected Gemini model: {SELECTED_MODEL}")
        else:
            print(f"❌ gemini-2.0-flash not found in available models: {available_models}")
            # Fallback to a text model (not vision)
            text_models = [m for m in available_models if 'vision' not in m.lower()]
            if text_models:
                SELECTED_MODEL = text_models[0]
                print(f"✅ Using fallback text model: {SELECTED_MODEL}")
            else:
                raise ValueError("No suitable text models found")
    else:
        raise ValueError("No Gemini models found that support 'generateContent'. Check your API key and permissions.")

except Exception as e:
    print(f"❌ Error during Gemini API setup or model listing: {e}")
    print("Please ensure your GEMINI_API_KEY is correctly set and active.")
    # Fallback to gemini-2.0-flash directly
    SELECTED_MODEL = "gemini-2.0-flash" 
    print(f"Attempting to use default model '{SELECTED_MODEL}'")

# Gemini's public API does not expose token-level logits, so downstream drift analysis relies on
# embedding-space similarity rather than simulated probabilities.

_SEMANTIC_MODEL: Optional[SentenceTransformer] = None


def get_semantic_model() -> SentenceTransformer:
    """Lazily instantiates the sentence-level encoder used for semantic comparisons."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        _SEMANTIC_MODEL = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return _SEMANTIC_MODEL


def embed_text(text: str) -> np.ndarray:
    if not text:
        return np.zeros((get_semantic_model().get_sentence_embedding_dimension(),), dtype=np.float32)
    model = get_semantic_model()
    return model.encode(text)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if not a.size or not b.size:
        return float('nan')
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return float('nan')
    similarity = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
    return float(similarity)


def generate_text(prompt: str) -> str:
    if not SELECTED_MODEL:
        print("🔴 No valid Gemini model selected. Cannot generate content.")
        return ""

    try:
        model = genai.GenerativeModel(SELECTED_MODEL)
        safety_settings = [
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
            }
        ]

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=200
            )
        )

        generated_text = ""
        if response and response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                generated_text = candidate.content.parts[0].text
            elif hasattr(candidate, 'text'):
                generated_text = candidate.text

        if not generated_text:
            print(f"⚠️ Warning: Gemini model '{SELECTED_MODEL}' returned no text for prompt: '{prompt[:100]}...'")
            if response and response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    print(f"   Finish reason: {candidate.finish_reason}")
                if hasattr(candidate, 'safety_ratings'):
                    print(f"   Safety ratings: {candidate.safety_ratings}")
            return ""

        return generated_text

    except Exception as e:
        print(f"❌ Error generating content with Gemini API (model: {SELECTED_MODEL}): {e}")
        return ""

# Cell 3.5: XQuAD Dataset Download and Loading Functions (Existing, no change)
def download_xquad_dataset(languages: List[str], data_dir: str = "xquad_data") -> Dict[str, str]:
    """
    Downloads XQuAD dataset files for specified languages.
    
    Args:
        languages: List of language codes (e.g., ['en', 'fr', 'de', 'ar'])
        data_dir: Directory to save the dataset files
    
    Returns:
        Dictionary mapping language codes to file paths
    """
    base_url = "https://raw.githubusercontent.com/google-deepmind/xquad/master/"
    
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True) # Use parents=True for nested creation
    
    file_paths = {}
    
    for lang in languages:
        filename = f"xquad.{lang}.json"
        url = base_url + filename
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"⬇️ Downloading XQuAD dataset for {lang}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"✅ Successfully downloaded {filename}")
            except Exception as e:
                print(f"❌ Error downloading {filename}: {e}")
                continue
        else:
            print(f"☑️ {filename} already exists, skipping download")
        
        file_paths[lang] = file_path
    
    return file_paths

def load_xquad_data(file_path: str, sample_size: int = 50, seed: Optional[int] = None) -> List[Dict]:
    """
    Loads XQuAD data from a JSON file and extracts question-context pairs.
    
    Args:
        file_path: Path to the XQuAD JSON file
        sample_size: Number of samples to extract (for efficient testing)
    
    Returns:
        List of dictionaries with 'context', 'question', 'answers', and 'id' keys
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []

        for article in data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    # Ensure at least one answer exists to form a prompt/label
                    if qa['answers']:
                        samples.append({
                            'context': context,
                            'question': qa['question'],
                            'answers': qa['answers'], # Keep all answers for evaluation
                            'id': qa['id']
                        })

        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(samples)

        if sample_size > 0:
            samples = samples[:sample_size]

        print(f"📊 Loaded {len(samples)} samples from {file_path}")
        return samples
    
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []

def create_qa_prompt(context: str, question: str) -> str:
    """
    Creates a prompt for question answering given context and question.
    
    Args:
        context: The context paragraph
        question: The question to answer
    
    Returns:
        Formatted prompt string
    """
    return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


def normalize_answer(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction_tokens: List[str], reference_tokens: List[str]) -> float:
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = set(prediction_tokens) & set(reference_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, references: List[str]) -> float:
    if not prediction or not references:
        return 0.0
    prediction_norm = normalize_answer(prediction)
    if not prediction_norm:
        return 0.0
    for reference in references:
        if normalize_answer(reference) == prediction_norm:
            return 1.0
    return 0.0


def compute_token_f1(prediction: str, references: List[str]) -> float:
    if not prediction or not references:
        return 0.0
    prediction_tokens = normalize_answer(prediction).split()
    scores = []
    for reference in references:
        reference_tokens = normalize_answer(reference).split()
        scores.append(token_f1(prediction_tokens, reference_tokens))
    return max(scores) if scores else 0.0


def compute_rouge_l(prediction: str, references: List[str]) -> float:
    if not prediction or not references:
        return 0.0
    scores = []
    for reference in references:
        try:
            rouge_result = ROUGE_SCORER.score(reference, prediction)
            scores.append(rouge_result['rougeL'].fmeasure)
        except Exception as exc:
            print(f"⚠️ Error computing ROUGE-L: {exc}")
    return max(scores) if scores else 0.0


def compute_semantic_similarities(prediction_embedding: np.ndarray, reference_embeddings: List[np.ndarray]) -> float:
    if reference_embeddings:
        similarities = [cosine_sim(prediction_embedding, ref_emb) for ref_emb in reference_embeddings]
        similarities = [s for s in similarities if not math.isnan(s)]
        if similarities:
            return float(max(similarities))
    return float('nan')


def sanitize_for_json(value):
    if isinstance(value, float):
        return None if math.isnan(value) else float(value)
    if isinstance(value, dict):
        return {key: sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def print_low_metric_samples(lang: str, lang_results: List[Dict], metric_key: str, top_k: int = 5) -> None:
    """Prints the lowest tagged-minus-raw delta samples for a metric to aid debugging."""
    raw_key = f"{metric_key}_raw"
    tagged_key = f"{metric_key}_tagged"
    deltas: List[Tuple[float, float, float, Dict]] = []

    for record in lang_results:
        raw_val = record.get(raw_key)
        tagged_val = record.get(tagged_key)
        if raw_val is None or tagged_val is None:
            continue
        if isinstance(raw_val, float) and math.isnan(raw_val):
            continue
        if isinstance(tagged_val, float) and math.isnan(tagged_val):
            continue
        deltas.append((tagged_val - raw_val, raw_val, tagged_val, record))

    if not deltas:
        return

    deltas.sort(key=lambda item: item[0])
    print(f"  🔍 Lowest {metric_key.upper()} delta samples (tagged - raw):")
    for diff, raw_val, tagged_val, record in deltas[:top_k]:
        print(
            f"    • ID {record['sample_id']}: Δ={diff:.4f}, raw={raw_val:.4f}, tagged={tagged_val:.4f}"
        )


# Cell 8: XQuAD Evaluation Functions
def evaluate_xquad_sample(sample: Dict, lang: str) -> Dict:
    raw_prompt = create_qa_prompt(sample['context'], sample['question'])
    tagged_prompt = f"[lang={lang}] " + raw_prompt

    raw_output_text = generate_text(raw_prompt)
    tagged_output_text = generate_text(tagged_prompt)

    expected_answers = [ans['text'] for ans in sample['answers']]

    raw_embedding = embed_text(raw_output_text)
    tagged_embedding = embed_text(tagged_output_text)
    answer_embeddings = [embed_text(answer) for answer in expected_answers]

    metrics = {
        "sample_id": sample['id'],
        "raw_prompt": raw_prompt,
        "tagged_prompt": tagged_prompt,
        "raw_text_output": raw_output_text,
        "tagged_text_output": tagged_output_text,
        "expected_answers": expected_answers,
        "semantic_similarity_raw_vs_tagged": cosine_sim(raw_embedding, tagged_embedding),
        "semantic_similarity_raw_to_answer": compute_semantic_similarities(raw_embedding, answer_embeddings),
        "semantic_similarity_tagged_to_answer": compute_semantic_similarities(tagged_embedding, answer_embeddings),
        "bertscore_f1_consistency": float('nan'),
        "bertscore_f1_raw_to_answer": float('nan'),
        "bertscore_f1_tagged_to_answer": float('nan'),
        "exact_match_raw": compute_exact_match(raw_output_text, expected_answers),
        "exact_match_tagged": compute_exact_match(tagged_output_text, expected_answers),
        "token_f1_raw": compute_token_f1(raw_output_text, expected_answers),
        "token_f1_tagged": compute_token_f1(tagged_output_text, expected_answers),
        "rouge_l_raw": compute_rouge_l(raw_output_text, expected_answers),
        "rouge_l_tagged": compute_rouge_l(tagged_output_text, expected_answers),
    }

    if raw_output_text and tagged_output_text:
        try:
            _, _, F1_consistency = bert_score([tagged_output_text], [raw_output_text], lang=lang, model_type="xlm-roberta-large")
            metrics["bertscore_f1_consistency"] = F1_consistency.item()
        except Exception as exc:
            print(f"⚠️ Error computing BERTScore for consistency: {exc}")

    if raw_output_text and expected_answers:
        try:
            _, _, F1_raw_answer = bert_score([raw_output_text], expected_answers, lang=lang, model_type="xlm-roberta-large")
            metrics["bertscore_f1_raw_to_answer"] = max(F1_raw_answer).item()
        except Exception as exc:
            print(f"⚠️ Error computing BERTScore for raw answer: {exc}")

    if tagged_output_text and expected_answers:
        try:
            _, _, F1_tagged_answer = bert_score([tagged_output_text], expected_answers, lang=lang, model_type="xlm-roberta-large")
            metrics["bertscore_f1_tagged_to_answer"] = max(F1_tagged_answer).item()
        except Exception as exc:
            print(f"⚠️ Error computing BERTScore for tagged answer: {exc}")

    return metrics


def run_xquad_evaluation(
    languages: List[str],
    sample_size: int = 50,
    seed: Optional[int] = None,
    save_samples_dir: Optional[str] = None,
) -> Dict:
    """
    Runs comprehensive evaluation on XQuAD dataset for multiple languages.
    
    Args:
        languages: List of language codes to evaluate
        sample_size: Number of samples per language to load from XQuAD (for efficient testing)
    
    Returns:
        Dictionary with aggregated and individual results for each language.
    """
    print(f"\n🚀 Starting XQuAD evaluation for languages: {', '.join(languages)}")
    # Download datasets
    file_paths = download_xquad_dataset(languages)
    
    results = {}
    
    for lang in languages:
        if lang not in file_paths:
            print(f"❌ Skipping {lang} - dataset file not found.")
            continue
        
        print(f"\n{'='*60}")
        print(f"📚 Evaluating language: {lang.upper()} ({sample_size} samples)")
        print(f"{'='*60}")
        
        # Load data
        lang_seed = None if seed is None else seed + (abs(hash(lang)) % 10000)
        samples = load_xquad_data(file_paths[lang], sample_size, seed=lang_seed)
        if not samples:
            print(f"⚠️ No samples loaded for {lang}. Skipping.")
            continue
        
        # Evaluate samples
        lang_results = []
        for i, sample in enumerate(samples):
            print(f"Processing sample {i+1}/{len(samples)} for {lang.upper()} (ID: {sample['id']})...")
            result = evaluate_xquad_sample(sample, lang)
            lang_results.append(result)
        
        # Calculate aggregated metrics while filtering NaNs
        def valid_values(key: str) -> List[float]:
            values = []
            for record in lang_results:
                value = record.get(key)
                if value is None:
                    continue
                if isinstance(value, float) and math.isnan(value):
                    continue
                values.append(value)
            return values

        bert_consistency = valid_values('bertscore_f1_consistency')
        bert_raw = valid_values('bertscore_f1_raw_to_answer')
        bert_tagged = valid_values('bertscore_f1_tagged_to_answer')
        semantic_alignment = valid_values('semantic_similarity_raw_vs_tagged')
        semantic_raw = valid_values('semantic_similarity_raw_to_answer')
        semantic_tagged = valid_values('semantic_similarity_tagged_to_answer')
        exact_match_raw = valid_values('exact_match_raw')
        exact_match_tagged = valid_values('exact_match_tagged')
        token_f1_raw = valid_values('token_f1_raw')
        token_f1_tagged = valid_values('token_f1_tagged')
        rouge_raw = valid_values('rouge_l_raw')
        rouge_tagged = valid_values('rouge_l_tagged')

        def paired_values(raw_key: str, tagged_key: str) -> List[Tuple[float, float]]:
            pairs: List[Tuple[float, float]] = []
            for record in lang_results:
                raw_value = record.get(raw_key)
                tagged_value = record.get(tagged_key)
                if raw_value is None or tagged_value is None:
                    continue
                if isinstance(raw_value, float) and math.isnan(raw_value):
                    continue
                if isinstance(tagged_value, float) and math.isnan(tagged_value):
                    continue
                pairs.append((raw_value, tagged_value))
            return pairs

        statistical_tests = {
            "semantic_similarity_to_answer": paired_summary(paired_values('semantic_similarity_raw_to_answer', 'semantic_similarity_tagged_to_answer')),
            "bert_f1_to_answer": paired_summary(paired_values('bertscore_f1_raw_to_answer', 'bertscore_f1_tagged_to_answer')),
            "exact_match": paired_summary(paired_values('exact_match_raw', 'exact_match_tagged')),
            "token_f1": paired_summary(paired_values('token_f1_raw', 'token_f1_tagged')),
            "rouge_l": paired_summary(paired_values('rouge_l_raw', 'rouge_l_tagged')),
        }

        aggregated = {
            "language": lang,
            "num_samples_processed": len(lang_results),
            "avg_semantic_similarity_raw_vs_tagged": float(np.mean(semantic_alignment)) if semantic_alignment else float('nan'),
            "std_semantic_similarity_raw_vs_tagged": float(np.std(semantic_alignment)) if semantic_alignment else float('nan'),
            "avg_semantic_similarity_raw_to_answer": float(np.mean(semantic_raw)) if semantic_raw else float('nan'),
            "std_semantic_similarity_raw_to_answer": float(np.std(semantic_raw)) if semantic_raw else float('nan'),
            "avg_semantic_similarity_tagged_to_answer": float(np.mean(semantic_tagged)) if semantic_tagged else float('nan'),
            "std_semantic_similarity_tagged_to_answer": float(np.std(semantic_tagged)) if semantic_tagged else float('nan'),
            "avg_bert_f1_consistency": float(np.mean(bert_consistency)) if bert_consistency else float('nan'),
            "std_bert_f1_consistency": float(np.std(bert_consistency)) if bert_consistency else float('nan'),
            "avg_bert_f1_raw_to_answer": float(np.mean(bert_raw)) if bert_raw else float('nan'),
            "std_bert_f1_raw_to_answer": float(np.std(bert_raw)) if bert_raw else float('nan'),
            "avg_bert_f1_tagged_to_answer": float(np.mean(bert_tagged)) if bert_tagged else float('nan'),
            "std_bert_f1_tagged_to_answer": float(np.std(bert_tagged)) if bert_tagged else float('nan'),
            "avg_exact_match_raw": float(np.mean(exact_match_raw)) if exact_match_raw else float('nan'),
            "avg_exact_match_tagged": float(np.mean(exact_match_tagged)) if exact_match_tagged else float('nan'),
            "avg_token_f1_raw": float(np.mean(token_f1_raw)) if token_f1_raw else float('nan'),
            "avg_token_f1_tagged": float(np.mean(token_f1_tagged)) if token_f1_tagged else float('nan'),
            "avg_rouge_l_raw": float(np.mean(rouge_raw)) if rouge_raw else float('nan'),
            "avg_rouge_l_tagged": float(np.mean(rouge_tagged)) if rouge_tagged else float('nan'),
            "individual_results": lang_results,
            "statistical_tests": statistical_tests,
        }
        
        results[lang] = aggregated

        if save_samples_dir:
            samples_path = Path(save_samples_dir)
            samples_path.mkdir(parents=True, exist_ok=True)
            per_language_file = samples_path / f"{lang}_samples.json"
            try:
                sanitized_samples = [sanitize_for_json(record) for record in lang_results]
                with open(per_language_file, 'w', encoding='utf-8') as sample_file:
                    json.dump(sanitized_samples, sample_file, indent=2, ensure_ascii=False)
                print(f"  💾 Saved per-sample details to {per_language_file}")
            except Exception as exc:
                print(f"⚠️ Unable to save per-sample results for {lang}: {exc}")

        # Print summary for the language
        print(f"\n✨ Results for {lang.upper()}:")
        print(f"  Samples evaluated: {aggregated['num_samples_processed']}")
        print(f"  Avg Semantic Similarity (Raw vs. Tagged): {aggregated['avg_semantic_similarity_raw_vs_tagged']:.4f} ± {aggregated['std_semantic_similarity_raw_vs_tagged']:.4f}")
        print(f"  Avg Semantic Similarity (Raw → Answer): {aggregated['avg_semantic_similarity_raw_to_answer']:.4f} ± {aggregated['std_semantic_similarity_raw_to_answer']:.4f}")
        print(f"  Avg Semantic Similarity (Tagged → Answer): {aggregated['avg_semantic_similarity_tagged_to_answer']:.4f} ± {aggregated['std_semantic_similarity_tagged_to_answer']:.4f}")
        print(f"  Avg BERTScore F1 Consistency: {aggregated['avg_bert_f1_consistency']:.4f} ± {aggregated['std_bert_f1_consistency']:.4f}")
        print(f"  Avg BERTScore F1 (Raw → Answer): {aggregated['avg_bert_f1_raw_to_answer']:.4f} ± {aggregated['std_bert_f1_raw_to_answer']:.4f}")
        print(f"  Avg BERTScore F1 (Tagged → Answer): {aggregated['avg_bert_f1_tagged_to_answer']:.4f} ± {aggregated['std_bert_f1_tagged_to_answer']:.4f}")
        print(f"  Avg Exact Match (Raw vs Tagged): {aggregated['avg_exact_match_raw']:.4f} | {aggregated['avg_exact_match_tagged']:.4f}")
        print(f"  Avg Token F1 (Raw vs Tagged): {aggregated['avg_token_f1_raw']:.4f} | {aggregated['avg_token_f1_tagged']:.4f}")
        print(f"  Avg ROUGE-L (Raw vs Tagged): {aggregated['avg_rouge_l_raw']:.4f} | {aggregated['avg_rouge_l_tagged']:.4f}")

        # Provide quick diagnostic listings for overlap-sensitive metrics
        print_low_metric_samples(lang, lang_results, "rouge_l")
        print_low_metric_samples(lang, lang_results, "token_f1")
    
    return results


# Cell 9: Baseline Comparison Functions (Existing, no change)
def get_xquad_baseline_scores() -> Dict[str, Dict[str, float]]:
    """
    Returns baseline F1 and EM scores from XQuAD paper for comparison.
    These are the mBERT and XLM-R Large zero-shot transfer results.
    """
    return {
        "mBERT_f1": {
            "en": 83.5, "ar": 61.5, "de": 70.6, "el": 62.6, "es": 75.5,
            "hi": 59.2, "ru": 71.3, "th": 42.7, "tr": 55.4, "vi": 69.5, "zh": 58.0
        },
        "xlm_r_large_f1": {
            "en": 86.5, "ar": 68.6, "de": 80.4, "el": 79.8, "es": 82.0,
            "hi": 76.7, "ru": 80.1, "th": 74.2, "tr": 75.9, "vi": 79.1, "zh": 59.3
        },
        "mBERT_em": {
            "en": 72.2, "ar": 45.1, "de": 54.0, "el": 44.9, "es": 56.9,
            "hi": 46.0, "ru": 53.3, "th": 33.5, "tr": 40.1, "vi": 49.6, "zh": 48.3
        },
        "xlm_r_large_em": {
            "en": 75.7, "ar": 49.0, "de": 63.4, "el": 61.7, "es": 63.9,
            "hi": 59.7, "ru": 64.3, "th": 62.8, "tr": 59.3, "vi": 59.0, "zh": 50.0
        }
    }

def compare_with_baselines(results: Dict, baseline_scores: Dict) -> None:
    """
    Compares our semantic consistency results (BERTScore F1 for tagged output to answer)
    with XQuAD baselines (F1 scores).

    Args:
        results: Our evaluation results (from run_xquad_evaluation)
        baseline_scores: Baseline F1 scores from XQuAD paper
    """
    print(f"\n{'='*70}")
    print("🎯 COMPARISON OF TAGGED OUTPUT PERFORMANCE WITH XQUAD BASELINES")
    print(" (Our Avg BERTScore F1 of Tagged Output to Expected Answer vs. Baseline QA F1)")
    print(f"{'='*70}")
    
    # Header adjusted to reflect comparison
    print(f"{'Language':<10} {'Our Tagged F1':<18} {'mBERT F1':<12} {'XLM-R F1':<12} {'Gap (vs mBERT)':<17} {'Gap (vs XLM-R)':<17}")
    print("-" * 100)
    
    for lang, result in results.items():
        if lang in baseline_scores["mBERT_f1"]: # Check if baseline exists for this language
            our_tagged_f1_score = result["avg_bert_f1_tagged_to_answer"]
            mbert_f1 = baseline_scores["mBERT_f1"][lang]
            xlmr_f1 = baseline_scores["xlm_r_large_f1"][lang]
            
            # Calculate performance gap relative to baselines
            # Multiply our score by 100 for percentage comparison with baselines
            our_score_percent = our_tagged_f1_score * 100 if not math.isnan(our_tagged_f1_score) else float('nan')
            
            gap_mbert = our_score_percent - mbert_f1 if not math.isnan(our_score_percent) else float('nan')
            gap_xlmr = our_score_percent - xlmr_f1 if not math.isnan(our_score_percent) else float('nan')
            
            print(f"{lang.upper():<10} {our_tagged_f1_score:.3f} ({our_score_percent:.1f}%) {mbert_f1:<12.1f} {xlmr_f1:<12.1f} {gap_mbert:+.1f} {gap_xlmr:+.1f}")
        else:
            print(f"{lang.upper():<10} N/A (No baseline data)")

# Cell 10: Main Experiment Runner
def run_comprehensive_xquad_experiment(
    sample_size: int = 50,
    seed: Optional[int] = None,
    save_samples_dir: Optional[str] = None,
):
    """
    Runs the complete XQuAD experiment for selected languages,
    categorized by their typical resource levels.
    """
    # Languages chosen based on typical resource levels and XQuAD coverage
    # Top languages: English (en), Spanish (es) - generally high-resource
    # Underrepresented: Thai (th), Arabic (ar) - generally lower-resource in NLP
    
    top_languages = ["en", "es"]
    underrepresented_languages = ["th", "ar"]
    
    print("✨ Starting Comprehensive XQuAD Evaluation for Implicit Translation Bias Mitigation ✨")
    print("Hypothesis: Explicit language tags will improve semantic consistency and translation accuracy in LLM outputs.")
    
    all_languages_to_evaluate = top_languages + underrepresented_languages
    # Reduced sample size for faster testing. Increase for more robust results.
    evaluation_sample_size = sample_size 
    
    # Run evaluation on the selected languages
    results = run_xquad_evaluation(
        all_languages_to_evaluate,
        sample_size=evaluation_sample_size,
        seed=seed,
        save_samples_dir=save_samples_dir,
    )
    
    # Get baseline scores from XQuAD paper
    baseline_scores = get_xquad_baseline_scores()
    
    # Compare our results (tagged output's F1 to ground truth answer) with baselines
    compare_with_baselines(results, baseline_scores)
    
    # Analysis by language category (semantic consistency between raw and tagged outputs)
    print(f"\n{'='*60}")
    print("📊 ANALYSIS OF SEMANTIC CONSISTENCY (RAW vs. TAGGED) BY LANGUAGE CATEGORY")
    print(" (Higher BERTScore F1 Consistency indicates less semantic drift due to tagging)")
    print(f"{'='*60}")
    
    top_consistency_scores = []
    under_consistency_scores = []
    
    for lang in top_languages:
        if lang in results and not math.isnan(results[lang]["avg_bert_f1_consistency"]):
            top_consistency_scores.append(results[lang]["avg_bert_f1_consistency"])
    
    for lang in underrepresented_languages:
        if lang in results and not math.isnan(results[lang]["avg_bert_f1_consistency"]):
            under_consistency_scores.append(results[lang]["avg_bert_f1_consistency"])
    
    if top_consistency_scores:
        print(f"Top languages (Avg BERTScore F1 Consistency): {np.mean(top_consistency_scores):.4f} ± {np.std(top_consistency_scores):.4f}")
    else:
        print("No consistency data for top languages.")

    if under_consistency_scores:
        print(f"Underrepresented languages (Avg BERTScore F1 Consistency): {np.mean(under_consistency_scores):.4f} ± {np.std(under_consistency_scores):.4f}")
    else:
        print("No consistency data for underrepresented languages.")
        
    if top_consistency_scores and under_consistency_scores:
        # Simple comparison of means for consistency
        diff_consistency = np.mean(top_consistency_scores) - np.mean(under_consistency_scores)
        print(f"Difference in consistency (Top - Underrepresented): {diff_consistency:+.4f} "
              f"(Positive implies higher consistency for top languages)")
    
    return results


def save_results_to_file(results: Dict, output_file: Path) -> None:
    """Serialize aggregated results to JSON, handling NaNs gracefully."""
    try:
        serializable_results = {}
        for lang, result in results.items():
            record = {
                "language": result["language"],
                "num_samples_processed": result["num_samples_processed"],
                "avg_semantic_similarity_raw_vs_tagged": result["avg_semantic_similarity_raw_vs_tagged"],
                "std_semantic_similarity_raw_vs_tagged": result["std_semantic_similarity_raw_vs_tagged"],
                "avg_semantic_similarity_raw_to_answer": result["avg_semantic_similarity_raw_to_answer"],
                "std_semantic_similarity_raw_to_answer": result["std_semantic_similarity_raw_to_answer"],
                "avg_semantic_similarity_tagged_to_answer": result["avg_semantic_similarity_tagged_to_answer"],
                "std_semantic_similarity_tagged_to_answer": result["std_semantic_similarity_tagged_to_answer"],
                "avg_bert_f1_consistency": result["avg_bert_f1_consistency"],
                "std_bert_f1_consistency": result["std_bert_f1_consistency"],
                "avg_bert_f1_raw_to_answer": result["avg_bert_f1_raw_to_answer"],
                "std_bert_f1_raw_to_answer": result["std_bert_f1_raw_to_answer"],
                "avg_bert_f1_tagged_to_answer": result["avg_bert_f1_tagged_to_answer"],
                "std_bert_f1_tagged_to_answer": result["std_bert_f1_tagged_to_answer"],
                "avg_exact_match_raw": result["avg_exact_match_raw"],
                "avg_exact_match_tagged": result["avg_exact_match_tagged"],
                "avg_token_f1_raw": result["avg_token_f1_raw"],
                "avg_token_f1_tagged": result["avg_token_f1_tagged"],
                "avg_rouge_l_raw": result["avg_rouge_l_raw"],
                "avg_rouge_l_tagged": result["avg_rouge_l_tagged"],
                "statistical_tests": result.get("statistical_tests", {}),
            }

            record = {key: sanitize_for_json(val) for key, val in record.items()}
            serializable_results[lang] = record

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ All aggregated results saved to {output_file}")
    except Exception as exc:
        print(f"❌ Error saving results to {output_file}: {exc}")


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run XQuAD implicit bias evaluation with tagging analysis.")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of samples per language to evaluate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed used for shuffling the dataset per language.")
    parser.add_argument(
        "--save-samples-dir",
        type=str,
        default=None,
        help="Directory where per-sample outputs should be saved (organized by language/run).",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of sequential runs to execute (increments the seed).")
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Increment applied to the seed between runs when --runs > 1.",
    )
    return parser.parse_args()

# Cell 11: Main Execution Block
if __name__ == "__main__":
    args = parse_cli_args()

    print("\n" + "="*70)
    print("         MULTILINGUAL LLM BIAS EVALUATION: XQuAD BENCHMARKING        ")
    print("="*70)

    total_runs = max(1, args.runs)
    for run_index in range(total_runs):
        run_suffix = f"_run_{run_index + 1:02d}" if total_runs > 1 else ""
        run_seed = None
        if args.seed is not None:
            run_seed = args.seed + (run_index * args.seed_step)

        samples_dir = None
        if args.save_samples_dir:
            samples_dir = Path(args.save_samples_dir) / (f"run_{run_index + 1:02d}" if total_runs > 1 else "run")

        if run_seed is not None:
            print(f"\n🧪 Starting run {run_index + 1}/{total_runs} with seed {run_seed}")
        else:
            print(f"\n🧪 Starting run {run_index + 1}/{total_runs}")

        xquad_results_final = run_comprehensive_xquad_experiment(
            sample_size=args.sample_size,
            seed=run_seed,
            save_samples_dir=str(samples_dir) if samples_dir else None,
        )

        output_path = Path(f"xquad_evaluation_results{run_suffix}.json")
        save_results_to_file(xquad_results_final, output_path)

    if total_runs > 1:
        print(f"\n🎉 Completed {total_runs} evaluation runs.")
    else:
        print("\n🎉 Experiment completed successfully! Review the output for insights into implicit translation bias.")