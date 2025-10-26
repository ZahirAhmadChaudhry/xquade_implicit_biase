"""
Comprehensive experiment runner that executes all experimental conditions.
Manages execution across multiple configs, languages, and seeds.
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from experiment_config import (
    PromptConfig, ExperimentSettings,
    get_all_experiment_configs, get_core_experiment_configs
)
from prompt_builder import PromptBuilder, FewShotExampleSelector
from experiment import (
    generate_text, embed_text, cosine_sim,
    compute_exact_match, compute_token_f1, compute_rouge_l,
    compute_semantic_similarities, sanitize_for_json,
    load_xquad_data, download_xquad_dataset
)
from bert_score import score as bert_score
from stats_analysis import paired_summary
import numpy as np
import math


class ComprehensiveExperimentRunner:
    """Runs full experimental matrix across conditions, languages, and seeds."""
    
    def __init__(self, settings: ExperimentSettings, use_full_matrix: bool = False):
        self.settings = settings
        self.results_dir = Path(settings.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Choose experiment configurations
        if use_full_matrix:
            self.configs = get_all_experiment_configs()
        else:
            self.configs = get_core_experiment_configs()
        
        # Download datasets
        print(f"📥 Downloading XQuAD datasets for {len(settings.languages)} languages...")
        self.file_paths = download_xquad_dataset(settings.languages)
        
        # Load all data upfront
        print(f"📊 Loading XQuAD data...")
        self.all_data = {}
        for lang in settings.languages:
            if lang in self.file_paths:
                # Load full dataset (we'll sample from it)
                self.all_data[lang] = load_xquad_data(
                    self.file_paths[lang], 
                    sample_size=0,  # 0 means load all
                    seed=None
                )
        
        print(f"✅ Ready to run {len(self.configs)} configurations × {len(settings.languages)} languages × {settings.num_seeds} seeds")
        print(f"   = {len(self.configs) * len(settings.languages) * settings.num_seeds} total runs")
        
    def run_single_condition(
        self,
        config: PromptConfig,
        language: str,
        seed: int,
        samples: List[Dict]
    ) -> Dict:
        """
        Run evaluation for a single experimental condition.
        
        Returns:
            Dictionary with aggregated metrics
        """
        print(f"\n{'='*60}")
        print(f"🔬 Config: {config.name} | Language: {language.upper()} | Seed: {seed}")
        print(f"   {config.get_description()}")
        print(f"{'='*60}")
        
        # Initialize prompt builder
        builder = PromptBuilder(config)
        
        # Set up few-shot selector if needed
        few_shot_selector = None
        if config.use_few_shot:
            few_shot_selector = FewShotExampleSelector(self.all_data[language], seed=seed)
        
        # Evaluate each sample
        sample_results = []
        for i, sample in enumerate(samples, 1):
            if i % 10 == 0:
                print(f"   Processing sample {i}/{len(samples)}...")
            
            # Get few-shot examples if needed
            few_shot_examples = None
            if config.use_few_shot and few_shot_selector:
                few_shot_examples = few_shot_selector.select_examples(
                    config.num_shots,
                    exclude_ids=[sample['id']]
                )
            
            # Build prompt
            prompt = builder.build_prompt(
                context=sample['context'],
                question=sample['question'],
                language=language,
                few_shot_examples=few_shot_examples,
                english_context=None,  # TODO: Add English context if available
            )
            
            # Generate output
            output_text = generate_text(prompt)
            
            # Compute metrics
            expected_answers = [ans['text'] for ans in sample['answers']]
            
            output_embedding = embed_text(output_text)
            answer_embeddings = [embed_text(ans) for ans in expected_answers]
            
            metrics = {
                "sample_id": sample['id'],
                "prompt": prompt,
                "output": output_text,
                "expected_answers": expected_answers,
                "semantic_similarity_to_answer": compute_semantic_similarities(output_embedding, answer_embeddings),
                "exact_match": compute_exact_match(output_text, expected_answers),
                "token_f1": compute_token_f1(output_text, expected_answers),
                "rouge_l": compute_rouge_l(output_text, expected_answers),
                "bertscore_f1": float('nan'),
            }
            
            # BERTScore
            if output_text and expected_answers:
                try:
                    _, _, F1_scores = bert_score(
                        [output_text], 
                        expected_answers, 
                        lang=language, 
                        model_type="xlm-roberta-large"
                    )
                    metrics["bertscore_f1"] = max(F1_scores).item()
                except Exception as exc:
                    print(f"⚠️ BERTScore error: {exc}")
            
            sample_results.append(metrics)
        
        # Aggregate results
        aggregated = self._aggregate_results(sample_results, config, language, seed)
        
        # Save per-sample results if requested
        if self.settings.save_samples:
            self._save_sample_results(sample_results, config, language, seed)
        
        return aggregated
    
    def _aggregate_results(
        self,
        sample_results: List[Dict],
        config: PromptConfig,
        language: str,
        seed: int
    ) -> Dict:
        """Aggregate sample-level results into summary statistics."""
        
        def valid_values(key: str) -> List[float]:
            values = []
            for record in sample_results:
                value = record.get(key)
                if value is None:
                    continue
                if isinstance(value, float) and math.isnan(value):
                    continue
                values.append(value)
            return values
        
        sem_sim = valid_values('semantic_similarity_to_answer')
        bert_f1 = valid_values('bertscore_f1')
        exact_match = valid_values('exact_match')
        token_f1 = valid_values('token_f1')
        rouge_l = valid_values('rouge_l')
        
        return {
            "config_name": config.name,
            "config_description": config.get_description(),
            "language": language,
            "seed": seed,
            "num_samples": len(sample_results),
            "avg_semantic_similarity": float(np.mean(sem_sim)) if sem_sim else float('nan'),
            "std_semantic_similarity": float(np.std(sem_sim)) if sem_sim else float('nan'),
            "avg_bertscore_f1": float(np.mean(bert_f1)) if bert_f1 else float('nan'),
            "std_bertscore_f1": float(np.std(bert_f1)) if bert_f1 else float('nan'),
            "avg_exact_match": float(np.mean(exact_match)) if exact_match else float('nan'),
            "std_exact_match": float(np.std(exact_match)) if exact_match else float('nan'),
            "avg_token_f1": float(np.mean(token_f1)) if token_f1 else float('nan'),
            "std_token_f1": float(np.std(token_f1)) if token_f1 else float('nan'),
            "avg_rouge_l": float(np.mean(rouge_l)) if rouge_l else float('nan'),
            "std_rouge_l": float(np.std(rouge_l)) if rouge_l else float('nan'),
        }
    
    def _save_sample_results(
        self,
        sample_results: List[Dict],
        config: PromptConfig,
        language: str,
        seed: int
    ):
        """Save per-sample results to JSON."""
        output_dir = self.results_dir / "samples" / config.name / language
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"seed_{seed}.json"
        sanitized = [sanitize_for_json(record) for record in sample_results]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sanitized, f, indent=2, ensure_ascii=False)
    
    def run_all(self) -> Dict:
        """
        Execute full experimental matrix.
        
        Returns:
            Complete results dictionary organized by config → language → seed
        """
        start_time = time.time()
        all_results = {}
        
        total_runs = len(self.configs) * len(self.settings.languages) * len(self.settings.get_seeds())
        current_run = 0
        
        for config in self.configs:
            config_results = {}
            
            for lang in self.settings.languages:
                if lang not in self.all_data:
                    print(f"⚠️ Skipping {lang} - no data loaded")
                    continue
                
                lang_results = {}
                
                for seed in self.settings.get_seeds():
                    current_run += 1
                    print(f"\n{'='*70}")
                    print(f"📊 Progress: {current_run}/{total_runs} runs")
                    print(f"{'='*70}")
                    
                    # Sample data with this seed
                    rng = np.random.RandomState(seed)
                    indices = rng.permutation(len(self.all_data[lang]))[:self.settings.sample_size]
                    samples = [self.all_data[lang][i] for i in indices]
                    
                    print(f"📝 Evaluating {len(samples)} samples...")
                    
                    # Run evaluation
                    result = self.run_single_condition(config, lang, seed, samples)
                    
                    lang_results[f"seed_{seed}"] = sanitize_for_json(result)
                    
                    # Save intermediate results
                    self._save_intermediate_results(all_results, config.name, lang, seed, result)
                
                config_results[lang] = lang_results
            
            all_results[config.name] = config_results
        
        # Save final aggregated results
        output_file = self.results_dir / "all_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sanitize_for_json(all_results), f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        print(f"\n✅ All experiments completed in {elapsed/3600:.2f} hours")
        print(f"📁 Results saved to: {self.results_dir}")
        
        return all_results
    
    def _save_intermediate_results(self, all_results: Dict, config_name: str, lang: str, seed: int, result: Dict):
        """Save intermediate checkpoint after each run."""
        checkpoint_file = self.results_dir / "checkpoint.json"
        
        # Update all_results structure
        if config_name not in all_results:
            all_results[config_name] = {}
        if lang not in all_results[config_name]:
            all_results[config_name][lang] = {}
        all_results[config_name][lang][f"seed_{seed}"] = result
        
        # Save checkpoint
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(sanitize_for_json(all_results), f, indent=2, ensure_ascii=False)


def main():
    """Run comprehensive experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive multilingual experiments")
    parser.add_argument("--full-matrix", action="store_true", help="Run full experiment matrix (all configs)")
    parser.add_argument("--sample-size", type=int, default=240, help="Samples per language per seed")
    parser.add_argument("--num-seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "th", "ar"], help="Languages to evaluate")
    parser.add_argument("--results-dir", type=str, default="comprehensive_results", help="Output directory")
    
    args = parser.parse_args()
    
    settings = ExperimentSettings(
        languages=args.languages,
        sample_size=args.sample_size,
        num_seeds=args.num_seeds,
        results_dir=args.results_dir,
    )
    
    runner = ComprehensiveExperimentRunner(settings, use_full_matrix=args.full_matrix)
    runner.run_all()


if __name__ == "__main__":
    main()
