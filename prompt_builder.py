"""
Enhanced prompt builder with support for few-shot, tag variations, and multilingual contexts.
"""

from typing import List, Dict, Optional
import random
from experiment_config import (
    PromptConfig, TagPlacement, TagFormat, 
    format_language_tag, get_incorrect_language
)


class PromptBuilder:
    """Builds prompts according to experimental configuration."""
    
    def __init__(self, config: PromptConfig):
        self.config = config
        
    def build_prompt(
        self,
        context: str,
        question: str,
        language: str,
        few_shot_examples: Optional[List[Dict]] = None,
        english_context: Optional[str] = None,
    ) -> str:
        """
        Build complete prompt according to configuration.
        
        Args:
            context: The context paragraph (in target language)
            question: The question (in target language)
            language: ISO language code
            few_shot_examples: List of example QA pairs for few-shot
            english_context: English translation for multilingual context
        """
        prompt_parts = []
        
        # Determine which language tag to use
        tag_lang = language
        if self.config.use_incorrect_tag:
            tag_lang = get_incorrect_language(language)
        
        # Add prefix tag if needed
        if self.config.tag_placement in [TagPlacement.PREFIX, TagPlacement.BOTH]:
            tag = format_language_tag(tag_lang, self.config.tag_format)
            prompt_parts.append(tag)
        
        # Add few-shot examples
        if self.config.use_few_shot and few_shot_examples:
            for i, example in enumerate(few_shot_examples[:self.config.num_shots], 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(f"Context: {example['context']}")
                prompt_parts.append(f"Question: {example['question']}")
                prompt_parts.append(f"Answer: {example['answer']}")
                prompt_parts.append("")  # Blank line
        
        # Build main prompt
        if self.config.use_multilingual_context and english_context:
            # Parallel context format
            prompt_parts.append(f"Context (English): {english_context}")
            prompt_parts.append(f"Context ({language.upper()}): {context}")
        else:
            prompt_parts.append(f"Context: {context}")
        
        prompt_parts.append(f"Question: {question}")
        
        # Add suffix tag if needed
        if self.config.tag_placement == TagPlacement.SUFFIX:
            tag = format_language_tag(tag_lang, self.config.tag_format)
            prompt_parts.append(tag)
        
        # Add instruction for BOTH placement
        if self.config.tag_placement == TagPlacement.BOTH:
            lang_name = {
                "en": "English", "es": "Spanish", "th": "Thai", "ar": "Arabic",
                "de": "German", "ru": "Russian", "zh": "Chinese", "hi": "Hindi",
            }.get(tag_lang, tag_lang.upper())
            prompt_parts.append(f"Answer in {lang_name}:")
        else:
            prompt_parts.append("Answer:")
        
        return "\n\n".join(prompt_parts)
    
    def get_baseline_prompt(self, context: str, question: str) -> str:
        """Generate baseline prompt without any tags."""
        return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"


class FewShotExampleSelector:
    """Selects few-shot examples from XQuAD data."""
    
    def __init__(self, xquad_data: List[Dict], seed: Optional[int] = None):
        """
        Args:
            xquad_data: Full list of XQuAD samples for a language
            seed: Random seed for reproducible selection
        """
        self.data = xquad_data
        self.rng = random.Random(seed)
    
    def select_examples(
        self,
        num_examples: int,
        exclude_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Select random examples for few-shot prompting.
        
        Args:
            num_examples: How many examples to select
            exclude_ids: Sample IDs to exclude (e.g., current test sample)
        
        Returns:
            List of example dictionaries with context, question, answer
        """
        exclude_set = set(exclude_ids or [])
        
        # Filter available examples
        available = [
            sample for sample in self.data 
            if sample['id'] not in exclude_set and sample.get('answers')
        ]
        
        if len(available) < num_examples:
            # If not enough, use what we have
            num_examples = len(available)
        
        # Sample randomly
        selected = self.rng.sample(available, num_examples)
        
        # Format for prompting
        examples = []
        for sample in selected:
            # Use first answer as the example answer
            answer_text = sample['answers'][0]['text'] if sample['answers'] else ""
            examples.append({
                'context': sample['context'],
                'question': sample['question'],
                'answer': answer_text,
            })
        
        return examples


def create_multilingual_context(
    target_lang_context: str,
    english_context: str,
    format_style: str = "parallel"
) -> str:
    """
    Create a multilingual context representation.
    
    Args:
        target_lang_context: Context in target language
        english_context: Context in English
        format_style: How to combine ("parallel", "interleaved", "code_switch")
    """
    if format_style == "parallel":
        # Side-by-side format
        return f"Context (English): {english_context}\n\nContext (Target): {target_lang_context}"
    elif format_style == "interleaved":
        # Sentence-by-sentence interleaving (simplified)
        return f"{english_context}\n\nTranslation: {target_lang_context}"
    elif format_style == "code_switch":
        # Simulated code-switching (basic concatenation)
        return f"{english_context[:len(english_context)//2]} ... {target_lang_context[len(target_lang_context)//2:]}"
    else:
        return target_lang_context
