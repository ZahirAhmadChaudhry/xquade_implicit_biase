"""
Experiment configuration for comprehensive multilingual LLM evaluation.
Defines all experimental conditions, tag formats, and prompting strategies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class TagPlacement(Enum):
    """Where to place the language tag in the prompt."""
    PREFIX = "prefix"  # [lang=es] Context: ...
    SUFFIX = "suffix"  # Context: ... [lang=es] Answer:
    BOTH = "both"      # [lang=es] Context: ... Answer in Spanish:
    NONE = "none"      # No tag (baseline)


class TagFormat(Enum):
    """Different language tag formatting styles."""
    BRACKET = "bracket"           # [lang=es]
    NATURAL = "natural"           # Language: Spanish
    XML = "xml"                   # <SPANISH>
    INSTRUCTION = "instruction"   # Answer in Spanish:
    ISO = "iso"                   # ISO 639-1: es


@dataclass
class PromptConfig:
    """Configuration for a single prompting strategy."""
    name: str                          # Unique identifier for this config
    tag_format: TagFormat
    tag_placement: TagPlacement
    use_few_shot: bool = False
    num_shots: int = 0                 # 0, 1, 2, 3
    use_multilingual_context: bool = False
    use_incorrect_tag: bool = False    # For ablation: use wrong language tag
    
    def get_description(self) -> str:
        """Human-readable description of this configuration."""
        parts = []
        if self.use_incorrect_tag:
            parts.append("Wrong-Tag")
        elif self.tag_placement == TagPlacement.NONE:
            parts.append("No-Tag")
        else:
            parts.append(f"{self.tag_format.value}-{self.tag_placement.value}")
        
        if self.use_few_shot:
            parts.append(f"{self.num_shots}-shot")
        else:
            parts.append("0-shot")
            
        if self.use_multilingual_context:
            parts.append("multilingual")
            
        return "_".join(parts)


def format_language_tag(lang: str, tag_format: TagFormat) -> str:
    """Generate language tag string based on format."""
    lang_names = {
        "en": "English",
        "es": "Spanish",
        "th": "Thai",
        "ar": "Arabic",
        "de": "German",
        "ru": "Russian",
        "zh": "Chinese",
        "hi": "Hindi",
        "tr": "Turkish",
        "vi": "Vietnamese",
        "el": "Greek",
    }
    
    if tag_format == TagFormat.BRACKET:
        return f"[lang={lang}]"
    elif tag_format == TagFormat.NATURAL:
        return f"Language: {lang_names.get(lang, lang.upper())}"
    elif tag_format == TagFormat.XML:
        return f"<{lang_names.get(lang, lang).upper()}>"
    elif tag_format == TagFormat.INSTRUCTION:
        return f"Answer in {lang_names.get(lang, lang)}:"
    elif tag_format == TagFormat.ISO:
        return f"ISO 639-1: {lang}"
    else:
        return f"[lang={lang}]"


def get_incorrect_language(correct_lang: str) -> str:
    """Return a deliberately incorrect language code for ablation."""
    # Map each language to a different one for wrong-tag experiments
    incorrect_map = {
        "en": "fr",
        "es": "de",
        "th": "ja",
        "ar": "he",
        "de": "en",
        "ru": "uk",
        "zh": "ko",
        "hi": "ur",
        "tr": "az",
        "vi": "th",
        "el": "ru",
        "fr": "es",
    }
    return incorrect_map.get(correct_lang, "en")


def get_all_experiment_configs() -> List[PromptConfig]:
    """Generate complete experiment matrix."""
    configs = []
    
    # 1. Baseline (no tag)
    configs.append(PromptConfig(
        name="baseline",
        tag_format=TagFormat.BRACKET,
        tag_placement=TagPlacement.NONE,
    ))
    
    # 2. Tag format variations (all prefix, zero-shot)
    for tag_format in [TagFormat.BRACKET, TagFormat.NATURAL, TagFormat.XML, 
                       TagFormat.INSTRUCTION, TagFormat.ISO]:
        configs.append(PromptConfig(
            name=f"tag_format_{tag_format.value}",
            tag_format=tag_format,
            tag_placement=TagPlacement.PREFIX,
        ))
    
    # 3. Tag placement variations (bracket format only)
    for placement in [TagPlacement.PREFIX, TagPlacement.SUFFIX, TagPlacement.BOTH]:
        configs.append(PromptConfig(
            name=f"tag_placement_{placement.value}",
            tag_format=TagFormat.BRACKET,
            tag_placement=placement,
        ))
    
    # 4. Few-shot experiments (1, 2, 3 shots with bracket prefix)
    for num_shots in [1, 2, 3]:
        configs.append(PromptConfig(
            name=f"few_shot_{num_shots}",
            tag_format=TagFormat.BRACKET,
            tag_placement=TagPlacement.PREFIX,
            use_few_shot=True,
            num_shots=num_shots,
        ))
    
    # 5. Incorrect language tag (ablation)
    configs.append(PromptConfig(
        name="incorrect_tag",
        tag_format=TagFormat.BRACKET,
        tag_placement=TagPlacement.PREFIX,
        use_incorrect_tag=True,
    ))
    
    # 6. Multilingual context
    configs.append(PromptConfig(
        name="multilingual_context",
        tag_format=TagFormat.BRACKET,
        tag_placement=TagPlacement.PREFIX,
        use_multilingual_context=True,
    ))
    
    return configs


def get_core_experiment_configs() -> List[PromptConfig]:
    """Get essential configurations for quick experimentation."""
    return [
        # Baseline
        PromptConfig(
            name="baseline",
            tag_format=TagFormat.BRACKET,
            tag_placement=TagPlacement.NONE,
        ),
        # Best performing from initial experiments
        PromptConfig(
            name="bracket_prefix",
            tag_format=TagFormat.BRACKET,
            tag_placement=TagPlacement.PREFIX,
        ),
        # Few-shot
        PromptConfig(
            name="few_shot_2",
            tag_format=TagFormat.BRACKET,
            tag_placement=TagPlacement.PREFIX,
            use_few_shot=True,
            num_shots=2,
        ),
        # Wrong tag ablation
        PromptConfig(
            name="incorrect_tag",
            tag_format=TagFormat.BRACKET,
            tag_placement=TagPlacement.PREFIX,
            use_incorrect_tag=True,
        ),
    ]


@dataclass
class ExperimentSettings:
    """Global experiment settings."""
    languages: List[str] = field(default_factory=lambda: ["en", "es", "th", "ar"])
    sample_size: int = 240  # Full XQuAD coverage
    num_seeds: int = 3
    base_seed: int = 42
    seed_step: int = 17
    save_samples: bool = True
    results_dir: str = "comprehensive_results"
    
    def get_seeds(self) -> List[int]:
        """Generate list of random seeds."""
        return [self.base_seed + i * self.seed_step for i in range(self.num_seeds)]
