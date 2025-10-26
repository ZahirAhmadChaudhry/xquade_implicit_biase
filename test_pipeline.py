"""
Quick test to validate the pipeline setup before running full experiments.
Tests all components with minimal samples.
"""

import sys
import subprocess
from pathlib import Path


def test_imports():
    """Test that all required packages are installed."""
    print("\n" + "="*60)
    print("🔍 Testing imports...")
    print("="*60)
    
    required_modules = [
        "google.generativeai",
        "sentence_transformers",
        "bert_score",
        "scipy",
        "numpy",
        "matplotlib",
        "seaborn",
        "rouge_score",
    ]
    
    failed = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed.append(module)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All imports successful")
    return True


def test_api_key():
    """Test that API key is configured."""
    print("\n" + "="*60)
    print("🔑 Testing API key...")
    print("="*60)
    
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            print("❌ GEMINI_API_KEY not found in .env file")
            print("\nCreate a .env file with:")
            print("GEMINI_API_KEY=your_key_here")
            return False
        
        print(f"✅ API key found: {api_key[:10]}...")
        return True
    
    except Exception as e:
        print(f"❌ Error loading API key: {e}")
        return False


def test_configuration():
    """Test experiment configuration."""
    print("\n" + "="*60)
    print("⚙️  Testing configuration...")
    print("="*60)
    
    try:
        from experiment_config import (
            get_core_experiment_configs,
            get_all_experiment_configs,
            ExperimentSettings
        )
        
        core_configs = get_core_experiment_configs()
        all_configs = get_all_experiment_configs()
        
        print(f"✅ Core configs: {len(core_configs)}")
        print(f"✅ Full matrix configs: {len(all_configs)}")
        
        # Test settings
        settings = ExperimentSettings()
        print(f"✅ Default settings: {settings.sample_size} samples, {settings.num_seeds} seeds")
        
        return True
    
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_prompt_builder():
    """Test prompt builder."""
    print("\n" + "="*60)
    print("📝 Testing prompt builder...")
    print("="*60)
    
    try:
        from prompt_builder import PromptBuilder, FewShotExampleSelector
        from experiment_config import PromptConfig, TagFormat, TagPlacement
        
        # Create a test config
        config = PromptConfig(
            name="test_config",
            tag_format=TagFormat.BRACKET,
            tag_placement=TagPlacement.PREFIX,
            use_few_shot=False,
        )
        
        builder = PromptBuilder(config)
        
        # Build a test prompt
        prompt = builder.build_prompt(
            context="This is a test context.",
            question="What is this?",
            language="en"
        )
        
        print(f"✅ Prompt builder working")
        print(f"   Sample prompt: {prompt[:100]}...")
        
        return True
    
    except Exception as e:
        print(f"❌ Prompt builder error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_experiment():
    """Run a very small experiment (5 samples, 1 seed, 1 language)."""
    print("\n" + "="*60)
    print("🧪 Running quick experiment (5 samples)...")
    print("="*60)
    
    try:
        cmd = [
            sys.executable,
            "run_comprehensive_experiments.py",
            "--sample-size", "5",
            "--num-seeds", "1",
            "--languages", "en",
            "--results-dir", "test_results"
        ]
        
        result = subprocess.run(cmd, shell=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("\n✅ Quick experiment successful")
            
            # Check output
            results_file = Path("test_results") / "all_results.json"
            if results_file.exists():
                print(f"✅ Results file created: {results_file}")
            else:
                print(f"⚠️  Results file not found: {results_file}")
            
            return True
        else:
            print(f"\n❌ Experiment failed with code {result.returncode}")
            return False
    
    except subprocess.TimeoutExpired:
        print("❌ Experiment timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Experiment error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 PIPELINE VALIDATION TEST")
    print("="*60)
    print("\nThis will validate your setup before running full experiments.")
    print("Expected time: 2-5 minutes")
    
    tests = [
        ("Imports", test_imports),
        ("API Key", test_api_key),
        ("Configuration", test_configuration),
        ("Prompt Builder", test_prompt_builder),
        ("Quick Experiment", test_quick_experiment),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<20}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - READY TO RUN EXPERIMENTS!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Quick test: python run_pipeline.py --sample-size 50 --num-seeds 1")
        print("  2. Full run:   python run_pipeline.py")
        print("\nSee QUICK_START.md for more options.")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ SOME TESTS FAILED - FIX ISSUES BEFORE RUNNING")
        print("="*60)
        print("\nCommon fixes:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - API key: Create .env file with GEMINI_API_KEY")
        print("  - Check error messages above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
