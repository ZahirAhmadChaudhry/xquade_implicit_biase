"""
Script to detect which Gemini API tier you're using (Free vs Paid).
Tests rate limits and billing status.
"""

import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
from datetime import datetime

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file")
    exit(1)

genai.configure(api_key=api_key)

print("="*70)
print("🔍 GEMINI API TIER DETECTION")
print("="*70)
print(f"Testing at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Test 1: Check API key validity
print("Test 1: API Key Validation")
print("-"*70)
try:
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content("Say 'Hello'")
    print("✅ API key is valid")
    print(f"   Response: {response.text.strip()}")
except Exception as e:
    print(f"❌ API key error: {e}")
    exit(1)

# Test 2: Rapid fire test to detect rate limits
print("\n\nTest 2: Rate Limit Detection")
print("-"*70)
print("Sending 20 rapid requests to test rate limits...")
print("(Free tier: 15 RPM, Paid tier: 1000+ RPM)\n")

successes = 0
rate_limit_hits = 0
errors = []

start_time = time.time()

for i in range(20):
    try:
        response = model.generate_content(f"Count: {i}")
        successes += 1
        print(f"  Request {i+1}/20: ✅ Success", end='\r')
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            rate_limit_hits += 1
            print(f"  Request {i+1}/20: ⚠️  Rate limit hit")
            errors.append(error_msg)
            break
        else:
            print(f"  Request {i+1}/20: ❌ Error: {e}")
            errors.append(error_msg)
    
    # Small delay to avoid overwhelming
    time.sleep(0.1)

elapsed = time.time() - start_time
print(f"\n\nCompleted {successes + rate_limit_hits} requests in {elapsed:.2f} seconds")
print(f"  Successes: {successes}")
print(f"  Rate limits hit: {rate_limit_hits}")

# Test 3: Check model access
print("\n\nTest 3: Model Access Check")
print("-"*70)

try:
    # Try to list models (paid tier has more access)
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
    
    print(f"✅ Found {len(available_models)} available models")
    
    # Check for paid-tier exclusive features
    has_context_caching = False
    for m in genai.list_models():
        if hasattr(m, 'supported_caching_methods') and m.supported_caching_methods:
            has_context_caching = True
            break
    
    if has_context_caching:
        print("✅ Context caching available (Paid tier feature)")
    else:
        print("⚠️  Context caching not detected")
        
except Exception as e:
    print(f"⚠️  Could not check models: {e}")

# Analysis
print("\n\n" + "="*70)
print("📊 ANALYSIS")
print("="*70)

if rate_limit_hits > 0:
    print("\n🆓 LIKELY FREE TIER")
    print("\nEvidence:")
    print(f"  • Hit rate limit after {successes} requests")
    print(f"  • Free tier limit: 15 requests per minute")
    print(f"\nError message:")
    if errors:
        print(f"  {errors[0][:200]}")
    
    print("\n💡 Recommendations:")
    print("  • Use smaller sample sizes (50-100)")
    print("  • Split experiments across multiple days")
    print("  • Or upgrade to paid tier for ~$4 total cost")
    
elif successes >= 20:
    print("\n💳 LIKELY PAID TIER")
    print("\nEvidence:")
    print(f"  • Completed {successes} rapid requests without rate limits")
    print(f"  • Paid tier supports 1000+ requests per minute")
    
    print("\n💡 You can run full experiments:")
    print("  • Cost: ~$3.88 for 240 samples × 3 seeds")
    print("  • Time: 6-8 hours for complete run")
    print("  • Command: python run_pipeline.py")

else:
    print("\n❓ UNCLEAR")
    print(f"\nResults:")
    print(f"  • Successes: {successes}")
    print(f"  • Rate limits: {rate_limit_hits}")
    print("\n💡 Try running the test again or check billing in Google AI Studio")

# Show pricing info
print("\n\n" + "="*70)
print("💰 GEMINI 2.0 FLASH PRICING")
print("="*70)

print("\nFREE TIER:")
print("  • Input tokens:  FREE")
print("  • Output tokens: FREE")
print("  • Rate limits:   15 RPM, 1,500 RPD")
print("  • Your data may be used to improve products")

print("\nPAID TIER:")
print("  • Input tokens:  $0.10 / 1M tokens")
print("  • Output tokens: $0.40 / 1M tokens")
print("  • Rate limits:   1,000-2,000 RPM")
print("  • Your data is NOT used to improve products")
print(f"\n  • Estimated cost for full experiment: $3.88")

print("\n\nTo upgrade: https://aistudio.google.com/apikey")
print("="*70)
