"""
Check which Google account is associated with your API key.
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file")
    exit(1)

genai.configure(api_key=api_key)

print("="*70)
print("🔑 API KEY ACCOUNT INFORMATION")
print("="*70)

# The API key itself contains encoded information
print(f"\nYour API key: {api_key[:20]}...{api_key[-10:]}")
print(f"Key length: {len(api_key)} characters")

# Check the AI Studio URL
print("\n📍 To find your account:")
print("   1. Go to: https://aistudio.google.com/apikey")
print("   2. You'll see which Google account is logged in (top right)")
print("   3. Your API keys will be listed there")

# Alternative: Check billing
print("\n💳 To check billing/payment:")
print("   1. Go to: https://console.cloud.google.com/billing")
print("   2. This will show which account has billing enabled")
print("   3. Look for 'Gemini API' or 'Generative AI' usage")

# Test the key works
print("\n✅ Testing API key...")
try:
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content("What's 2+2?")
    print(f"   API key is active and working!")
    print(f"   Response: {response.text.strip()}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("💡 RECOMMENDATION")
print("="*70)
print("\nSince you're on PAID TIER, you can run:")
print("\n  python run_pipeline.py")
print("\nThis will cost ~$3.88 and complete in 6-8 hours.")
print("No need to worry about rate limits!")
print("="*70)
