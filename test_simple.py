import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ Error: Please set the GEMINI_API_KEY environment variable in your .env file or system.")
genai.configure(api_key=GEMINI_API_KEY)

def test_simple_generation():
    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        # Test with a simple prompt first
        simple_prompt = "What is 2+2?"
        
        # Using more permissive safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        
        response = model.generate_content(
            simple_prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=200
            )
        )
        
        if response and response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text
                print(f"✅ Simple test successful: {text}")
                return True
            elif hasattr(candidate, 'text'):
                text = candidate.text
                print(f"✅ Simple test successful: {text}")
                return True
        else:
            print("❌ No response generated")
            if response and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    print(f"   Finish reason: {candidate.finish_reason}")
                if hasattr(candidate, 'safety_ratings'):
                    print(f"   Safety ratings: {candidate.safety_ratings}")
            return False
            
    except Exception as e:
        print(f"❌ Error in simple test: {e}")
        return False

def test_xquad_prompt():
    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        # Test with an XQuAD-style prompt
        xquad_prompt = """Context: The Panthers defense gave up just 308 points, ranking sixth in the league, while also leading the NFL in interceptions with 24 and boasting four Pro Bowl selections.

Question: How many points did the Panthers defense surrender?

Answer:"""
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        
        response = model.generate_content(
            xquad_prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=200
            )
        )
        
        if response and response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text
                print(f"✅ XQuAD test successful: {text}")
                return True
            elif hasattr(candidate, 'text'):
                text = candidate.text
                print(f"✅ XQuAD test successful: {text}")
                return True
        else:
            print("❌ XQuAD test failed")
            if response and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    print(f"   Finish reason: {candidate.finish_reason}")
                if hasattr(candidate, 'safety_ratings'):
                    print(f"   Safety ratings: {candidate.safety_ratings}")
            return False
            
    except Exception as e:
        print(f"❌ Error in XQuAD test: {e}")
        return False

if __name__ == "__main__":
    print("Testing Gemini API with safety settings...")
    
    print("\n1. Testing simple prompt:")
    simple_success = test_simple_generation()
    
    print("\n2. Testing XQuAD-style prompt:")
    xquad_success = test_xquad_prompt()
    
    if simple_success and xquad_success:
        print("\n✅ Both tests passed! The API should work for the full experiment.")
    else:
        print("\n❌ Some tests failed. Check the API settings or try alternative approaches.")
