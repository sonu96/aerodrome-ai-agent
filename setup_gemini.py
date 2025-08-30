#!/usr/bin/env python3
"""
Setup and test Gemini API key with 2.0 model in Secret Manager
"""

import sys
import getpass
import os
os.environ['GOOGLE_CLOUD_PROJECT'] = 'aerodrome-brain-1756490979'

from src.config.secrets import SecretManager

def test_gemini_api(api_key: str) -> bool:
    """Test the Gemini API key with 2.0 model"""
    try:
        import google.generativeai as genai
        
        # Configure with the API key
        genai.configure(api_key=api_key)
        
        print("\n🧪 Testing Gemini API key...")
        
        # Test different models
        models_to_test = [
            ('gemini-2.0-flash-exp', 'Gemini 2.0 Flash (Experimental)'),
            ('gemini-1.5-flash', 'Gemini 1.5 Flash (Stable)'),
            ('gemini-1.5-pro', 'Gemini 1.5 Pro')
        ]
        
        working_models = []
        
        for model_name, description in models_to_test:
            try:
                print(f"\nTesting {description}...")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    "Say 'Hello Aerodrome Brain!' in exactly 5 words"
                )
                print(f"✅ {description} works!")
                print(f"   Response: {response.text.strip()}")
                working_models.append(model_name)
            except Exception as e:
                print(f"❌ {description} failed: {str(e)[:100]}")
        
        if working_models:
            print(f"\n✅ API key is valid! Working models: {', '.join(working_models)}")
            return True
        else:
            print("\n❌ API key didn't work with any models")
            return False
            
    except Exception as e:
        print(f"\n❌ Error testing API key: {e}")
        return False

def main():
    print("========================================")
    print("🚀 Gemini 2.0 API Key Setup")
    print("========================================")
    print()
    
    # Initialize Secret Manager
    sm = SecretManager(project_id='aerodrome-brain-1756490979')
    
    if not sm._available:
        print("❌ Secret Manager is not available. Please check your Google Cloud setup.")
        return
    
    # Check if API key already exists
    existing_key = sm.get_secret('gemini-api-key')
    if existing_key:
        print("📝 Found existing Gemini API key in Secret Manager")
        print()
        test_choice = input("Do you want to test it? (y/n): ").strip().lower()
        if test_choice == 'y':
            if test_gemini_api(existing_key):
                print("\n✅ Existing API key is working!")
                update_choice = input("\nDo you want to update it with a new key? (y/n): ").strip().lower()
                if update_choice != 'y':
                    print("\n✨ Setup complete! Using existing key.")
                    return
    
    # Get new API key
    print("\n📝 Adding new Gemini API key")
    print()
    print("Steps to get your API key:")
    print("1. Go to: https://aistudio.google.com/app/apikey")
    print("2. Click 'Create API Key' or 'Get API Key'")
    print("3. Select your Google Cloud project (aerodrome-brain-1756490979)")
    print("4. Copy the API key")
    print()
    
    api_key = getpass.getpass("Enter your Gemini API key (hidden): ").strip()
    
    if not api_key:
        print("\n❌ No API key provided")
        return
    
    # Test the API key
    if not test_gemini_api(api_key):
        print("\n⚠️  API key test failed, but continuing to save it anyway.")
        print("You may need to:")
        print("1. Enable the Generative Language API in your project")
        print("2. Wait a few minutes for the API key to activate")
    
    # Save to Secret Manager
    print("\n💾 Saving API key to Secret Manager...")
    
    try:
        # Try to update if exists, otherwise create
        if existing_key:
            success = sm.update_secret('gemini-api-key', api_key)
        else:
            success = sm.create_secret('gemini-api-key', api_key)
        
        if success:
            print("✅ API key saved to Secret Manager!")
            
            # Verify it was saved
            saved_key = sm.get_secret('gemini-api-key')
            if saved_key == api_key:
                print("✅ Verified: API key is correctly stored")
            else:
                print("⚠️  Warning: Retrieved key doesn't match")
        else:
            print("❌ Failed to save API key to Secret Manager")
            return
            
    except Exception as e:
        print(f"❌ Error saving to Secret Manager: {e}")
        return
    
    print("\n========================================")
    print("✅ Setup Complete!")
    print("========================================")
    print()
    print("Your Gemini API key is now securely stored in Secret Manager.")
    print("The Aerodrome Brain will automatically use it.")
    print()
    print("Configuration:")
    print(f"  Project: aerodrome-brain-1756490979")
    print(f"  Default Model: Gemini 2.0 Flash (Experimental)")
    print(f"  Fallback Model: Gemini 1.5 Flash")
    print()
    print("To test the brain with Gemini:")
    print("  python3 -c \"from src.intelligence.gemini_client import GeminiClient; ")
    print("             client = GeminiClient(); ")
    print("             print(client.generate('Hello!'))\"")
    print()
    print("To view your secret:")
    print("  ~/google-cloud-sdk/bin/gcloud secrets versions access latest \\")
    print("    --secret=gemini-api-key --project=aerodrome-brain-1756490979")

if __name__ == "__main__":
    main()