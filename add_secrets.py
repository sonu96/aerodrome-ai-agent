#!/usr/bin/env python3
"""
Interactive script to add API keys to Google Cloud Secret Manager.
Run this to securely store all your API keys.
"""

import sys
import getpass
from src.config.secrets import SecretManager

def main():
    print("========================================")
    print("🔐 Add API Keys to Secret Manager")
    print("========================================")
    print()
    print("This will securely store your API keys in Google Cloud Secret Manager.")
    print("Leave any field blank to skip it.")
    print()
    
    sm = SecretManager(project_id='aerodrome-brain-1756490979')
    
    if not sm._available:
        print("❌ Secret Manager is not available. Please check your Google Cloud setup.")
        return
    
    secrets_to_add = []
    
    # Gemini API Key
    print("1. Gemini API Key")
    print("   Get from: https://aistudio.google.com/app/apikey")
    gemini_key = getpass.getpass("   Enter Gemini API Key (hidden): ").strip()
    if gemini_key:
        secrets_to_add.append(('gemini-api-key', gemini_key))
    
    # QuickNode URL
    print("\n2. QuickNode URL")
    print("   Get from: https://dashboard.quicknode.com")
    print("   Need: Aerodrome addon on Base network")
    quicknode_url = input("   Enter QuickNode URL: ").strip()
    if quicknode_url:
        secrets_to_add.append(('quicknode-url', quicknode_url))
    
    # Mem0 API Key
    print("\n3. Mem0 API Key")
    print("   Get from: https://app.mem0.ai")
    mem0_key = getpass.getpass("   Enter Mem0 API Key (hidden): ").strip()
    if mem0_key:
        secrets_to_add.append(('mem0-api-key', mem0_key))
    
    # Neo4j Configuration
    print("\n4. Neo4j Configuration (optional)")
    print("   For local: docker run -d -p 7474:7474 -p 7687:7687 neo4j")
    print("   Or use Neo4j Aura: https://neo4j.com/cloud/aura/")
    neo4j_uri = input("   Enter Neo4j URI (default: bolt://localhost:7687): ").strip()
    if not neo4j_uri:
        neo4j_uri = "bolt://localhost:7687"
    secrets_to_add.append(('neo4j-uri', neo4j_uri))
    
    neo4j_user = input("   Enter Neo4j username (default: neo4j): ").strip()
    if not neo4j_user:
        neo4j_user = "neo4j"
    secrets_to_add.append(('neo4j-username', neo4j_user))
    
    neo4j_pass = getpass.getpass("   Enter Neo4j password (hidden): ").strip()
    if neo4j_pass:
        secrets_to_add.append(('neo4j-password', neo4j_pass))
    
    # OpenAI API Key (optional)
    print("\n5. OpenAI API Key (optional - for embeddings)")
    print("   Get from: https://platform.openai.com")
    openai_key = getpass.getpass("   Enter OpenAI API Key (hidden): ").strip()
    if openai_key:
        secrets_to_add.append(('openai-api-key', openai_key))
    
    # CDP Keys (optional)
    print("\n6. Coinbase Developer Platform (optional - for Part 2)")
    print("   Get from: https://portal.cdp.coinbase.com")
    cdp_name = input("   Enter CDP API Key Name: ").strip()
    if cdp_name:
        secrets_to_add.append(('cdp-api-key-name', cdp_name))
        cdp_private = getpass.getpass("   Enter CDP Private Key (hidden): ").strip()
        if cdp_private:
            secrets_to_add.append(('cdp-api-key-private', cdp_private))
    
    # Confirm and add secrets
    if not secrets_to_add:
        print("\n❌ No secrets to add.")
        return
    
    print(f"\n📝 Ready to add {len(secrets_to_add)} secrets to Secret Manager")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled.")
        return
    
    print("\nAdding secrets...")
    for secret_id, secret_value in secrets_to_add:
        try:
            if sm.create_secret(secret_id, secret_value):
                print(f"✅ Added: {secret_id}")
            else:
                print(f"❌ Failed to add: {secret_id}")
        except Exception as e:
            print(f"❌ Error adding {secret_id}: {e}")
    
    print("\n✅ Done! Your secrets are now stored in Google Cloud Secret Manager.")
    print("\nTo verify, run:")
    print("  ~/google-cloud-sdk/bin/gcloud secrets list --project=aerodrome-brain-1756490979")
    print("\nThe Aerodrome Brain will automatically use these secrets.")

if __name__ == "__main__":
    main()