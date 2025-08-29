# Authentication Setup Guide

This guide explains how to set up authentication for the Aerodrome Protocol Intelligence Brain.

## Required Services

1. **QuickNode** - For Aerodrome protocol data
2. **Mem0** - For memory management
3. **Google Cloud (Gemini)** - For AI intelligence
4. **Neo4j** - For graph memory (optional)
5. **OpenAI** - For embeddings (optional)

## Google Cloud / Gemini Setup

The brain uses Google's Gemini 2.0 Flash model via Vertex AI. You have two authentication options:

### Option 1: Google Cloud CLI (Recommended)

This is the easiest method for local development:

```bash
# 1. Install gcloud CLI if not already installed
# Visit: https://cloud.google.com/sdk/docs/install

# 2. Run our setup script
./scripts/setup_gcloud_auth.sh

# 3. Follow the prompts to:
#    - Login with your Google account
#    - Select or create a project
#    - Configure Application Default Credentials
```

The script will automatically:
- Configure Application Default Credentials
- Set your Google Cloud project
- Update your .env file
- Test Vertex AI access

### Option 2: Service Account (Production)

For production deployments:

```bash
# 1. Create a service account in Google Cloud Console
# 2. Download the JSON key file
# 3. Set the environment variable:
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Or add to .env:
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### Option 3: API Key (Testing Only)

For quick testing (not recommended for production):

```bash
# Get API key from https://aistudio.google.com
# Add to .env:
GEMINI_API_KEY="your-api-key-here"
```

## Authentication Priority

The system checks for authentication in this order:
1. `GEMINI_API_KEY` (if set, uses API key)
2. `GOOGLE_APPLICATION_CREDENTIALS` (if set, uses service account)
3. Application Default Credentials (uses gcloud auth)

## Other Required Keys

### QuickNode Setup

1. Go to [QuickNode Dashboard](https://dashboard.quicknode.com)
2. Create an endpoint on Base network
3. Add the Aerodrome Swap API addon (ID: 1051)
4. Copy your endpoint URL to `.env`:
   ```
   QUICKNODE_URL="https://your-endpoint.quiknode.pro/your-token/"
   ```

### Mem0 Setup

1. Sign up at [Mem0 Platform](https://app.mem0.ai)
2. Get your API key from the dashboard
3. Add to `.env`:
   ```
   MEM0_API_KEY="your-mem0-api-key"
   ```

### Neo4j Setup (Optional)

For local development:
```bash
# Using Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest
```

Or use [Neo4j Aura](https://neo4j.com/cloud/aura/) for cloud hosting.

Add to `.env`:
```
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="password123"
```

### OpenAI Setup (Optional)

Only needed if you want to use OpenAI embeddings:

1. Get API key from [OpenAI Platform](https://platform.openai.com)
2. Add to `.env`:
   ```
   OPENAI_API_KEY="sk-..."
   ```

## Complete .env Example

```bash
# Copy the template
cp .env.example .env

# Edit with your values
nano .env
```

Minimal required configuration:
```env
# Google Cloud (using CLI auth)
GOOGLE_CLOUD_PROJECT="your-project-id"
GEMINI_API_KEY=""  # Leave empty for gcloud auth

# QuickNode
QUICKNODE_URL="https://your-endpoint.quiknode.pro/token/"

# Mem0
MEM0_API_KEY="your-mem0-key"

# Neo4j (local or cloud)
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your-password"
```

## Verification

Test your setup:

```bash
# Run the verification script
python scripts/verify_setup.py

# Or test individual components:
# Test Gemini
python -c "from src.intelligence.gemini_client import GeminiClient; client = GeminiClient(); print('✅ Gemini OK')"

# Test Mem0
python -c "from src.memory.mem0_client import EnhancedMem0Client; print('✅ Mem0 OK')"

# Test QuickNode
python -c "from src.protocol.aerodrome_client import AerodromeClient; print('✅ QuickNode OK')"
```

## Troubleshooting

### "Application Default Credentials not found"

Run:
```bash
gcloud auth application-default login
```

### "Project not set"

Run:
```bash
gcloud config set project YOUR_PROJECT_ID
```

### "Vertex AI API not enabled"

Enable the API:
```bash
gcloud services enable aiplatform.googleapis.com
```

### "Insufficient permissions"

Ensure your account/service account has these roles:
- `Vertex AI User`
- `Service Usage Consumer`

Grant roles:
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:your-email@gmail.com" \
  --role="roles/aiplatform.user"
```

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use Application Default Credentials** for local development
3. **Use Service Accounts** for production with minimal permissions
4. **Rotate keys regularly**
5. **Use secret management** services in production (e.g., Google Secret Manager)

## Next Steps

Once authentication is configured:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the brain:
   ```bash
   python -m src.brain.core
   ```

3. Or use the API:
   ```bash
   python -m src.api.main
   ```

For more details, see the [main README](../README.md).