# ✅ Google Cloud Setup Status

## Completed Steps ✅

1. **Google Cloud SDK Installed** ✅
   - Location: `~/google-cloud-sdk/`

2. **Authentication Configured** ✅
   - Application Default Credentials saved
   - Location: `~/.config/gcloud/application_default_credentials.json`

3. **Project Created** ✅
   - Project ID: `aerodrome-brain-1756490979`
   - Project Name: `Aerodrome Brain`
   - Set as default project

4. **.env File Created** ✅
   - Configured with project ID
   - Set to use Application Default Credentials (no API key needed)

## ⚠️ Required Action: Enable Billing

To complete the setup, you need to enable billing:

### Option 1: Through Console (Easiest)

1. **Open the billing setup page:**
   https://console.cloud.google.com/billing/linkedaccount?project=aerodrome-brain-1756490979

2. **Create or select a billing account**

3. **Link it to the project**

### Option 2: Through Command Line

```bash
# If you have an existing billing account:
~/google-cloud-sdk/bin/gcloud billing accounts list

# Link it to the project:
~/google-cloud-sdk/bin/gcloud billing projects link aerodrome-brain-1756490979 \
  --billing-account=YOUR_BILLING_ACCOUNT_ID
```

## After Billing is Enabled

Run this command to enable all required APIs:

```bash
~/google-cloud-sdk/bin/gcloud services enable \
  aiplatform.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  --project=aerodrome-brain-1756490979
```

## Test Vertex AI Access

After enabling billing and APIs, test with:

```bash
python3 -c "
import vertexai
from vertexai.generative_models import GenerativeModel
vertexai.init(project='aerodrome-brain-1756490979', location='us-central1')
model = GenerativeModel('gemini-1.5-flash')
response = model.generate_content('Hello, Gemini!')
print('✅ Vertex AI is working!')
print('Response:', response.text[:100])
"
```

## Other Required Keys

Add these to your `.env` file:

1. **QuickNode**: 
   - Get from: https://dashboard.quicknode.com
   - Add: `QUICKNODE_URL="your-endpoint"`

2. **Mem0**:
   - Get from: https://app.mem0.ai
   - Add: `MEM0_API_KEY="your-key"`

3. **Neo4j** (optional):
   - Local: `docker run -d -p 7474:7474 -p 7687:7687 neo4j`
   - Or use Neo4j Aura

## Quick Commands Reference

```bash
# Check current project
~/google-cloud-sdk/bin/gcloud config get-value project

# Check authentication
~/google-cloud-sdk/bin/gcloud auth list

# Check enabled APIs
~/google-cloud-sdk/bin/gcloud services list --enabled

# Test Application Default Credentials
~/google-cloud-sdk/bin/gcloud auth application-default print-access-token
```

## Delete Old Projects (Optional)

To clean up old projects:

```bash
# List projects
~/google-cloud-sdk/bin/gcloud projects list

# Delete a project
~/google-cloud-sdk/bin/gcloud projects delete PROJECT_ID
```

---

**Next Step**: Enable billing at https://console.cloud.google.com/billing/linkedaccount?project=aerodrome-brain-1756490979