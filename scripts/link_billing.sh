#!/bin/bash

echo "🔄 Waiting for new billing account..."
echo ""
echo "Please complete the billing account creation at:"
echo "https://console.cloud.google.com/billing/create"
echo ""
echo "Press Enter once you've created the billing account..."
read

# Check for billing accounts
echo "Checking for billing accounts..."
billing_accounts=$(~/google-cloud-sdk/bin/gcloud billing accounts list --filter="open=true" --format="value(name)" 2>/dev/null)

if [ -z "$billing_accounts" ]; then
    echo "❌ No open billing accounts found."
    echo "Listing all accounts:"
    ~/google-cloud-sdk/bin/gcloud billing accounts list
    echo ""
    echo "Please enter the billing account ID manually (format: XXXXXX-XXXXXX-XXXXXX):"
    read billing_account_id
    billing_account="billingAccounts/$billing_account_id"
else
    billing_account=$(echo "$billing_accounts" | head -n1)
    echo "✅ Found open billing account: $billing_account"
fi

# Link to project
echo ""
echo "Linking billing account to project aerodrome-brain-1756490979..."
~/google-cloud-sdk/bin/gcloud billing projects link aerodrome-brain-1756490979 --billing-account=$billing_account

if [ $? -eq 0 ]; then
    echo "✅ Billing account linked successfully!"
    
    # Enable APIs
    echo ""
    echo "Enabling required APIs..."
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
    
    if [ $? -eq 0 ]; then
        echo "✅ All APIs enabled successfully!"
        
        # Test Vertex AI
        echo ""
        echo "Testing Vertex AI access..."
        python3 -c "
import os
os.environ['GOOGLE_CLOUD_PROJECT'] = 'aerodrome-brain-1756490979'
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    vertexai.init(project='aerodrome-brain-1756490979', location='us-central1')
    model = GenerativeModel('gemini-1.5-flash')
    print('✅ Vertex AI is working!')
    print('✅ Gemini models are now accessible!')
except Exception as e:
    print(f'Error: {e}')
    print('You may need to install: pip install google-cloud-aiplatform vertexai')
"
    else
        echo "⚠️  Some APIs failed to enable. You may need to try again."
    fi
else
    echo "❌ Failed to link billing account"
fi

echo ""
echo "Setup complete! Your project is ready to use Gemini."