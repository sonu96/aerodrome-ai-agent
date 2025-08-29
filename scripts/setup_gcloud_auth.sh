#!/bin/bash

# Setup script for Google Cloud authentication for Gemini AI
# This script helps configure authentication for the Aerodrome Brain

echo "==========================================="
echo "Google Cloud Authentication Setup for Gemini"
echo "==========================================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed."
    echo ""
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    echo ""
    exit 1
fi

echo "✅ gcloud CLI is installed"
echo ""

# Check current authentication status
echo "Current authentication status:"
gcloud auth list
echo ""

# Menu for authentication options
echo "Choose authentication method:"
echo "1) Use Application Default Credentials (recommended for local development)"
echo "2) Use Service Account key file (for production/CI)"
echo "3) Check current project and credentials"
echo "4) Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Setting up Application Default Credentials..."
        echo "This will open a browser for authentication."
        echo ""
        gcloud auth application-default login
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Application Default Credentials configured successfully!"
            echo ""
            
            # Set project if not set
            current_project=$(gcloud config get-value project 2>/dev/null)
            if [ -z "$current_project" ]; then
                echo "No project is currently set."
                read -p "Enter your Google Cloud project ID: " project_id
                gcloud config set project "$project_id"
                echo "✅ Project set to: $project_id"
            else
                echo "Current project: $current_project"
                read -p "Do you want to change it? (y/n): " change_project
                if [ "$change_project" = "y" ] || [ "$change_project" = "Y" ]; then
                    read -p "Enter new project ID: " project_id
                    gcloud config set project "$project_id"
                    echo "✅ Project updated to: $project_id"
                fi
            fi
            
            # Update .env file
            echo ""
            read -p "Do you want to update .env file with project ID? (y/n): " update_env
            if [ "$update_env" = "y" ] || [ "$update_env" = "Y" ]; then
                project_id=$(gcloud config get-value project)
                if [ -f ".env" ]; then
                    # Update existing .env
                    sed -i.bak "s/GOOGLE_CLOUD_PROJECT=.*/GOOGLE_CLOUD_PROJECT=\"$project_id\"/" .env
                    echo "✅ Updated .env file"
                else
                    # Create from template
                    cp .env.example .env
                    sed -i.bak "s/GOOGLE_CLOUD_PROJECT=.*/GOOGLE_CLOUD_PROJECT=\"$project_id\"/" .env
                    echo "✅ Created .env file from template"
                fi
                # Remove Gemini API key since we're using ADC
                sed -i.bak "s/GEMINI_API_KEY=.*/GEMINI_API_KEY=\"\"/" .env
                echo "✅ Configured to use Application Default Credentials"
            fi
        else
            echo "❌ Failed to configure Application Default Credentials"
            exit 1
        fi
        ;;
        
    2)
        echo ""
        echo "Setting up Service Account authentication..."
        read -p "Enter path to service account key JSON file: " key_path
        
        if [ -f "$key_path" ]; then
            export GOOGLE_APPLICATION_CREDENTIALS="$key_path"
            echo ""
            echo "✅ Service account key configured!"
            echo ""
            echo "Add this to your .env file or shell profile:"
            echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$key_path\""
            echo ""
            
            # Extract project ID from service account
            project_id=$(python3 -c "import json; print(json.load(open('$key_path'))['project_id'])" 2>/dev/null)
            if [ ! -z "$project_id" ]; then
                echo "Project ID from service account: $project_id"
                gcloud config set project "$project_id"
                
                # Update .env file
                read -p "Do you want to update .env file? (y/n): " update_env
                if [ "$update_env" = "y" ] || [ "$update_env" = "Y" ]; then
                    if [ -f ".env" ]; then
                        sed -i.bak "s|GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=\"$key_path\"|" .env
                        sed -i.bak "s/GOOGLE_CLOUD_PROJECT=.*/GOOGLE_CLOUD_PROJECT=\"$project_id\"/" .env
                    else
                        cp .env.example .env
                        sed -i.bak "s|GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=\"$key_path\"|" .env
                        sed -i.bak "s/GOOGLE_CLOUD_PROJECT=.*/GOOGLE_CLOUD_PROJECT=\"$project_id\"/" .env
                    fi
                    # Remove Gemini API key since we're using service account
                    sed -i.bak "s/GEMINI_API_KEY=.*/GEMINI_API_KEY=\"\"/" .env
                    echo "✅ Updated .env file"
                fi
            fi
        else
            echo "❌ File not found: $key_path"
            exit 1
        fi
        ;;
        
    3)
        echo ""
        echo "Current Configuration:"
        echo "====================="
        echo ""
        echo "Active account:"
        gcloud auth list --filter=status:ACTIVE --format="value(account)"
        echo ""
        echo "Current project:"
        gcloud config get-value project
        echo ""
        echo "Application Default Credentials:"
        if [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
            echo "Using service account: $GOOGLE_APPLICATION_CREDENTIALS"
        else
            adc_path="$HOME/.config/gcloud/application_default_credentials.json"
            if [ -f "$adc_path" ]; then
                echo "Using ADC from: $adc_path"
                account=$(python3 -c "import json; data=json.load(open('$adc_path')); print(data.get('client_email', 'User account'))" 2>/dev/null)
                echo "Account: $account"
            else
                echo "Not configured"
            fi
        fi
        echo ""
        
        # Test Vertex AI access
        echo "Testing Vertex AI access..."
        python3 -c "
import os
os.environ['GOOGLE_CLOUD_PROJECT'] = '$(gcloud config get-value project 2>/dev/null)'
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    vertexai.init(location='us-central1')
    print('✅ Vertex AI authentication successful!')
except Exception as e:
    print(f'❌ Vertex AI authentication failed: {e}')
" 2>/dev/null || echo "❌ Python packages not installed or authentication failed"
        ;;
        
    4)
        echo "Exiting..."
        exit 0
        ;;
        
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "==========================================="
echo "Setup complete!"
echo ""
echo "To use Gemini with the Aerodrome Brain:"
echo "1. Make sure your .env file has GOOGLE_CLOUD_PROJECT set"
echo "2. Leave GEMINI_API_KEY empty to use Google Cloud auth"
echo "3. Run: python -m src.brain.core"
echo ""
echo "For more info: https://cloud.google.com/vertex-ai/docs/generative-ai/start/quickstarts/api-quickstart"
echo "==========================================="