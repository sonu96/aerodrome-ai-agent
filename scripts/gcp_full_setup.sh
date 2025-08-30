#!/bin/bash

# Complete Google Cloud Setup Script for Aerodrome Brain
# This script handles authentication, project creation, cleanup, and billing setup

set -e

echo "=========================================="
echo "🚀 Google Cloud Complete Setup"
echo "=========================================="
echo ""

GCLOUD_PATH="$HOME/google-cloud-sdk/bin/gcloud"

# Function to check if gcloud is installed
check_gcloud() {
    if [ ! -f "$GCLOUD_PATH" ]; then
        echo "❌ gcloud not found at $GCLOUD_PATH"
        echo "Installing gcloud CLI..."
        curl https://sdk.cloud.google.com | bash
        exec -l $SHELL
    fi
    echo "✅ gcloud CLI found"
}

# Function to login
do_login() {
    echo ""
    echo "📝 Logging into Google Cloud..."
    $GCLOUD_PATH auth login --brief
    
    # Also set up Application Default Credentials
    echo ""
    echo "🔐 Setting up Application Default Credentials..."
    $GCLOUD_PATH auth application-default login
}

# Function to list and delete old projects
cleanup_projects() {
    echo ""
    echo "🗑️  Cleaning up existing projects..."
    echo ""
    
    # List current projects
    echo "Current projects:"
    $GCLOUD_PATH projects list --format="table(projectId,name,createTime)" 2>/dev/null || echo "No projects found"
    
    echo ""
    read -p "Do you want to delete any existing projects? (y/n): " delete_choice
    
    if [ "$delete_choice" = "y" ] || [ "$delete_choice" = "Y" ]; then
        # Get list of project IDs
        projects=$($GCLOUD_PATH projects list --format="value(projectId)" 2>/dev/null)
        
        if [ -z "$projects" ]; then
            echo "No projects to delete."
        else
            echo "Select projects to delete (space-separated project IDs, or 'all' for all projects):"
            echo "$projects"
            read -p "Enter project IDs to delete: " projects_to_delete
            
            if [ "$projects_to_delete" = "all" ]; then
                projects_to_delete=$projects
            fi
            
            for project in $projects_to_delete; do
                echo "Deleting project: $project"
                $GCLOUD_PATH projects delete $project --quiet || echo "Failed to delete $project"
            done
        fi
    fi
}

# Function to create new project
create_project() {
    echo ""
    echo "✨ Creating new project for Aerodrome Brain..."
    echo ""
    
    # Generate a unique project ID
    timestamp=$(date +%Y%m%d%H%M%S)
    default_project_id="aerodrome-brain-${timestamp}"
    
    read -p "Enter project ID (default: $default_project_id): " project_id
    project_id=${project_id:-$default_project_id}
    
    read -p "Enter project name (default: Aerodrome Brain): " project_name
    project_name=${project_name:-"Aerodrome Brain"}
    
    # Create the project
    echo "Creating project: $project_id"
    $GCLOUD_PATH projects create $project_id --name="$project_name" --set-as-default
    
    if [ $? -eq 0 ]; then
        echo "✅ Project created successfully!"
        
        # Set as default project
        $GCLOUD_PATH config set project $project_id
        echo "✅ Set as default project"
        
        # Store for later use
        echo "$project_id" > .gcp_project_id
    else
        echo "❌ Failed to create project"
        return 1
    fi
}

# Function to set up billing
setup_billing() {
    echo ""
    echo "💳 Setting up billing..."
    echo ""
    
    # List billing accounts
    echo "Available billing accounts:"
    billing_accounts=$($GCLOUD_PATH billing accounts list --format="table(name,displayName,open)" 2>/dev/null)
    
    if [ -z "$billing_accounts" ]; then
        echo "❌ No billing accounts found."
        echo ""
        echo "Please create a billing account at:"
        echo "https://console.cloud.google.com/billing/create"
        echo ""
        read -p "Press Enter after creating a billing account to continue..."
        billing_accounts=$($GCLOUD_PATH billing accounts list --format="table(name,displayName,open)" 2>/dev/null)
    fi
    
    echo "$billing_accounts"
    
    # Get billing account ID
    billing_account=$($GCLOUD_PATH billing accounts list --filter="open=true" --format="value(name)" --limit=1 2>/dev/null)
    
    if [ -z "$billing_account" ]; then
        read -p "Enter billing account ID (format: billingAccounts/XXXXXX-XXXXXX-XXXXXX): " billing_account
    else
        echo "Found billing account: $billing_account"
        read -p "Use this billing account? (y/n): " use_billing
        if [ "$use_billing" != "y" ] && [ "$use_billing" != "Y" ]; then
            read -p "Enter billing account ID: " billing_account
        fi
    fi
    
    # Link billing account to project
    if [ -f .gcp_project_id ]; then
        project_id=$(cat .gcp_project_id)
    else
        project_id=$($GCLOUD_PATH config get-value project)
    fi
    
    echo "Linking billing account to project $project_id..."
    $GCLOUD_PATH billing projects link $project_id --billing-account=$billing_account
    
    if [ $? -eq 0 ]; then
        echo "✅ Billing account linked successfully!"
    else
        echo "❌ Failed to link billing account"
    fi
}

# Function to enable required APIs
enable_apis() {
    echo ""
    echo "🔧 Enabling required APIs..."
    echo ""
    
    if [ -f .gcp_project_id ]; then
        project_id=$(cat .gcp_project_id)
    else
        project_id=$($GCLOUD_PATH config get-value project)
    fi
    
    # List of required APIs
    apis=(
        "aiplatform.googleapis.com"           # Vertex AI
        "compute.googleapis.com"               # Compute Engine
        "storage.googleapis.com"               # Cloud Storage
        "cloudbuild.googleapis.com"           # Cloud Build
        "run.googleapis.com"                  # Cloud Run
        "cloudresourcemanager.googleapis.com" # Resource Manager
        "iam.googleapis.com"                  # IAM
        "logging.googleapis.com"              # Cloud Logging
        "monitoring.googleapis.com"           # Cloud Monitoring
    )
    
    for api in "${apis[@]}"; do
        echo "Enabling $api..."
        $GCLOUD_PATH services enable $api --project=$project_id || echo "Failed to enable $api"
    done
    
    echo "✅ APIs enabled"
}

# Function to test Vertex AI
test_vertex_ai() {
    echo ""
    echo "🧪 Testing Vertex AI access..."
    echo ""
    
    python3 -c "
import os
import sys

# Set project from file or config
try:
    with open('.gcp_project_id', 'r') as f:
        project_id = f.read().strip()
        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
except:
    pass

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    
    project = os.environ.get('GOOGLE_CLOUD_PROJECT')
    if not project:
        print('❌ No project set')
        sys.exit(1)
        
    print(f'Initializing Vertex AI with project: {project}')
    vertexai.init(project=project, location='us-central1')
    
    # Try to create a model instance
    model = GenerativeModel('gemini-1.5-flash')
    print('✅ Vertex AI authentication successful!')
    print(f'✅ Ready to use Gemini models in project: {project}')
    
except Exception as e:
    print(f'❌ Vertex AI test failed: {e}')
    print('You may need to:')
    print('1. Enable the Vertex AI API')
    print('2. Install required Python packages: pip install google-cloud-aiplatform vertexai')
" 2>&1
}

# Function to update .env file
update_env_file() {
    echo ""
    echo "📝 Updating .env file..."
    echo ""
    
    if [ -f .gcp_project_id ]; then
        project_id=$(cat .gcp_project_id)
    else
        project_id=$($GCLOUD_PATH config get-value project)
    fi
    
    if [ ! -f .env ]; then
        cp .env.example .env
        echo "Created .env from template"
    fi
    
    # Update project ID
    sed -i.bak "s/GOOGLE_CLOUD_PROJECT=.*/GOOGLE_CLOUD_PROJECT=\"$project_id\"/" .env
    
    # Clear Gemini API key since we're using ADC
    sed -i.bak "s/GEMINI_API_KEY=.*/GEMINI_API_KEY=\"\"/" .env
    
    echo "✅ Updated .env with project: $project_id"
    echo "✅ Configured to use Application Default Credentials"
}

# Main execution flow
main() {
    echo "This script will:"
    echo "1. Set up Google Cloud authentication"
    echo "2. Clean up old projects (optional)"
    echo "3. Create a new project"
    echo "4. Set up billing"
    echo "5. Enable required APIs"
    echo "6. Test Vertex AI access"
    echo "7. Update your .env file"
    echo ""
    read -p "Continue? (y/n): " continue_choice
    
    if [ "$continue_choice" != "y" ] && [ "$continue_choice" != "Y" ]; then
        echo "Cancelled."
        exit 0
    fi
    
    # Check gcloud installation
    check_gcloud
    
    # Login
    do_login
    
    # Cleanup old projects
    cleanup_projects
    
    # Create new project
    create_project
    
    # Set up billing
    setup_billing
    
    # Enable APIs
    enable_apis
    
    # Test Vertex AI
    test_vertex_ai
    
    # Update .env
    update_env_file
    
    # Final summary
    echo ""
    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    
    if [ -f .gcp_project_id ]; then
        project_id=$(cat .gcp_project_id)
    else
        project_id=$($GCLOUD_PATH config get-value project)
    fi
    
    echo "Project ID: $project_id"
    echo "Region: us-central1"
    echo "Authentication: Application Default Credentials"
    echo ""
    echo "Next steps:"
    echo "1. Ensure you have the required Python packages:"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "2. Add your other API keys to .env:"
    echo "   - QUICKNODE_URL"
    echo "   - MEM0_API_KEY"
    echo "   - NEO4J credentials"
    echo ""
    echo "3. Run the brain:"
    echo "   python -m src.brain.core"
    echo ""
    echo "=========================================="
    
    # Clean up temp file
    rm -f .gcp_project_id
}

# Run main function
main