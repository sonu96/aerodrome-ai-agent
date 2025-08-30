#!/bin/bash

# Secret Manager setup script for Aerodrome Brain
PROJECT_ID="aerodrome-brain-1756490979"
GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"

echo "=========================================="
echo "🔐 Google Cloud Secret Manager Setup"
echo "=========================================="
echo ""

# Function to create or update a secret
create_or_update_secret() {
    local secret_name=$1
    local secret_value=$2
    
    # Check if secret exists
    if $GCLOUD secrets describe $secret_name --project=$PROJECT_ID &>/dev/null; then
        echo "Updating secret: $secret_name"
        echo -n "$secret_value" | $GCLOUD secrets versions add $secret_name --data-file=- --project=$PROJECT_ID
    else
        echo "Creating secret: $secret_name"
        echo -n "$secret_value" | $GCLOUD secrets create $secret_name --data-file=- --replication-policy=automatic --project=$PROJECT_ID
    fi
}

# Function to read a secret
read_secret() {
    local secret_name=$1
    $GCLOUD secrets versions access latest --secret=$secret_name --project=$PROJECT_ID 2>/dev/null
}

# Function to list all secrets
list_secrets() {
    echo "Current secrets:"
    $GCLOUD secrets list --project=$PROJECT_ID --format="table(name,createTime)" 2>/dev/null
}

# Main menu
while true; do
    echo ""
    echo "Choose an option:"
    echo "1) Set up all secrets"
    echo "2) Update individual secret"
    echo "3) List all secrets"
    echo "4) Read a secret value"
    echo "5) Delete a secret"
    echo "6) Exit"
    echo ""
    read -p "Enter choice (1-6): " choice
    
    case $choice in
        1)
            echo ""
            echo "Setting up all secrets..."
            echo "Leave blank to skip any secret."
            echo ""
            
            # Gemini API Key
            read -p "Enter Gemini API Key (or press Enter to skip): " gemini_key
            if [ ! -z "$gemini_key" ]; then
                create_or_update_secret "gemini-api-key" "$gemini_key"
            fi
            
            # QuickNode URL
            read -p "Enter QuickNode URL: " quicknode_url
            if [ ! -z "$quicknode_url" ]; then
                create_or_update_secret "quicknode-url" "$quicknode_url"
            fi
            
            # Mem0 API Key
            read -p "Enter Mem0 API Key: " mem0_key
            if [ ! -z "$mem0_key" ]; then
                create_or_update_secret "mem0-api-key" "$mem0_key"
            fi
            
            # Neo4j credentials
            read -p "Enter Neo4j URI (default: bolt://localhost:7687): " neo4j_uri
            neo4j_uri=${neo4j_uri:-"bolt://localhost:7687"}
            create_or_update_secret "neo4j-uri" "$neo4j_uri"
            
            read -p "Enter Neo4j username (default: neo4j): " neo4j_user
            neo4j_user=${neo4j_user:-"neo4j"}
            create_or_update_secret "neo4j-username" "$neo4j_user"
            
            read -s -p "Enter Neo4j password: " neo4j_pass
            echo ""
            if [ ! -z "$neo4j_pass" ]; then
                create_or_update_secret "neo4j-password" "$neo4j_pass"
            fi
            
            # OpenAI API Key (optional)
            read -p "Enter OpenAI API Key (optional): " openai_key
            if [ ! -z "$openai_key" ]; then
                create_or_update_secret "openai-api-key" "$openai_key"
            fi
            
            # CDP Keys (optional for Part 2)
            read -p "Enter CDP API Key Name (optional): " cdp_key_name
            if [ ! -z "$cdp_key_name" ]; then
                create_or_update_secret "cdp-api-key-name" "$cdp_key_name"
                
                read -s -p "Enter CDP Private Key: " cdp_private_key
                echo ""
                if [ ! -z "$cdp_private_key" ]; then
                    create_or_update_secret "cdp-api-key-private" "$cdp_private_key"
                fi
            fi
            
            echo ""
            echo "✅ Secrets setup complete!"
            ;;
            
        2)
            echo ""
            echo "Available secret names:"
            echo "  gemini-api-key"
            echo "  quicknode-url"
            echo "  mem0-api-key"
            echo "  neo4j-uri"
            echo "  neo4j-username"
            echo "  neo4j-password"
            echo "  openai-api-key"
            echo "  cdp-api-key-name"
            echo "  cdp-api-key-private"
            echo ""
            read -p "Enter secret name to update: " secret_name
            read -s -p "Enter new value: " secret_value
            echo ""
            create_or_update_secret "$secret_name" "$secret_value"
            ;;
            
        3)
            echo ""
            list_secrets
            ;;
            
        4)
            echo ""
            read -p "Enter secret name to read: " secret_name
            echo "Secret value:"
            read_secret "$secret_name"
            ;;
            
        5)
            echo ""
            read -p "Enter secret name to delete: " secret_name
            read -p "Are you sure? (y/n): " confirm
            if [ "$confirm" = "y" ]; then
                $GCLOUD secrets delete $secret_name --project=$PROJECT_ID --quiet
                echo "✅ Secret deleted"
            fi
            ;;
            
        6)
            echo "Exiting..."
            exit 0
            ;;
            
        *)
            echo "Invalid choice"
            ;;
    esac
done