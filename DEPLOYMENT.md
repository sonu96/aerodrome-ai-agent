# Deployment Guide - Aerodrome AI Agent API

This guide covers deploying the Aerodrome AI Agent API to Google Cloud Platform using both App Engine and Cloud Run.

## Prerequisites

### 1. Google Cloud Setup
```bash
# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable Required APIs
```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  appengine.googleapis.com \
  secretmanager.googleapis.com \
  sqladmin.googleapis.com \
  compute.googleapis.com
```

### 3. Required Accounts & Keys

You'll need accounts and API keys for:
- **Mem0**: Memory management service
- **Google Gemini**: AI/LLM service
- **QuickNode**: Ethereum RPC endpoint
- **Neo4j**: Graph database (optional, can use Aura cloud)

## Configuration

### 1. Environment Variables

Copy the example environment file and configure:
```bash
cp .env.example .env
# Edit .env with your actual values
```

### 2. Google Cloud Secrets

Store sensitive configuration in Secret Manager:
```bash
# API Key
echo -n "your-production-api-key" | \
  gcloud secrets create aerodrome-api-key --data-file=-

# Mem0 API Key  
echo -n "your-mem0-api-key" | \
  gcloud secrets create mem0-api-key --data-file=-

# Gemini API Key
echo -n "your-gemini-api-key" | \
  gcloud secrets create gemini-api-key --data-file=-

# QuickNode URL
echo -n "https://your-endpoint.quiknode.pro/token/" | \
  gcloud secrets create quicknode-url --data-file=-

# Neo4j Configuration
echo -n "bolt://your-neo4j-instance:7687" | \
  gcloud secrets create neo4j-uri --data-file=-
echo -n "neo4j" | \
  gcloud secrets create neo4j-username --data-file=-
echo -n "your-neo4j-password" | \
  gcloud secrets create neo4j-password --data-file=-
```

## Deployment Options

### Option 1: Google Cloud Run (Recommended)

Cloud Run offers better scalability and cost efficiency for API services.

#### 1. Build and Deploy
```bash
# Deploy using Cloud Build
gcloud builds submit --config cloudbuild.yaml

# Or manual deployment
docker build -t gcr.io/YOUR_PROJECT_ID/aerodrome-ai-agent .
docker push gcr.io/YOUR_PROJECT_ID/aerodrome-ai-agent

gcloud run deploy aerodrome-ai-agent \
  --image gcr.io/YOUR_PROJECT_ID/aerodrome-ai-agent \
  --region us-central1 \
  --platform managed \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 1 \
  --max-instances 10 \
  --port 8080 \
  --allow-unauthenticated
```

#### 2. Configure Environment
```bash
gcloud run services update aerodrome-ai-agent \
  --region us-central1 \
  --set-env-vars="ENV=production,LOG_LEVEL=INFO" \
  --set-secrets="API_KEY=aerodrome-api-key:latest,MEM0_API_KEY=mem0-api-key:latest"
```

### Option 2: Google App Engine

App Engine provides managed hosting with automatic scaling.

#### 1. Configure app.yaml
Edit `app.yaml` to match your project configuration:
- Update `cloud_sql_instances` if using Cloud SQL
- Configure `vpc_access_connector` if using VPC
- Set appropriate `instance_class` and scaling parameters

#### 2. Deploy
```bash
gcloud app deploy app.yaml --quiet
```

## Database Setup (Optional)

### Neo4j Setup
If using Neo4j for memory graphs:

#### Option A: Neo4j Aura (Managed)
1. Create account at https://neo4j.com/cloud/aura/
2. Create a database instance
3. Note connection details for configuration

#### Option B: Self-hosted on GCE
```bash
# Create VM instance
gcloud compute instances create neo4j-instance \
  --machine-type e2-standard-2 \
  --image-family ubuntu-2004-lts \
  --image-project ubuntu-os-cloud \
  --boot-disk-size 50GB

# Install Neo4j (connect to instance first)
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 4.4' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update && sudo apt install neo4j
```

## Monitoring & Observability

### 1. Cloud Logging
Logs are automatically collected by Google Cloud Logging.

### 2. Cloud Monitoring
The API exposes Prometheus metrics at `/metrics`:
```bash
# View metrics
curl https://your-service-url/metrics
```

### 3. Health Checks
- Liveness: `/health/live`
- Readiness: `/health/ready`
- Detailed: `/health`

### 4. Set up Monitoring
```bash
# Create uptime check
gcloud monitoring uptime-check-configs create \
  --resource-labels=service_name=aerodrome-ai-agent \
  --resource-type=gce_instance \
  --host your-service-url \
  --path /health/live
```

## Security Configuration

### 1. Identity and Access Management
```bash
# Create service account
gcloud iam service-accounts create aerodrome-api-sa \
  --display-name "Aerodrome API Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member serviceAccount:aerodrome-api-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --role roles/secretmanager.secretAccessor

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member serviceAccount:aerodrome-api-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --role roles/cloudsql.client
```

### 2. VPC Configuration (Optional)
For enhanced security, deploy in a VPC:
```bash
# Create VPC
gcloud compute networks create aerodrome-vpc --subnet-mode regional

# Create subnet
gcloud compute networks subnets create aerodrome-subnet \
  --network aerodrome-vpc \
  --region us-central1 \
  --range 10.0.0.0/24
```

## CI/CD Pipeline

The included `cloudbuild.yaml` provides a complete CI/CD pipeline:

### 1. Automated Deployment
```bash
# Create build trigger
gcloud builds triggers create github \
  --repo-name=aerodrome-ai-agent \
  --repo-owner=YOUR_GITHUB_USERNAME \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml
```

### 2. Manual Deployment
```bash
# Build and deploy
gcloud builds submit --config cloudbuild.yaml

# Check build status
gcloud builds list --limit=5
```

## Troubleshooting

### 1. Check Logs
```bash
# Cloud Run logs
gcloud logs read --service=aerodrome-ai-agent --limit=50

# App Engine logs
gcloud logs read --service=default --limit=50
```

### 2. Debug Health Issues
```bash
# Check service status
curl https://your-service-url/health

# Check detailed metrics
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://your-service-url/metrics/detailed
```

### 3. Common Issues

#### Memory Issues
- Increase memory allocation in Cloud Run/App Engine
- Check for memory leaks in brain components
- Monitor memory usage metrics

#### Timeout Issues  
- Increase timeout settings
- Check brain initialization time
- Verify external service connectivity

#### Rate Limiting
- Monitor rate limit headers
- Adjust limits in configuration
- Scale up instances if needed

## Performance Tuning

### 1. Instance Configuration
```bash
# Cloud Run optimization
gcloud run services update aerodrome-ai-agent \
  --concurrency 10 \
  --cpu 2 \
  --memory 4Gi \
  --max-instances 20
```

### 2. Database Optimization
- Use connection pooling
- Optimize Neo4j queries
- Consider Redis for caching

### 3. Monitoring Performance
- Monitor response times
- Track error rates
- Set up alerting for SLA violations

## Cost Optimization

### 1. Right-sizing
- Start with minimal resources
- Scale based on actual usage
- Use preemptible instances where appropriate

### 2. Request Optimization
- Implement proper caching
- Use connection pooling
- Optimize database queries

### 3. Resource Scheduling
- Scale down during low usage periods
- Use minimum instances judiciously
- Monitor and adjust based on usage patterns

## Production Checklist

- [ ] All secrets stored in Secret Manager
- [ ] Service accounts with minimal permissions
- [ ] Health checks configured
- [ ] Monitoring and alerting set up
- [ ] Backup and disaster recovery plan
- [ ] Load testing completed
- [ ] Security scanning performed
- [ ] Documentation updated
- [ ] Runbooks created for common issues

## Support

For deployment issues:
1. Check the troubleshooting section above
2. Review Cloud Console logs and monitoring
3. Validate configuration and secrets
4. Test individual components separately

For production issues:
1. Check service health endpoints
2. Review metrics and alerts
3. Scale resources if needed
4. Follow incident response procedures