# CI/CD Configuration Guide

## Overview
This guide explains how to set up and configure the CI/CD pipeline for the RAG Chat Application.

## Prerequisites

### 1. GitHub Container Registry
The pipeline uses GitHub Container Registry (ghcr.io) to store Docker images. No additional setup is required as it uses the default `GITHUB_TOKEN`.

### 2. Kubernetes Clusters
You need Kubernetes clusters for staging and production environments.

### 3. Required Secrets
Add these secrets to your GitHub repository:

#### Repository Secrets (Settings → Secrets and variables → Actions)

- `KUBE_CONFIG_STAGING`: Base64 encoded kubeconfig for staging cluster
- `KUBE_CONFIG_PROD`: Base64 encoded kubeconfig for production cluster

To get the base64 encoded kubeconfig:
```bash
# For staging
cat ~/.kube/staging-config | base64 -w 0

# For production  
cat ~/.kube/prod-config | base64 -w 0
```

### 4. Environment Configuration
Set up GitHub Environments:

1. Go to Settings → Environments
2. Create `staging` environment
3. Create `production` environment
4. Add protection rules for production (require reviews, restrict to main branch, etc.)

## Pipeline Workflow

### Triggers
- **Push to main/develop**: Runs full pipeline
- **Tags (v*)**: Deploys to production
- **Pull Requests**: Runs tests and builds

### Jobs

#### 1. Test Job
- Runs Python and Node.js tests
- Performs linting
- Validates code quality

#### 2. Build Job
- Builds Docker images for backend and frontend
- Pushes to GitHub Container Registry
- Uses BuildKit for efficient caching

#### 3. Deploy Jobs
- **Staging**: Auto-deploys on `develop` branch
- **Production**: Auto-deploys on version tags (`v*`)

#### 4. Security Scan
- Runs Trivy vulnerability scanner
- Uploads results to GitHub Security tab

## Configuration Steps

### 1. Update Docker Images
If you change the image names, update these files:
- `.github/workflows/ci-cd.yml` (env section)
- `k8s/02-backend.yaml`
- `k8s/03-frontend.yaml`

### 2. Customize Deployment
Edit the deployment script sections in `ci-cd.yml` to match your cluster configuration.

### 3. Add Tests
Update the test sections in the pipeline:
```yaml
# Backend tests
- name: Run Python tests
  run: |
    cd backend
    python -m pytest tests/

# Frontend tests  
- name: Run Node.js tests
  run: |
    cd frontend
    npm run test
```

### 4. Configure Smoke Tests
Add production verification in the deploy-production job:
```yaml
- name: Run smoke tests
  run: |
    INGRESS_IP=$(kubectl get ingress rag-chat-ingress -n rag-chat-app-prod -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    curl -f http://$INGRESS_IP/health
```

## Deployment Process

### Staging Deployment
1. Push to `develop` branch
2. Pipeline runs tests and builds images
3. Automatically deploys to staging cluster
4. Updates namespace: `rag-chat-app-staging`

### Production Deployment
1. Create and push a version tag:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
2. Pipeline runs full workflow
3. Deploys to production with approval (if environment protection is enabled)
4. Updates namespace: `rag-chat-app-prod`

## Monitoring and Troubleshooting

### View Pipeline Status
- Go to Actions tab in GitHub repository
- Click on specific workflow run for details

### Check Deployment Status
```bash
# Staging
kubectl get pods -n rag-chat-app-staging
kubectl logs -f deployment/rag-chat-backend -n rag-chat-app-staging

# Production
kubectl get pods -n rag-chat-app-prod
kubectl logs -f deployment/rag-chat-backend -n rag-chat-app-prod
```

### Common Issues

1. **Image pull errors**: Check if GitHub Container Registry permissions are correct
2. **Deployment timeouts**: Increase timeout values in workflow
3. **Config errors**: Verify kubeconfig secrets are properly base64 encoded

## Security Best Practices

1. **Environment Protection**: Enable required reviewers for production
2. **Secret Rotation**: Regularly rotate kubeconfig and other secrets
3. **Image Scanning**: Pipeline includes Trivy security scanning
4. **Branch Protection**: Protect main branch and require PR reviews

## Customization

### Adding New Environments
1. Create new environment in GitHub
2. Add corresponding kubeconfig secret
3. Add new deployment job in workflow
4. Update namespace configurations

### Modifying Build Process
- Update Dockerfile paths in `docker/build-push-action`
- Adjust build contexts for multi-service applications
- Configure additional build arguments if needed

## Local Testing

Test the deployment locally:
```bash
# Test with our deploy script
./deploy.sh deploy dev

# Or manually apply manifests
kubectl apply -f k8s/ -n your-namespace
```