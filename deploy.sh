#!/bin/bash

# RAG Chat App Kubernetes Deployment Script
# Usage: ./deploy.sh [environment] [action]
# Environments: dev, staging, prod
# Actions: deploy, delete, update, status

set -e

# Configuration
ENVIRONMENT=${1:-dev}
ACTION=${2:-deploy}
NAMESPACE="rag-chat-app"
KUBECTL_TIMEOUT="300s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check Docker images exist (optional)
    if [[ "$ACTION" == "deploy" ]]; then
        log_info "Checking Docker images..."
        # Add image checks here if needed
    fi
    
    log_success "Prerequisites check passed"
}

# Create secrets
setup_secrets() {
    log_info "Setting up secrets..."
    
    # Check if secrets exist
    if kubectl get secret api-secrets -n $NAMESPACE &> /dev/null; then
        log_warning "Secrets already exist, skipping creation"
        return
    fi
    
    # Prompt for API keys
    read -p "Enter OpenAI API Key: " -s OPENAI_KEY
    echo
    read -p "Enter Pinecone API Key: " -s PINECONE_KEY
    echo
    
    if [[ -z "$OPENAI_KEY" || -z "$PINECONE_KEY" ]]; then
        log_error "API keys are required"
        exit 1
    fi
    
    # Create secret
    kubectl create secret generic api-secrets \
        --from-literal=OPENAI_API_KEY="$OPENAI_KEY" \
        --from-literal=PINECONE_API_KEY="$PINECONE_KEY" \
        -n $NAMESPACE
    
    log_success "Secrets created successfully"
}

# Deploy application
deploy_app() {
    log_info "Deploying RAG Chat App to $ENVIRONMENT environment..."
    
    # Apply configurations in order
    log_info "Creating namespace and configurations..."
    kubectl apply -f k8s/00-namespace-config.yaml --timeout=$KUBECTL_TIMEOUT
    
    log_info "Creating secrets and persistent volumes..."
    kubectl apply -f k8s/01-secrets-pvc.yaml --timeout=$KUBECTL_TIMEOUT
    setup_secrets
    
    log_info "Deploying backend..."
    kubectl apply -f k8s/02-backend.yaml --timeout=$KUBECTL_TIMEOUT
    
    log_info "Deploying frontend..."
    kubectl apply -f k8s/03-frontend.yaml --timeout=$KUBECTL_TIMEOUT
    
    log_info "Setting up ingress..."
    kubectl apply -f k8s/04-ingress.yaml --timeout=$KUBECTL_TIMEOUT
    
    log_info "Configuring autoscaling..."
    kubectl apply -f k8s/05-autoscaling.yaml --timeout=$KUBECTL_TIMEOUT
    
    log_info "Setting up monitoring and security..."
    kubectl apply -f k8s/06-monitoring-security.yaml --timeout=$KUBECTL_TIMEOUT
    
    # Wait for deployments
    log_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=$KUBECTL_TIMEOUT deployment/rag-chat-backend -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=$KUBECTL_TIMEOUT deployment/rag-chat-frontend -n $NAMESPACE
    
    log_success "Deployment completed successfully!"
    show_status
}

# Delete application
delete_app() {
    log_warning "Deleting RAG Chat App from $ENVIRONMENT environment..."
    
    read -p "Are you sure you want to delete the entire application? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deletion cancelled"
        exit 0
    fi
    
    # Delete in reverse order
    kubectl delete -f k8s/06-monitoring-security.yaml --ignore-not-found=true
    kubectl delete -f k8s/05-autoscaling.yaml --ignore-not-found=true
    kubectl delete -f k8s/04-ingress.yaml --ignore-not-found=true
    kubectl delete -f k8s/03-frontend.yaml --ignore-not-found=true
    kubectl delete -f k8s/02-backend.yaml --ignore-not-found=true
    kubectl delete -f k8s/01-secrets-pvc.yaml --ignore-not-found=true
    kubectl delete -f k8s/00-namespace-config.yaml --ignore-not-found=true
    
    log_success "Application deleted successfully"
}

# Update application
update_app() {
    log_info "Updating RAG Chat App..."
    
    # Rolling update
    kubectl rollout restart deployment/rag-chat-backend -n $NAMESPACE
    kubectl rollout restart deployment/rag-chat-frontend -n $NAMESPACE
    
    # Wait for rollout
    kubectl rollout status deployment/rag-chat-backend -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    kubectl rollout status deployment/rag-chat-frontend -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    
    log_success "Update completed successfully!"
    show_status
}

# Show application status
show_status() {
    log_info "Application Status:"
    echo
    
    echo "📊 Pods:"
    kubectl get pods -n $NAMESPACE -o wide
    echo
    
    echo "🚀 Services:"
    kubectl get services -n $NAMESPACE
    echo
    
    echo "🌐 Ingress:"
    kubectl get ingress -n $NAMESPACE
    echo
    
    echo "💾 Persistent Volumes:"
    kubectl get pvc -n $NAMESPACE
    echo
    
    echo "📈 HPA:"
    kubectl get hpa -n $NAMESPACE
    echo
    
    # Get external IP/URL
    log_info "Getting access information..."
    INGRESS_IP=$(kubectl get ingress rag-chat-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    INGRESS_HOST=$(kubectl get ingress rag-chat-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    
    if [[ -n "$INGRESS_IP" ]]; then
        log_success "Application accessible at: http://$INGRESS_IP"
    elif [[ -n "$INGRESS_HOST" ]]; then
        log_success "Application accessible at: http://$INGRESS_HOST"
    else
        log_warning "Ingress IP not yet assigned. Use 'kubectl get ingress -n $NAMESPACE' to check later."
    fi
}

# Get logs
get_logs() {
    log_info "Getting application logs..."
    
    echo "Backend logs:"
    kubectl logs -l app=rag-chat-backend -n $NAMESPACE --tail=50
    echo
    
    echo "Frontend logs:"
    kubectl logs -l app=rag-chat-frontend -n $NAMESPACE --tail=50
}

# Main execution
main() {
    echo "🚀 RAG Chat App Kubernetes Deployment"
    echo "Environment: $ENVIRONMENT"
    echo "Action: $ACTION"
    echo
    
    check_prerequisites
    
    case $ACTION in
        deploy)
            deploy_app
            ;;
        delete)
            delete_app
            ;;
        update)
            update_app
            ;;
        status)
            show_status
            ;;
        logs)
            get_logs
            ;;
        *)
            log_error "Unknown action: $ACTION"
            echo "Usage: $0 [environment] [action]"
            echo "Actions: deploy, delete, update, status, logs"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"