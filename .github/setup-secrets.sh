#!/bin/bash

###############################################################################
# GitHub Secrets Setup Script
# Helps you configure GitHub secrets for CI/CD
###############################################################################

set -e

echo "==========================================="
echo "GitHub Secrets Setup for DeepSeek-OCR CI/CD"
echo "==========================================="
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not found. Please install it first:"
    echo "   https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub CLI"
    echo "   Run: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI is installed and authenticated"
echo ""

# Get repository info
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "Repository: $REPO"
echo ""

# AWS credentials
echo "📋 Step 1: AWS Credentials"
echo "================================"
read -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
read -sp "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
echo ""

gh secret set AWS_ACCESS_KEY_ID -b"$AWS_ACCESS_KEY_ID"
gh secret set AWS_SECRET_ACCESS_KEY -b"$AWS_SECRET_ACCESS_KEY"

echo "✅ AWS credentials set"
echo ""

# EC2 Instance ID
echo "📋 Step 2: EC2 Instance Configuration"
echo "================================"
read -p "EC2 Instance ID (e.g., i-1234567890abcdef0): " EC2_INSTANCE_ID

gh secret set EC2_INSTANCE_ID -b"$EC2_INSTANCE_ID"

echo "✅ EC2 instance ID set"
echo ""

# ECR Registry
echo "📋 Step 3: ECR Registry"
echo "================================"
echo "Find your ECR registry URL:"
echo "  aws ecr describe-repositories --repository-names deepseek-ocr-service --query 'repositories[0].repositoryUri' --output text"
echo ""
read -p "ECR Registry URL (e.g., 123456789012.dkr.ecr.us-east-1.amazonaws.com): " ECR_REGISTRY

gh secret set ECR_REGISTRY -b"$ECR_REGISTRY"

echo "✅ ECR registry set"
echo ""

# Optional: Slack webhook
echo "📋 Step 4: Slack Notifications (Optional)"
echo "================================"
read -p "Slack Webhook URL (press Enter to skip): " SLACK_WEBHOOK_URL

if [ -n "$SLACK_WEBHOOK_URL" ]; then
    gh secret set SLACK_WEBHOOK_URL -b"$SLACK_WEBHOOK_URL"
    echo "✅ Slack webhook set"
else
    echo "⏭️  Skipped Slack webhook"
fi

echo ""
echo "==========================================="
echo "✅ All secrets configured!"
echo "==========================================="
echo ""
echo "Configured secrets:"
gh secret list

echo ""
echo "Next steps:"
echo "1. Push your code to GitHub"
echo "2. Create ECR repository if not exists:"
echo "   aws ecr create-repository --repository-name deepseek-ocr-service --region us-east-1"
echo "3. Deploy EC2 instance with IAM role for SSM"
echo "4. Push code to trigger CI/CD:"
echo "   git push origin main"
echo ""
echo "View workflows at:"
echo "https://github.com/$REPO/actions"
echo ""
