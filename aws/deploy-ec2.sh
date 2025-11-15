#!/bin/bash

###############################################################################
# DeepSeek-OCR EC2 Deployment Script
# Provisions AWS EC2 g4dn.xlarge instance with GPU support
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 DeepSeek-OCR AWS EC2 Deployment${NC}"
echo "================================================"

# Configuration
INSTANCE_TYPE="g4dn.xlarge"  # NVIDIA T4, 16GB GPU
AMI_ID=""  # Will be auto-detected for Deep Learning AMI
REGION="us-east-1"
KEY_NAME=""
SECURITY_GROUP_NAME="deepseek-ocr-sg"
INSTANCE_NAME="deepseek-ocr-service"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Please install it first.${NC}"
    echo "   Install: https://aws.amazon.com/cli/"
    exit 1
fi

# Prompt for configuration
echo ""
echo -e "${YELLOW}📋 Configuration${NC}"
read -p "AWS Region [us-east-1]: " input_region
REGION=${input_region:-$REGION}

read -p "EC2 Key Pair Name (required): " KEY_NAME
if [ -z "$KEY_NAME" ]; then
    echo -e "${RED}❌ Key pair name is required${NC}"
    exit 1
fi

read -p "API Key for DeepSeek-OCR (leave empty for no auth): " API_KEY

echo ""
echo -e "${YELLOW}🔍 Configuration Summary:${NC}"
echo "   Region: $REGION"
echo "   Instance Type: $INSTANCE_TYPE (~\$0.526/hour)"
echo "   Key Pair: $KEY_NAME"
echo "   Security Group: $SECURITY_GROUP_NAME"
echo ""
read -p "Continue? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Deployment cancelled."
    exit 0
fi

# Get latest Deep Learning AMI (Ubuntu 22.04 with NVIDIA drivers)
echo ""
echo -e "${YELLOW}🔍 Finding latest Deep Learning AMI...${NC}"
AMI_ID=$(aws ec2 describe-images \
    --region $REGION \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)

if [ -z "$AMI_ID" ]; then
    echo -e "${RED}❌ Could not find Deep Learning AMI${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found AMI: $AMI_ID${NC}"

# Create security group if it doesn't exist
echo ""
echo -e "${YELLOW}🔒 Setting up Security Group...${NC}"

SG_EXISTS=$(aws ec2 describe-security-groups \
    --region $REGION \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "None")

if [ "$SG_EXISTS" = "None" ]; then
    echo "Creating new security group..."

    SG_ID=$(aws ec2 create-security-group \
        --region $REGION \
        --group-name $SECURITY_GROUP_NAME \
        --description "Security group for DeepSeek-OCR service" \
        --query 'GroupId' \
        --output text)

    # Allow SSH
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0

    # Allow HTTP
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0

    # Allow HTTPS
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0

    # Allow API port (optional, for direct access)
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 8000 \
        --cidr 0.0.0.0/0

    echo -e "${GREEN}✅ Created security group: $SG_ID${NC}"
else
    SG_ID=$SG_EXISTS
    echo -e "${GREEN}✅ Using existing security group: $SG_ID${NC}"
fi

# Prepare user data script
USER_DATA=$(cat user-data.sh | base64 -w 0)

# Create environment file content
ENV_CONTENT="REQUIRE_API_KEY=true\nAPI_KEY=${API_KEY:-your-secret-api-key-change-this}"

# Launch EC2 instance
echo ""
echo -e "${YELLOW}🚀 Launching EC2 instance...${NC}"

INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --user-data file://user-data.sh \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}✅ Instance launched: $INSTANCE_ID${NC}"

# Wait for instance to be running
echo ""
echo -e "${YELLOW}⏳ Waiting for instance to be running...${NC}"
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --region $REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo -e "${GREEN}✅ Instance is running!${NC}"
echo ""
echo "================================================"
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo "================================================"
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "⏳ The service will take ~10-15 minutes to fully start."
echo "   (Downloading model weights ~6.6GB + building Docker image)"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. SSH into instance:"
echo "   ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$PUBLIC_IP"
echo ""
echo "2. Check deployment logs:"
echo "   tail -f /var/log/cloud-init-output.log"
echo ""
echo "3. Check Docker logs:"
echo "   docker-compose -f /opt/deepseek-ocr-service/docker-compose.yml logs -f"
echo ""
echo "4. Test the API:"
echo "   curl http://$PUBLIC_IP/health"
echo ""
echo "5. API Endpoint:"
echo "   http://$PUBLIC_IP/ocr/process"
echo ""
echo "6. API Documentation:"
echo "   http://$PUBLIC_IP/docs"
echo ""
echo "💰 Cost: ~\$0.526/hour for g4dn.xlarge"
echo ""
echo "⚠️  Remember to stop/terminate the instance when not in use!"
echo "   aws ec2 stop-instances --region $REGION --instance-ids $INSTANCE_ID"
echo ""
