#!/bin/bash

# Configuration
RG="kaustav_rg"
LOCATION="canadacentral" # or your preferred region
ENVIRONMENT="milvus-env"
STORAGE_ACCOUNT="kaustavst" # Your existing storage account
CONTAINER_NAME="milvus-storage"
STORAGE_ACCESS_KEY=$(az storage account keys list -n $STORAGE_ACCOUNT -g $RG --query "[0].value" -o tsv)

echo "Creating Resource Group..."
az group create --name $RG --location $LOCATION

echo "Creating Azure Container Apps Environment..."
az containerapp env create --name $ENVIRONMENT --resource-group $RG --location $LOCATION

echo "Creating Blob Container for Milvus storage..."
az storage container create --name $CONTAINER_NAME --account-name $STORAGE_ACCOUNT --account-key $STORAGE_ACCESS_KEY

# Deployment of etcd (metadata storage)
echo "Deploying etcd..."
az containerapp create \
  --name milvus-etcd \
  --resource-group $RG \
  --environment $ENVIRONMENT \
  --image quay.io/coreos/etcd:v3.5.5 \
  --cpu 0.5 --memory 1.0Gi \
  --command "etcd" \
  --args "--advertise-client-urls http://0.0.0.0:2379 --listen-client-urls http://0.0.0.0:2379 --data-dir /etcd-data" \
  --target-port 2379 \
  --ingress internal

ETCD_URL="http://milvus-etcd.internal.$(az containerapp env show -n $ENVIRONMENT -g $RG --query "properties.defaultDomain" -o tsv):2379"
echo "etcd endpoint: $ETCD_URL"

# Deployment of Milvus Standalone
echo "Deploying Milvus Standalone..."
az containerapp create \
  --name milvus-standalone \
  --resource-group $RG \
  --environment $ENVIRONMENT \
  --image milvusdb/milvus:v2.3.0 \
  --cpu 2.0 --memory 4.0Gi \
  --target-port 19530 \
  --ingress external \
  --env-vars \
    ETCD_ENDPOINTS=$ETCD_URL \
    MINIO_ADDRESS=https://$STORAGE_ACCOUNT.blob.core.windows.net \
    MINIO_ACCESS_KEY=$STORAGE_ACCOUNT \
    MINIO_SECRET_KEY=$STORAGE_ACCESS_KEY \
    MINIO_USE_SSL=true \
    MINIO_BUCKET_NAME=$CONTAINER_NAME \
    MINIO_ADDRESS_TYPE=azure \
    COMMON_STORAGE_TYPE=minio

MILVUS_HOSTNAME=$(az containerapp show -n milvus-standalone -g $RG --query "properties.configuration.ingress.fqdn" -o tsv)
echo "--------------------------------------------------"
echo "Milvus Deployment SUCCESS!"
echo "Endpoint: $MILVUS_HOSTNAME"
echo "Port: 19530 (gRPC)"
echo "--------------------------------------------------"
echo "Update your .env file with:"
echo "MILVUS_URL=https://$MILVUS_HOSTNAME"
