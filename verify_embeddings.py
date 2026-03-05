import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv(override=True)
print(f"Loaded .env from: {os.path.abspath('.env')}")

def verify_embeddings():
    print("--- Azure OpenAI Embeddings Verification ---")
    
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    
    print(f"Testing Deployment: {deployment}")
    print(f"Endpoint: {endpoint}")
    
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        
        # For AI Foundry Projects, we need the Machine Learning scope
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://ai.azure.com/.default"
        )
        
        # Initialize the LangChain embeddings client with the token provider
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_ad_token_provider=token_provider,
        )
        
        # Test string
        test_text = "Hello, this is a test of the Azure OpenAI embedding deployment."
        
        print("Sending embedding request...")
        vector = embeddings.embed_query(test_text)
        
        print("✅ SUCCESS!")
        print(f"Generated vector length: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")
        
    except Exception as e:
        print(f"❌ FAILED!")
        print(f"Error: {e}")
        print("\nSuggestions:")
        #print(1. "Ensure the deployment name '{deployment}' is exactly what you see in the Azure AI Foundry portal.")
        #print(2. "Ensure your Service Principal has the 'Cognitive Services OpenAI User' role on the resource.")
        #print(3. "Wait 5 minutes if you just created the deployment.")

if __name__ == "__main__":
    verify_embeddings()
