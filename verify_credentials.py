import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

# Load environment variables from .env
load_dotenv()

def verify_azure_openai():
    """
    Verifies Azure OpenAI connection using Service Principal (via DefaultAzureCredential)
    or API Key if provided.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        print("Error: AZURE_OPENAI_ENDPOINT not set in .env")
        return

    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment_id = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

    print(f"Connecting to: {endpoint}")
    print(f"Deployment: {deployment_id}")

    try:
        # Check if we should use Service Principal or API Key
        if os.getenv("AZURE_OPENAI_API_KEY"):
            print("Using API Key authentication...")
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=api_version,
                azure_endpoint=endpoint
            )
        else:
            print("Using Service Principal (DefaultAzureCredential) authentication...")
            # Use explicit token retrieval to debug audience issues
            credential = DefaultAzureCredential()
            # The error message suggests cognitiveservices.azure.com or ai.azure.com
            scope = "https://ai.azure.com/.default"
            print(f"Requesting token for scope: {scope}")
            token_obj = credential.get_token(scope)
            
            # If the endpoint contains '/api/projects/', it's a Foundry Inference endpoint
            if "/api/projects/" in endpoint:
                print("Detected AI Foundry Project Inference endpoint.")
                client = AzureOpenAI(
                    azure_ad_token=token_obj.token,
                    api_version=api_version,
                    azure_endpoint=endpoint
                )
            else:
                client = AzureOpenAI(
                    azure_ad_token=token_obj.token,
                    api_version=api_version,
                    azure_endpoint=endpoint
                )

        # Simple test call
        response = client.chat.completions.create(
            model=deployment_id,
            messages=[{"role": "user", "content": "Say hello!"}],
            max_completion_tokens=50
        )
        print("\nSuccess! Connection established.")
        print("in first block")
        content = response.choices[0].message.content
        if content:
            print(f"Response: {content}")
        else:
            print("Response: <empty response content>")
            print(f"Finish Reason: {response.choices[0].finish_reason}")
            print(f"Full Message Object: {response.choices[0].message}")

    except Exception as e:
        print(f"\nError connecting to Azure OpenAI: {e}")
        print("\nCheck your .env file and ensure:")
        print("1. Service Principal has 'Cognitive Services OpenAI User' role.")
        print("2. Endpoint and Deployment Name are correct.")
        print("3. AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_TENANT_ID are set correctly.")

if __name__ == "__main__":
    verify_azure_openai()
