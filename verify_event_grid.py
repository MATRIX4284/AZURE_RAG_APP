import os
import time
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.storage.queue import QueueClient
from dotenv import load_dotenv

load_dotenv()

def verify_event_grid():
    # Configuration from environment
    storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "uploads"
    queue_name = "blob-event-queue"
    blob_name = "test-event-blob.txt"

    if not storage_connection_string:
        print("Error: AZURE_STORAGE_CONNECTION_STRING not set.")
        return

    try:
        # 1. Upload a file to Blob Storage
        print(f"Uploading {blob_name} to container '{container_name}'...")
        blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Ensure container exists
        if not container_client.exists():
            container_client.create_container()

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob("Hello Event Grid!", overwrite=True)
        print("Upload successful.")

        # 2. Wait for Event Grid to deliver the message
        print("Waiting 10 seconds for event delivery to queue...")
        time.sleep(10)

        # 3. Poll the Storage Queue
        print(f"Checking queue '{queue_name}' for messages...")
        queue_client = QueueClient.from_connection_string(storage_connection_string, queue_name)
        
        messages = queue_client.receive_messages(messages_per_page=1)
        found = False
        for msg in messages:
            print("\nSuccess! Received event message:")
            print(f"Message ID: {msg.id}")
            print(f"Content: {msg.content}")
            queue_client.delete_message(msg)
            found = True
            break
        
        if not found:
            print("\nNo messages found in the queue. Possible causes:")
            print("1. Event Grid Subscription is not configured correctly.")
            print("2. System Topic registration is still pending (can take a few minutes).")
            print("3. Filters in the subscription are excluding this event.")

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    verify_event_grid()
