import os
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from azure.storage.queue.aio import QueueClient
from azure.storage.queue import BinaryBase64DecodePolicy
from dotenv import load_dotenv
from processor import process_pdf_to_faiss

# Load environment variables
load_dotenv()

# Configuration
NUM_LISTENER_THREADS = 4  # Number of parallel threads polling the queue
MESSAGES_PER_POLL = 5     # Messages to fetch in each async batch
POLL_INTERVAL = 2         # Seconds to wait if no messages are found

async def process_message(queue_client, msg):
    """
    Handles the actual processing of an individual message.
    """
    thread_id = threading.get_ident()
    try:
        # Parse the Event Grid message (JSON)
        event_data = json.loads(msg.content)
        
        # If it's a list (Event Grid Schema), take the first item
        if isinstance(event_data, list):
            event = event_data[0]
        else:
            event = event_data

        subject = event.get("subject", "Unknown")
        event_type = event.get("eventType", "Unknown")
        
        print(f"[Thread-{thread_id}] Event Received: {event_type} | {subject}")
        
        # Extract blob information
        data = event.get("data", {})
        url = data.get("url", "N/A")
        
        if url.lower().endswith(".pdf"):
            print(f"[Thread-{thread_id}] PDF detected. Starting ingestion pipeline...")
            # Use asyncio.to_thread to run the parsing (CPU-bound) without blocking this loop
            await asyncio.to_thread(process_pdf_to_faiss, url)
        else:
            print(f"[Thread-{thread_id}] Skipping non-PDF file: {url}")
            
        print(f"[Thread-{thread_id}] Finished processing message.")
        print("-" * 40)

        # Delete the message after successful processing
        await queue_client.delete_message(msg)
    
    except json.JSONDecodeError:
        print(f"[Thread-{thread_id}] Error: Received non-JSON message.")
        await queue_client.delete_message(msg)
    except Exception as e:
        print(f"[Thread-{thread_id}] Error processing message: {e}")

async def consume_worker(worker_id):
    """
    An individual worker that runs its own asyncio loop to poll the queue.
    """
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    queue_name = "blob-event-queue"

    if not connection_string:
        print(f"Worker-{worker_id} Error: AZURE_STORAGE_CONNECTION_STRING not set.")
        return

    print(f"Worker-{worker_id} starting...")

    # Use 'async with' to manage the client lifecycle
    async with QueueClient.from_connection_string(
        connection_string, 
        queue_name,
        message_decode_policy=BinaryBase64DecodePolicy()
    ) as queue_client:
        
        try:
            while True:
                # Receive messages asynchronously
                messages = queue_client.receive_messages(
                    messages_per_page=MESSAGES_PER_POLL, 
                    visibility_timeout=60 # Long timeout to allow for PDF parsing
                )
                
                tasks = []
                async for msg in messages:
                    # Each message within this worker is also handled concurrently via asyncio
                    tasks.append(asyncio.create_task(process_message(queue_client, msg)))

                if tasks:
                    await asyncio.gather(*tasks)
                else:
                    # Sleep only if we didn't find any messages
                    await asyncio.sleep(POLL_INTERVAL)

        except asyncio.CancelledError:
            print(f"Worker-{worker_id} stopping...")
        except Exception as e:
            print(f"Worker-{worker_id} critical error: {e}")

def start_event_loop(worker_id):
    """
    Entry point for a thread to run its own asyncio event loop.
    """
    asyncio.run(consume_worker(worker_id))

if __name__ == "__main__":
    print(f"Starting Highly Concurrent Consumer Architecture")
    print(f"Configuration: {NUM_LISTENER_THREADS} Threads, each running an AsyncIO loop.")
    print("Press Ctrl+C to stop.\n")

    # Use a ThreadPoolExecutor to scale out the listeners
    with ThreadPoolExecutor(max_workers=NUM_LISTENER_THREADS) as executor:
        try:
            # Launch multiple threads, each starting its own asyncio loop
            futures = [executor.submit(start_event_loop, i) for i in range(NUM_LISTENER_THREADS)]
            
            # Wait for any to raise an exception or for user interrupt
            for future in futures:
                future.result() 
                
        except KeyboardInterrupt:
            print("\nShutdown requested. Stopping all threads...")
            # Threads will be cleaned up by the executor context manager
