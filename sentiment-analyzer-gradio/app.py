import os
import json
import gradio as gr
from pinecone import Pinecone # Import the Pinecone class

# 1) Initialize Pinecone
# It's crucial to set these environment variables in your Hugging Face Space settings
# For example, in your Hugging Face Space settings, go to "Environment variables"
# Add a variable named: PINECONE_API_KEY (with your actual API key as its value)
# And optionally: PINECONE_ENVIRONMENT (with your actual environment value if still needed for other operations)

# Retrieve API key from environment variable
# If you don't set PINECONE_API_KEY as an env var, this will be None.
# If you're running locally and want to test, you can hardcode it temporarily, but remove it for deployment.
api_key = os.getenv("PINECONE_API_KEY")

# Check if the API key is set
if not api_key:
    # Fallback to the hardcoded value if the environment variable isn't set.
    # This is not recommended for production but might be what you were trying to do initially.
    # If "pcsk_2indG_7bikinqpKq6rseXDUYfWgbrNvjFEMvmSXt96tT6HQxejv76tpacdmm4N7jVoreK" IS your actual API key string.
    print("PINECONE_API_KEY environment variable not found. Using hardcoded value. Please set it for production!")
    api_key = "pcsk_2indG_7bikinqpKq6rseXDUYfWgbrNvjFEMvmSXt96tT6HQxejv76tpacdmm4N7jVoreK"

# Initialize the Pinecone client
pc = Pinecone(api_key=api_key)

# Access your specific index
# Assuming "fb-comments-q4h4rly" is the name of your Pinecone index.
# You can also get this from an environment variable:
index_name = os.getenv("PINECONE_INDEX_NAME", "fb-comments-q4h4rly") # Default to your index name
index = pc.Index(index_name)

def analyze(raw_query_json: str) -> str:
    try:
        payload = json.loads(raw_query_json)
        resp = index.query(**payload)
        # Ensure 'matches' and 'metadata' exist before accessing
        if resp and resp.get("matches") and len(resp["matches"]) > 0 and resp["matches"][0].get("metadata"):
            return resp["matches"][0]["metadata"].get("sentiment", "Sentiment not found")
        else:
            return "No matches found or sentiment metadata missing."
    except json.JSONDecodeError:
        return "Error: Invalid JSON payload."
    except Exception as e:
        return f"Error: {e}"

# 3) Build the Gradio interface
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(
        lines=10,
        placeholder='Paste JSON: {"namespace":..., "topK":1, ...}',
        label="Raw Query JSON"
    ),
    outputs=gr.Label(label="Sentiment"),
    title="🎯 Sentiment Analyzer",
    description="Paste in a Pinecone query JSON and get back the top comment’s sentiment."
)

if __name__ == "__main__":
    demo.launch()
