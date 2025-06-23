import os
import json
import openai
import gradio as gr
from pinecone import Pinecone # Keep this import for the main client

# ── CONFIGURE KEYS & CLIENTS ───────────────────────────────────────────────
# Load OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if OpenAI API key is set
if not openai.api_key:
    print("OPENAI_API_KEY environment variable not found. Please set it for production!")
    # For local testing, you might temporarily hardcode here, but REMOVE FOR PRODUCTION
    # openai.api_key = "sk-proj-YOUR_ACTUAL_OPENAI_KEY_HERE"

# 1) Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fb-comments")

# --- IMPORTANT CHANGE HERE ---
# Pinecone environment/host are now typically provided during Pinecone() initialization
# or are automatically inferred for serverless indexes.
# If your index is serverless, you might not need 'environment' or 'host' explicitly here.
# If you are using a Pod-based index, ensure PINECONE_ENVIRONMENT is set as a secret.
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # e.g., "us-east-1-aws"

if not PINECONE_API_KEY:
    print("PINECONE_API_KEY environment variable not found. Using hardcoded value for testing. Please set it for production!")
    PINECONE_API_KEY = "pcsk_2indG_7bikinqpKq6rseXDUYfWgbrNvjFEMvmSXt96tT6HQxejv76tpacdmm4N7jVoreK" # REMOVE THIS FOR PRODUCTION

# Initialize the main Pinecone client
# For Pinecone client v3.x.x+, you initialize Pinecone first, then get the index from it.
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT) # Pass environment if you have it

# Get the specific index instance
# --- IMPORTANT CHANGE HERE ---
# Access the index via the Pinecone client object.
index = pc.Index(PINECONE_INDEX_NAME)

# --- Helper Function to get Embeddings from OpenAI ---
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    """
    Generates an embedding for the given text using OpenAI's API.
    """
    if not text: # Handle empty text gracefully
        return []
    try:
        response = openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding from OpenAI: {e}")
        raise # Re-raise to propagate the error if necessary

# --- New Function to get Sentiment and Themes from OpenAI GPT-3.5-TURBO ---
def get_sentiment_and_themes_from_openai(text: str) -> tuple[str, list[str]]:
    """
    Analyzes sentiment and extracts themes from text using OpenAI's GPT-3.5-TURBO.
    """
    if not text:
        return "N/A", []
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis AI. Analyze the sentiment (positive, neutral, negative) and identify key themes (up to 3 words each) from the following user comment. Respond ONLY with a JSON object in this exact format: {\"sentiment\":\"[sentiment_label]\",\"themes\":[\"theme1\",\"theme2\"]}. If no clear themes, use an empty array. If the comment is very short or ambiguous, categorize as neutral. Do not add any other text outside the JSON."},
                {"role": "user", "content": f"Comment: {text}"}
            ],
            temperature=0.0 # Keep low for consistent JSON output
        )
        
        sentiment_json_str = response['choices'][0]['message']['content']
        sentiment_data = json.loads(sentiment_json_str)
        
        sentiment = sentiment_data.get("sentiment", "Sentiment not found")
        themes = sentiment_data.get("themes", [])
        return sentiment, themes
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Error processing sentiment/themes with GPT-3.5-TURBO: {e}")
        return "Error: Could not analyze sentiment", []

# --- MODIFIED ANALYZE FUNCTION to include direct sentiment and themes ---
def analyze_comment(comment_text: str) -> tuple[str, str, str]:
    if not comment_text:
        return "Please enter a comment.", "N/A", "N/A"

    # --- 1. Get Sentiment and Themes for the CURRENT typed comment ---
    current_sentiment, current_themes = get_sentiment_and_themes_from_openai(comment_text)
    current_themes_str = ", ".join(current_themes) if current_themes else "No themes identified."

    # --- 2. (Optional) Get Embedding and Search Pinecone for similar comments ---
    similar_comments_info = ""
    try:
        embedding = get_embedding(comment_text)

        pinecone_query_payload = {
            "vector": embedding,
            "top_k": 1,
            "namespace": "fb_comments",
            "include_metadata": True
        }

        resp = index.query(**pinecone_query_payload)

        if resp and resp.get("matches") and len(resp["matches"]) > 0:
            top_match = resp["matches"][0]
            if top_match.get("metadata"):
                similar_comments_info = (
                    f"**Most Similar Comment from Pinecone (Score: {top_match.score:.2f}):**\n"
                    f"Text: {top_match['metadata'].get('text', 'N/A')}\n"
                    f"Sentiment: {top_match['metadata'].get('sentiment', 'N/A')}\n"
                    f"Themes: {', '.join(top_match['metadata'].get('themes', [])) if top_match['metadata'].get('themes') else 'N/A'}\n"
                    f"ID: {top_match.id}"
                )
            else:
                similar_comments_info = "Top similar comment found, but metadata missing."
        else:
            similar_comments_info = "No similar comments found in Pinecone."
    except Exception as e:
        similar_comments_info = f"Error during Pinecone similarity search: {e}"
        print(similar_comments_info)

    return f"Sentiment: {current_sentiment}", f"Themes: {current_themes_str}", similar_comments_info

# --- MODIFIED GRADIO INTERFACE ---
demo = gr.Interface(
    fn=analyze_comment,
    inputs=gr.Textbox(
        lines=5,
        placeholder='Enter a comment to analyze its sentiment and find similar ones...',
        label="Comment Text"
    ),
    outputs=[
        gr.Textbox(label="Sentiment of Typed Comment"),
        gr.Textbox(label="Themes of Typed Comment"),
        gr.Textbox(label="Similar Comments from Pinecone")
    ],
    title="🎯 Comment Sentiment & Similarity Analyzer",
    description="Enter a comment to get its real-time sentiment and themes. The app will also search for similar comments stored in Pinecone."
)

if __name__ == "__main__":
    demo.launch()
