import os
import json
from openai import OpenAI
import gradio as gr
from pinecone import Pinecone

# ── CONFIGURE KEYS & CLIENTS ───────────────────────────────────────────────
# Get OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("OPENAI_API_KEY environment variable not found. Please set it for production!")
    raise ValueError("OPENAI_API_KEY is required")

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fb-comments")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if not PINECONE_API_KEY:
    print("PINECONE_API_KEY environment variable not found. Using hardcoded value for testing. Please set it for production!")
    PINECONE_API_KEY = "pcsk_2indG_7bikinqpKq6rseXDUYfWgbrNvjFEMvmSXt96tT6HQxejv76tpacdmm4N7jVoreK"

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(PINECONE_INDEX_NAME)

def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    if not text:
        return []
    try:
        response = openai_client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding from OpenAI: {e}")
        raise

def get_sentiment_and_themes_from_openai(text: str) -> tuple[str, list[str]]:
    if not text:
        return "N/A", []
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis AI. Analyze the sentiment (positive, neutral, negative) and identify key themes (up to 3 words each) from the following user comment. Respond ONLY with a JSON object in this exact format: {\"sentiment\":\"[sentiment_label]\",\"themes\":[\"theme1\",\"theme2\"]}. If no clear themes, use an empty array. If the comment is very short or ambiguous, categorize as neutral. Do not add any other text outside the JSON."},
                {"role": "user", "content": f"Comment: {text}"}
            ],
            temperature=0.1
        )
        
        # --- ADDED DEBUGGING PRINTS HERE ---
        print(f"Raw OpenAI ChatCompletion response: {response}")
        
        sentiment_json_str = response.choices[0].message.content
        print(f"String to parse as JSON: {sentiment_json_str}")
        
        # Attempt to parse
        sentiment_data = json.loads(sentiment_json_str)
        
        sentiment = sentiment_data.get("sentiment", "Sentiment not found")
        themes = sentiment_data.get("themes", [])
        return sentiment, themes
            
    except (json.JSONDecodeError, KeyError) as e:
        print(f"JSON Parsing Error or Key Missing: {type(e).__name__}: {e}")
        if isinstance(e, json.JSONDecodeError):
            print(f"Problematic JSON string: {sentiment_json_str}")
        return "Error: Could not analyze sentiment (JSON parse/key error)", []
    except Exception as e:
        print(f"General Error during OpenAI ChatCompletion: {type(e).__name__}: {e}")
        return "Error: Could not analyze sentiment (general error)", []

def analyze_comment(comment_text: str) -> tuple[str, str, str]:
    if not comment_text:
        return "Please enter a comment.", "N/A", "N/A"

    current_sentiment, current_themes = get_sentiment_and_themes_from_openai(comment_text)
    current_themes_str = ", ".join(current_themes) if current_themes else "No themes identified."

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

demo = gr.Interface(
    fn=analyze_comment,
    inputs=gr.Textbox(
        lines=5,
        placeholder='Enter a comment to analyze its emotion and find similar ones...',
        label="Comment Text"
    ),
    outputs=[
        gr.Textbox(label="Emotion of Typed Comment"),
        gr.Textbox(label="Themes of Typed Comment"),
        gr.Textbox(label="Similar Comments from Pinecone")
    ],
    title="🎯 Comment Emotion & Similarity Analyzer",
    description="Enter a comment to get its real-time emotion and themes. The app will also search for similar comments stored in Pinecone."
)

if __name__ == "__main__":
    demo.launch()
