import os
import json
import gradio as gr
import pinecone

# 1) Initialize Pinecone from environment variables
pinecone.init(
    api_key=os.getenv("pcsk_2indG_7bikinqpKq6rseXDUYfWgbrNvjFEMvmSXt96tT6HQxejv76tpacdmm4N7jVoreK"),
    environment=os.getenv("aped-4627-b74a.pinecone.io")
)
index = pinecone.Index(os.getenv("fb-comments-q4h4rly.svc"))
def analyze(raw_query_json: str) -> str:
    try:
        payload = json.loads(raw_query_json)
        resp = index.query(**payload)
        return resp["matches"][0]["metadata"]["sentiment"]
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
    description="Paste in a Pinecone query JSON and get back the top comment’s sentiment.")

if __name__ == "__main__":
    demo.launch()
