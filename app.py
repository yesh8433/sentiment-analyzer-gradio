import streamlit as st
import os
from openai import OpenAI, RateLimitError



# Load API key – prefer Streamlit secrets, otherwise environment variable
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.warning(
        "❗️ Please set your OpenAI API key.\n\n"
        "• **Option 1:** In Streamlit Cloud → *Settings → Secrets* add `OPENAI_API_KEY`.\n"
        "• **Option 2:** Locally, set an environment variable:\n"
        "  `export OPENAI_API_KEY=your_key_here` (macOS / Linux)\n"
        "  or `set OPENAI_API_KEY=your_key_here` (Windows)."
    )
    st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(page_title="Fashion Caption Generator")
st.title("👗 Fashion Caption Generator")

style = st.selectbox("Choose a caption style", ["Classy", "Sassy", "Gen Z", "Poetic", "Minimalist"])
product = st.text_area("Describe your product (e.g. Red satin dress with ruffles)")

if st.button("Generate Instagram Caption"):
    with st.spinner("Generating..."):
        prompt = f"Write an Instagram caption in a {style.lower()} tone for the following fashion product: {product}. Include 3 trendy hashtags and make it catchy."
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100,
            )
            caption = response.choices[0].message.content
            st.subheader("✨ Your Caption:")
            st.success(caption)
        except RateLimitError:
            st.error(
                "🚫 **Rate limit or quota exceeded.**\n\n"
                "Check your OpenAI plan/usage or wait and try again."
            )
        except Exception as e:
            st.error(f"Unexpected error: {e}")