import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
from data_logic import create_simulated_data, apply_blending_strategy
# Import docx library for handling Word files (ensure it's installed: pip install python-docx)
import docx

# --- Configuration & State ---
st.set_page_config(
    page_title="Advanced Data Curation Dashboard",
    layout="wide", # Use wide layout for better space utilization
    initial_sidebar_state="expanded"
)

OPENROUTER_API_KEY = st.secrets["api_keys"]["openrouter"] 
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "meta-llama/llama-3-8b-instruct" 

def generate_llm_response(prompt_text, model=LLM_MODEL):
    """Sends a prompt to the OpenRouter API."""
    if OPENROUTER_API_KEY == "":
        return "❌ **API Key Error:** Please replace 'YOUR_OPENROUTER_API_KEY_HERE' in app.py with your OpenRouter key."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    system_prompt = "You are a specialized AI data analyst. Analyze the provided file metrics and data blending strategy. Be concise and focus on risks and effectiveness."

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.1, 
        "max_tokens": 512
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    except Exception as e:
        return f"❌ **API Connection Error:** Could not connect/process request: {e}"

# --- File Analysis Functions (SIMULATION/BASIC) ---

def analyze_uploaded_file(uploaded_file):
    """Provides simulated scores and basic file parsing."""
    file_type = uploaded_file.type
    
    # Simple Content Preview
    content_preview = f"File type: {file_type}. Size: {uploaded_file.size / 1024:.2f} KB."
    
    # Simulate Originality Score (Higher for images/docs, lower for CSV/web data)
    if 'image' in file_type or 'document' in file_type:
        originality_score = 90 + hash(uploaded_file.name) % 10 # High score
    else:
        originality_score = 40 + hash(uploaded_file.name) % 30 # Low/Medium score

    # Simulate Dataset Score (How valuable the data is)
    dataset_score = 75 + hash(uploaded_file.name) % 20

    # Read specific file types for display/manipulation
    if 'text/csv' in file_type:
        try:
            df = pd.read_csv(uploaded_file)
            content_preview += f"\nCSV Preview:\n{df.head().to_markdown()}"
        except Exception:
            content_preview += "\n(Could not parse CSV)"
            
    elif 'docx' in file_type:
        try:
            doc = docx.Document(uploaded_file)
            paragraphs = [p.text for p in doc.paragraphs if p.text]
            content_preview += f"\nDOCX Preview (First 2 lines):\n{paragraphs[0]} {paragraphs[1]}"
        except Exception:
            content_preview += "\n(Could not parse DOCX)"

    elif 'image' in file_type:
         st.image(uploaded_file, caption=uploaded_file.name, width=150)

    return originality_score, dataset_score, content_preview

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Data Blending Controls")
    
    # File Uploader Box (accepts multiple types)
    st.subheader("Data Input Box")
    uploaded_files = st.file_uploader(
        "Upload Custom Data (.csv, .docx, .png, .jpg)", 
        type=["csv", "docx", "png", "jpg"], 
        accept_multiple_files=True
    )
    
    st.markdown("---")
    st.subheader("Simulation Parameters")
    total_samples = st.slider("Total Simulated Samples", 
                              min_value=1000, max_value=10000, value=5000, step=100)
    anchor_ratio = st.slider("Human Anchor Ratio (%)", 
                             min_value=10, max_value=50, value=25, step=5) / 100
    quality_threshold = st.slider("Min AI Quality Threshold", 
                                  min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    st.markdown("---")
    run_button = st.button("🚀 Run Pipeline & Analyze")

# --- Main Content ---
st.title("🛡️ Advanced Data Curation & Model Collapse Dashboard")
st.markdown("Use the sidebar to upload external data and adjust simulation parameters.")

if uploaded_files:
    st.header("📂 Uploaded File Analysis")
    # Use columns to display multiple file analyses
    cols = st.columns(len(uploaded_files))
    
    for i, file in enumerate(uploaded_files):
        with cols[i]:
            score_o, score_d, preview = analyze_uploaded_file(file)
            st.success(f"**{file.name}**")
            st.metric("Originality Score (Simulated)", f"{score_o}%")
            st.metric("Dataset Value Score (Simulated)", f"{score_d}%")
            st.code(preview, language='markdown')
            
    st.markdown("---")

if run_button:
    # --- 1. Run Data Blending ---
    with st.spinner('Running Data Blending Simulation...'):
        raw_df = create_simulated_data(total_samples)
        final_df, anchor_set, high_quality_ai = apply_blending_strategy(
            raw_df, anchor_ratio, quality_threshold
        )

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("📊 Blending Results Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Initial Samples", f"{len(raw_df):,}")
            c2.metric("Anchor Retained", f"{len(anchor_set):,}")
            c3.metric("Final Training Size", f"{len(final_df):,}")

            human_percent = (len(anchor_set) / len(final_df)) * 100 if len(final_df) > 0 else 0
            ai_percent = (len(high_quality_ai) / len(final_df)) * 100 if len(final_df) > 0 else 0
            st.info(f"Final Blend Ratio (Human:AI): **{human_percent:.1f}% : {ai_percent:.1f}%**")

            st.subheader("Curation Logic")
            st.markdown(f"**Strategy:** Filtered {total_samples - len(final_df)} low-quality samples. Retained {len(anchor_set)} anchor samples.")
        
        with col_right:
            st.subheader("Data Quality Distribution Comparison")
            fig, ax = plt.subplots(figsize=(8, 4))
            raw_df['quality'].plot(kind='hist', ax=ax, alpha=0.5, bins=15, label='Initial Data Pool (Polluted)', density=True, color='skyblue')
            final_df['quality'].plot(kind='hist', ax=ax, alpha=0.7, bins=15, label='Final Training Set (Cured)', density=True, color='darkgreen')
            ax.axvline(x=quality_threshold, color='red', linestyle='--', label='AI Quality Threshold')
            ax.legend()
            ax.set_title("Quality Score Distribution Shift")
            ax.set_xlabel("Quality Score (0.0 to 1.0)")
            st.pyplot(fig)
    
    st.markdown("---")
    
    # --- 2. REAL LLM Analysis ---
    st.header("🧠 Real Llama 3 Analysis of Strategy")
    
    analysis_data = {
        "Initial Samples": len(raw_df),
        "Human Anchor Retained": len(anchor_set),
        "Filtered AI Content": len(high_quality_ai),
        "Final Blend Ratio (Human:AI)": f"{human_percent:.1f}% : {ai_percent:.1f}%",
        "Quality Threshold Used": quality_threshold
    }
    
    prompt = f"""
    A machine learning team performed data curation to prevent model collapse.
    Here are the statistics of the resulting training dataset:

    {json.dumps(analysis_data, indent=4)}

    Analyze these results. How effective was the filtering strategy and what is the primary role of the Human Anchor Set? Discuss the ethical risks of using a blend predominantly synthetic data.
    """

    st.subheader("Expert Analysis")
    with st.spinner("Calling Llama 3 8B API for Real Analysis..."):
        real_response = generate_llm_response(prompt)
        st.success(f"**Llama 3 8B Response:**\n\n{real_response}")