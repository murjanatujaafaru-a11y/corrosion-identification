import streamlit as st
import torch
from transformers import pipeline

# --- Page Configuration ---
st.set_page_config(page_title="Corrosion Risk NLP", page_icon="📝")

# --- Title & Branding ---
st.title("📝 Pipeline Integrity: NLP Report Analyzer")
st.markdown("""
**Project Objective:** Utilizing Natural Language Processing to analyze maintenance logs and detect the urgency of corrosion-related reports.
""")
st.write("---")

# --- Load NLP Model ---
@st.cache_resource
def load_nlp_pipeline():
    # This model detects 'emotions' which we map to 'Urgency Levels'
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

classifier = load_nlp_pipeline()

# --- User Interface ---
st.subheader("Enter Technician Maintenance Log")
user_text = st.text_area("Example: 'Found severe pitting on the offshore joint. Requires immediate attention!'", 
                         placeholder="Paste the report text here...")

if st.button("Analyze Report Urgency"):
    if user_text.strip() != "":
        # Prediction
        results = classifier(user_text)
        label = results[0]['label']
        score = results[0]['score'] * 100

        # --- Mapping Emotion to Corrosion Urgency ---
        # We relate NLP 'emotions' to technical 'integrity risks'
        urgency_map = {
            "fear": "🚨 HIGH URGENCY: Report suggests safety concerns.",
            "anger": "⚠️ CRITICAL: Report indicates frustration with asset failure.",
            "sadness": "📉 CONCERN: Report indicates structural degradation.",
            "joy": "✅ STABLE: Report indicates successful maintenance.",
            "surprise": "🔍 INVESTIGATE: Unusual findings reported.",
            "love": "✅ OPTIMAL: Asset performing well."
        }

        status = urgency_map.get(label, "Checking integrity...")

        # --- Display Results ---
        st.info(f"**NLP Analysis:** The tone of this report is categorized as **{label.upper()}**.")
        st.subheader(status)
        st.write(f"**System Confidence:** {score:.2f}%")
        
        # Actionable advice for Seplat/NCDMB context
        if label in ['fear', 'anger', 'sadness']:
            st.error("RECOMMENDATION: Schedule an immediate physical inspection (Computer Vision check).")
        else:
            st.success("RECOMMENDATION: Continue with standard monitoring schedule.")
    else:
        st.info("Please enter the text of a maintenance log to analyze.")