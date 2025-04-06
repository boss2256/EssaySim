# app.py (transformer-only version with heatmap)
import os
from utils.similarity import calculate_similarity, word_level_contribution
from docx import Document
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="EssaySimScore",
    page_icon="🧠",
    layout="centered"
)

st.title("📝 EssaySimScore")
st.subheader("Check semantic similarity between two texts")

remove_stopwords = st.checkbox("Remove common English stopwords (e.g., 'the', 'is')")

col1, col2 = st.columns(2)
with col1:
    text1 = st.text_area("Enter the first sentence or paragraph", height=200)
with col2:
    text2 = st.text_area("Enter the second sentence or paragraph", height=200)

st.markdown("### 📎 Or upload two files to compare (.txt or .docx)")
file1 = st.file_uploader("Upload first file", type=["txt", "docx"], key="file1")
file2 = st.file_uploader("Upload second file", type=["txt", "docx"], key="file2")

def read_file(file):
    if file is None:
        return ""
    ext = os.path.splitext(file.name)[-1].lower()
    if ext == ".txt":
        return file.read().decode("utf-8")
    elif ext == ".docx":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

if file1 and file2:
    text1 = read_file(file1)
    text2 = read_file(file2)

if file1 and file2:
    st.markdown("### 📄 File Content Preview")
    p1, p2 = st.columns(2)
    with p1:
        st.code(text1[:1000] + ("..." if len(text1) > 1000 else ""), language='text')
    with p2:
        st.code(text2[:1000] + ("..." if len(text2) > 1000 else ""), language='text')

if text1.strip() and text2.strip():
    with st.spinner("Calculating similarity scores..."):
        st.markdown("---")
        score = calculate_similarity(text1, text2)
        st.markdown(f"### 🔎 Similarity Score: `{score:.4f}`")

        if score > 0.7:
            st.success("✅ Very similar (duplicate meaning)")
        elif score > 0.4:
            st.info("🤝 Related but not identical")
        else:
            st.error("❌ Not semantically similar")

        st.markdown("### 🔍 Word-by-word contribution analysis")
        progress_container = st.empty()
        progress_bar = progress_container.progress(0, "Preparing for analysis...")
        info_container = st.empty()
        info_container.info("Analyzing word pairs for semantic similarity. This uses AI embeddings to find matching concepts.")

        with st.spinner(""):
            progress_bar.progress(25, "Tokenizing text...")
            contributions = word_level_contribution(text1, text2, remove_stopwords=remove_stopwords)
            progress_bar.progress(100, "Analysis complete!")

        progress_container.empty()
        info_container.empty()

        if contributions:
            display_limit = 50
            display_contributions = contributions[:display_limit]

            for w1, w2, s in display_contributions:
                if s > 0.7:
                    bg_color = "#d4edda"
                    text_color = "#155724"
                    border_color = "#c3e6cb"
                elif s > 0.4:
                    bg_color = "#fff3cd"
                    text_color = "#856404"
                    border_color = "#ffeeba"
                else:
                    bg_color = "#f8d7da"
                    text_color = "#721c24"
                    border_color = "#f5c6cb"

                st.markdown(f"""
                <div style='background-color:{bg_color};color:{text_color};padding:10px;border-radius:5px;margin-bottom:8px;border: 1px solid {border_color};font-weight: 500;display: flex;align-items: center;justify-content: space-between;'>
                    <div style='font-family: monospace; font-size: 15px;'>
                        <span style='background-color: rgba(255,255,255,0.6); padding: 3px 6px; border-radius: 3px;'>{w1}</span>
                        <span style='margin: 0 10px;'>→</span>
                        <span style='background-color: rgba(255,255,255,0.6); padding: 3px 6px; border-radius: 3px;'>{w2}</span>
                    </div>
                    <div style='background-color: rgba(255,255,255,0.8); color: {text_color}; padding: 2px 8px; border-radius: 10px; font-size: 14px; min-width: 80px; text-align: center;'>
                        {s:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if len(contributions) > display_limit:
                st.info(f"Showing {display_limit} out of {len(contributions)} word matches. Download the report for the full analysis.")

            st.markdown("### 📥 Download Report")
            report = f"""
EssaySim Report

Text 1:
{text1}

Text 2:
{text2}

Similarity Score: {score:.4f}

Match Level: {'Very similar' if score > 0.7 else 'Related but not identical' if score > 0.4 else 'Not semantically similar'}

Word-by-word Contributions:
"""
            for w1, w2, s in contributions:
                report += f"- {w1} → {w2} (score: {s:.4f})\n"

            buffer = BytesIO()
            buffer.write(report.encode())
            buffer.seek(0)
            st.download_button("📄 Download .txt Report", buffer, file_name="similarity_report.txt", mime="text/plain")

            # ✅ Heatmap visualization block
            st.markdown("### 📊 Heatmap of Similarity Scores")
            try:
                heatmap_limit = min(20, len(contributions))
                heatmap_data = pd.DataFrame(contributions[:heatmap_limit], columns=["Word 1", "Word 2", "Score"])

                heatmap_data['Word 1'] = heatmap_data['Word 1'].apply(lambda x, c={}: f"{x}_{c.setdefault(x, 0)}" if x in c else x)
                heatmap_data['Word 2'] = heatmap_data['Word 2'].apply(lambda x, c={}: f"{x}_{c.setdefault(x, 0)}" if x in c else x)

                pivot = heatmap_data.pivot(index="Word 1", columns="Word 2", values="Score")

                if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Not enough unique word pairs for a meaningful heatmap.")
            except Exception as e:
                st.warning(f"Heatmap failed to render: {e}")

        else:
            st.warning("_No significant word-level contributions detected. Try using shorter, more focused texts._")
else:
    st.warning("Please input or upload two texts to compare.")