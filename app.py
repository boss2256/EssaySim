# app.py (transformer-only version with heatmap)
import os
from utils.similarity import calculate_similarity, word_level_contribution
from docx import Document
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Prevent HuggingFace from downloading models online
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = './sentence-transformers'

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

            # app.py (Modified Heatmap Block)

            # ✅ Heatmap visualization block
            st.markdown("### 📊 Heatmap of Similarity Scores")
            try:
                # Limit the number of pairs for the heatmap for readability
                heatmap_limit = min(20, len(contributions))
                if heatmap_limit < 2:  # Need at least 2 pairs for a meaningful pivot
                    st.info("Not enough contribution pairs (need at least 2) for a heatmap.")
                else:
                    heatmap_data = pd.DataFrame(contributions[:heatmap_limit], columns=["Word 1", "Word 2", "Score"])

                    # --- CHANGE START ---
                    # Use pivot_table which can handle duplicate index/column entries
                    # 'aggfunc='first'' takes the score from the first occurrence if duplicates exist
                    # (since the list is sorted by score descending, this is often the highest score for that pair)
                    # You could also use 'mean' or 'max' if preferred.
                    pivot_df = heatmap_data.pivot_table(index="Word 1", columns="Word 2", values="Score",
                                                        aggfunc='first')
                    # --- CHANGE END ---

                    # Ensure the pivoted table has enough dimensions for a heatmap
                    if pivot_df.shape[0] >= 1 and pivot_df.shape[1] >= 1:  # Allow 1xN or Nx1 heatmaps too
                        # Adjust figsize dynamically? Maybe later. Start with fixed.
                        fig_height = max(5, pivot_df.shape[0] * 0.5)
                        fig_width = max(6, pivot_df.shape[1] * 0.6)
                        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

                        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, linewidths=.5)
                        plt.xticks(rotation=45, ha='right')  # Improve label readability
                        plt.yticks(rotation=0)
                        plt.tight_layout()  # Adjust layout to prevent labels overlapping
                        st.pyplot(fig)
                    else:
                        st.info(
                            f"Could not create a meaningful heatmap. Pivoted shape: {pivot_df.shape}. Need at least 1 row and 1 column after pivoting.")

            except Exception as e:
                # Print the specific error to the console/log for debugging
                print(f"Heatmap generation error: {e}")
                # Provide a more user-friendly message in the app
                st.warning(f"Could not generate heatmap. Error: {e}")

        else:
            st.warning("_No significant word-level contributions detected. Try using shorter, more focused texts._")
else:
    st.warning("Please input or upload two texts to compare.")