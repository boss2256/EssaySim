# app.py
import os

# 👇 Ensure HF uses the offline model from cache
#os.environ["TRANSFORMERS_CACHE"] = ".cache"
#os.environ["TRANSFORMERS_OFFLINE"] = "1"

from utils.similarity import calculate_similarity, word_level_contribution
from docx import Document
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st



st.set_page_config(
    page_title="EssaySim",
    page_icon="🧠",
    layout="centered"
)
st.title("📝 EssaySimScore")
st.subheader("Check semantic similarity between two texts")

# Input toggle for stopword removal
remove_stopwords = st.checkbox("Remove common English stopwords (e.g., 'the', 'is')")

# Text inputs
col1, col2 = st.columns(2)
with col1:
    text1 = st.text_area("Enter the first sentence or paragraph", height=200)
with col2:
    text2 = st.text_area("Enter the second sentence or paragraph", height=200)

# File upload option
st.markdown("### 📎 Or upload two files to compare (.txt or .docx)")
file1 = st.file_uploader("Upload first file", type=["txt", "docx"], key="file1")
file2 = st.file_uploader("Upload second file", type=["txt", "docx"], key="file2")


# Extract text from uploaded files
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

# Show extracted content if files are uploaded
if file1 and file2:
    st.markdown("### 📄 File Content Preview")
    p1, p2 = st.columns(2)
    with p1:
        st.code(text1[:1000] + ("..." if len(text1) > 1000 else ""), language='text')
    with p2:
        st.code(text2[:1000] + ("..." if len(text2) > 1000 else ""), language='text')

# Trigger similarity
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

        # Word-by-word with progress indication
        st.markdown("### 🔍 Word-by-word contribution analysis")
        progress_container = st.empty()
        progress_bar = progress_container.progress(0, "Preparing for analysis...")

        # Add small description about the process
        info_container = st.empty()
        info_container.info(
            "Analyzing word pairs for semantic similarity. This uses AI embeddings to find matching concepts.")

        # Process the word-level contributions with progress updates
        with st.spinner(""):
            # Set a callback or use the progress bar directly
            progress_bar.progress(25, "Tokenizing text...")
            contributions = word_level_contribution(text1, text2, remove_stopwords=remove_stopwords)
            progress_bar.progress(100, "Analysis complete!")

        # Clear the progress indicators once done
        progress_container.empty()
        info_container.empty()

        if contributions:
            # Display a sample of contributions if there are many
            display_limit = 50
            display_contributions = contributions[:display_limit]

            for w1, w2, s in display_contributions:
                # Set better contrast colors based on similarity score
                if s > 0.7:  # High similarity - green
                    bg_color = "#d4edda"
                    text_color = "#155724"
                    border_color = "#c3e6cb"
                elif s > 0.4:  # Medium similarity - yellow/orange
                    bg_color = "#fff3cd"
                    text_color = "#856404"
                    border_color = "#ffeeba"
                else:  # Low similarity - red
                    bg_color = "#f8d7da"
                    text_color = "#721c24"
                    border_color = "#f5c6cb"

                # Create a more visually appealing card with better contrast
                st.markdown(f"""
                <div style='
                    background-color:{bg_color}; 
                    color:{text_color}; 
                    padding:10px; 
                    border-radius:5px; 
                    margin-bottom:8px;
                    border: 1px solid {border_color};
                    font-weight: 500;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                '>
                    <div style='font-family: monospace; font-size: 15px;'>
                        <span style='background-color: rgba(255,255,255,0.6); padding: 3px 6px; border-radius: 3px;'>{w1}</span>
                        <span style='margin: 0 10px;'>→</span>
                        <span style='background-color: rgba(255,255,255,0.6); padding: 3px 6px; border-radius: 3px;'>{w2}</span>
                    </div>
                    <div style='
                        background-color: rgba(255,255,255,0.8); 
                        color: {text_color}; 
                        padding: 2px 8px; 
                        border-radius: 10px; 
                        font-size: 14px;
                        min-width: 80px;
                        text-align: center;
                    '>
                        {s:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if len(contributions) > display_limit:
                st.info(
                    f"Showing {display_limit} out of {len(contributions)} word matches. Download the report for the full analysis.")

            # Update this section in your app.py file

            # Replace the existing heatmap code with this improved version
            try:
                # Create heatmap with a limited number of words to avoid visual clutter
                st.markdown("### 📊 Heatmap of Similarity Scores")

                # Take a subset if there are too many words
                heatmap_limit = min(20, len(contributions))
                heatmap_sample = contributions[:heatmap_limit]

                # Create the DataFrame for the heatmap
                heatmap_data = pd.DataFrame(heatmap_sample, columns=["Word 1", "Word 2", "Score"])

                # Check for and handle duplicate word pairs
                # Create a composite key to identify unique word pairs
                heatmap_data['word_pair'] = heatmap_data['Word 1'] + '___' + heatmap_data['Word 2']

                # Keep only the first occurrence of each word pair (highest score)
                heatmap_data = heatmap_data.drop_duplicates(subset=['word_pair'])

                # If there are still duplicate words, make them unique by adding a suffix
                word1_counts = {}
                for i, word in enumerate(heatmap_data['Word 1']):
                    if word in word1_counts:
                        word1_counts[word] += 1
                        heatmap_data.loc[heatmap_data.index[i], 'Word 1'] = f"{word}_{word1_counts[word]}"
                    else:
                        word1_counts[word] = 0

                word2_counts = {}
                for i, word in enumerate(heatmap_data['Word 2']):
                    if word in word2_counts:
                        word2_counts[word] += 1
                        heatmap_data.loc[heatmap_data.index[i], 'Word 2'] = f"{word}_{word2_counts[word]}"
                    else:
                        word2_counts[word] = 0

                # Only create pivot if we have enough unique words after processing
                unique_words1 = len(set(heatmap_data["Word 1"]))
                unique_words2 = len(set(heatmap_data["Word 2"]))

                if unique_words1 >= 2 and unique_words2 >= 2:
                    # Create the pivot table for the heatmap
                    heatmap_pivot = heatmap_data.pivot(index="Word 1", columns="Word 2", values="Score")

                    # Create and display the heatmap
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, linewidths=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Not enough unique word pairs for a meaningful heatmap visualization.")

            except Exception as e:
                st.warning(f"Could not create heatmap: {str(e)}")
                # Provide a detailed alternative visualization
                fig, ax = plt.subplots(figsize=(12, 8))

                # Sort by score for better visualization
                sorted_data = heatmap_data.sort_values(by="Score", ascending=False)

                # Create a more informative barplot that shows word pairs
                bars = sns.barplot(x="Word 1", y="Score", data=sorted_data, ax=ax)

                # Add word 2 labels above each bar
                for i, bar in enumerate(bars.patches):
                    if i < len(sorted_data):
                        word2 = sorted_data.iloc[i]["Word 2"]
                        score = sorted_data.iloc[i]["Score"]
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.02,
                            f"→ {word2}",
                            ha='center',
                            va='bottom',
                            rotation=45,
                            fontsize=10,
                            color='black'
                        )

                plt.xticks(rotation=45, ha='right')
                plt.title("Word Similarity Scores (Word 1 → Word 2)")
                plt.tight_layout()
                st.pyplot(fig)

            # Downloadable Report
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
        else:
            st.warning("_No significant word-level contributions detected. Try using shorter, more focused texts._")
else:
    st.warning("Please input or upload two texts to compare.")