# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title="Reddit Anxiety Dashboard", layout="wide")

# ---------------------------------
# Title and Description
# ---------------------------------
st.title("🧠 Reddit Anxiety Dataset Dashboard")
st.write("""
This interactive dashboard allows you to explore and visualize the **Reddit Anxiety Dataset**.
You can view post distributions, label statistics, and word frequencies for different anxiety subtypes.
""")

# ---------------------------------
# File Uploader or Default Dataset
# ---------------------------------
uploaded_file = st.file_uploader("📂 Upload your labeled CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
else:
    st.info("No file uploaded. Using example dataset (reddit_anxiety_labeled.csv).")
    df = pd.read_csv("reddit_anxiety_labeled.csv")

# ---------------------------------
# Basic Information
# ---------------------------------
st.subheader("📋 Dataset Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head())

st.markdown("---")

# ---------------------------------
# Label Distribution
# ---------------------------------
st.subheader("🏷️ Anxiety Subtype Distribution")

if 'label' in df.columns:
    label_counts = df['label'].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(label_counts)
    with col2:
        fig, ax = plt.subplots()
        ax.pie(label_counts.values, labels=label_counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Label Proportion")
        st.pyplot(fig)
else:
    st.warning("No 'label' column found in dataset.")

st.markdown("---")

# ---------------------------------
# Interactive Filtering
# ---------------------------------
st.subheader("🎯 Filter Data by Label")

if 'label' in df.columns:
    label_options = df['label'].unique().tolist()
    selected_label = st.selectbox("Select an anxiety subtype:", label_options)

    filtered_df = df[df['label'] == selected_label]
    st.write(f"Showing {len(filtered_df)} posts for label: **{selected_label}**")
    st.dataframe(filtered_df.head())

    # Word count analysis
    st.subheader("📝 Word Count Distribution")
    filtered_df['text_length'] = filtered_df['text'].astype(str).apply(lambda x: len(x.split()))
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['text_length'], bins=30, kde=True, ax=ax, color='teal')
    ax.set_title(f"Word Count Distribution — {selected_label}")
    st.pyplot(fig)

    # Word cloud
    st.subheader("☁️ Word Cloud for Selected Label")
    text = " ".join(filtered_df['text'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
else:
    st.warning("Please make sure your dataset has a 'label' column.")

st.markdown("---")

# ---------------------------------
# Score Analysis (if available)
# ---------------------------------
if 'score' in df.columns:
    st.subheader("⭐ Post Engagement (Score) Analysis")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='label', y='score', palette="Set2", ax=ax)
    ax.set_title("Score Distribution by Label")
    st.pyplot(fig)
else:
    st.info("No 'score' column found — skipping engagement analysis.")

# ---------------------------------
# Summary
# ---------------------------------
st.markdown("---")
st.subheader("📊 Summary Insights")
st.write("""
- The dataset includes posts from anxiety-related subreddits labeled as **GAD, Panic, and Social Anxiety.**
- The label distribution chart highlights class balance.
- Word count and word cloud help understand linguistic expression for each subtype.
- If score data is present, you can analyze engagement differences across categories.
""")
