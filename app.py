import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# -------------------------
# Load Data + Build FAISS Index
# -------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("tickets.csv")

    # Convert Created column to datetime
    df['Created'] = pd.to_datetime(df['Created'], errors='coerce')

    # Combine short + full description for embeddings
    df['text'] = df['Short description'].fillna('') + " " + df['Description'].fillna('')

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build embeddings
    embeddings = model.encode(df['text'].tolist(), batch_size=128, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return df, model, index

df, model, index = load_data()

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔎 Ticket Search App")
st.write("Search tickets by meaning + filter by state, assignment group, or date")

# Filters
states = ["All"] + sorted(df['State'].dropna().unique().tolist())
groups = ["All"] + sorted(df['Assignment group'].dropna().unique().tolist())

state_filter = st.selectbox("Filter by State", states)
group_filter = st.selectbox("Filter by Assignment Group", groups)
date_range = st.date_input("Filter by Date Range", [])

# Search box
query = st.text_input("Enter your query:", "")

if query:
    # Build query embedding
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    # Search FAISS
    scores, idx = index.search(q_emb, k=10)
    results = df.iloc[idx[0]].copy()
    results['score'] = scores[0]

    # Apply filters
    if state_filter != "All":
        results = results[results['State'] == state_filter]
    if group_filter != "All":
        results = results[results['Assignment group'] == group_filter]
    if len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        results = results[(results['Created'] >= start) & (results['Created'] <= end)]

    # Show results
    st.subheader("Top Matches")
    if results.empty:
        st.warning("No tickets match your query and filters.")
    else:
        for _, row in results.head(5).iterrows():
            st.markdown(f"""
            **Ticket ID:** {row['Number']}  
            **Created:** {row['Created'].date() if pd.notna(row['Created']) else 'N/A'}  
            **State:** {row['State']}  
            **Assignment Group:** {row['Assignment group']}  
            **Short Description:** {row['Short description']}  
            **Description:** {row['Description']}  
            **Score:** {row['score']:.2f}  
            ---
            """)
