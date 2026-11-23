import streamlit as st
from esg_core import *

st.set_page_config(page_title="ESG Lens", layout="wide")
st.title("ESG Lens")
st.caption("Answers come only from the sustainability reports, with the page for each source.")


@st.cache_resource
def get_index():
    return load_index()


chunks, vectors, embed_model = get_index()
companies = sorted({c["company"] for c in chunks})

mode = st.sidebar.radio("Mode", ["Ask one company", "Compare all companies"])
if mode == "Ask one company":
    choice = st.sidebar.selectbox("Company", ["Any"] + [pretty(c) for c in companies])
    company = None if choice == "Any" else companies[[pretty(c) for c in companies].index(choice)]

question = st.chat_input("Ask a question about the reports")

if question:
    st.markdown(f"**Question:** {question}")

    if mode == "Ask one company":
        with st.spinner("Searching the reports..."):
            ans, hits = answer(question, chunks, vectors, embed_model, company=company)
        st.write(ans)
        with st.expander("Sources"):
            for h in hits:
                st.markdown(f"**{pretty(h['company'])} {h['year']}, page {h['page']}**")
                st.caption(h["text"][:400])
    else:
        cols = st.columns(len(companies))
        for col, comp in zip(cols, companies):
            with col:
                st.subheader(pretty(comp))
                with st.spinner("Searching..."):
                    ans, hits = answer(question, chunks, vectors, embed_model, company=comp)
                st.write(ans)
                with st.expander("Sources"):
                    for h in hits:
                        st.markdown(f"**{h['year']}, page {h['page']}**")
                        st.caption(h["text"][:300])
