import streamlit as st
from summarize import *
from transformers import pipeline

hub_model_id = "shivaniNK8/t5-small-finetuned-cnn-news"

summarizer = pipeline("summarization", model=hub_model_id, framework="pt", use_fast=False)

st.title("Briefly")
st.write("### A News Summarizer\nShare a news article and receive a summary in seconds")

st.sidebar.header('Select Summary Parameters')
with st.sidebar.form("input_form"):
    st.write('Select Summary Length for Extractive Summary')
    max_sentences = st.slider('Summary Length', 1, 10, step=1, value=3)
    st.write('Select Word Limits for Abstractive Summary')
    max_words = st.slider('Max Words', 50, 500, step=10, value=200)
    min_words = st.slider('Min Words', 10, 450, step=10, value=100)

    submit_button = st.form_submit_button("Summarize")

article = st.text_area("Enter article text here:", value="Paste the article text here", height=300)

news_summarizer = NewsSummarization()

if submit_button:
    st.write("## Extractive Summary")
    extractive_summary = news_summarizer.extractive_summary(article, num_sentences=max_sentences)
    st.write(extractive_summary)

    summary = summarizer(article, max_length=max_words, min_length=min_words, do_sample=False)
    abstractive_summary = summary[0]['summary_text']
    st.write("## Abstractive Summary")
    st.write(abstractive_summary)
