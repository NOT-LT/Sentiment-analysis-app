import streamlit as st
from textblob import TextBlob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="wide", page_title="Sentiment Analysis Web App")

# Initialize session state
if 'analyzed_texts' not in st.session_state:
    st.session_state['analyzed_texts'] = []
if 'current_text' not in st.session_state:
    st.session_state['current_text'] = ''

def sentiment_to_dataframe(sentiment):
    data = {
        'Polarity': sentiment.polarity,
        'Subjectivity': sentiment.subjectivity
    }
    return pd.DataFrame(data.items(), columns=['Metric', 'Value'])

def analyze_sentiment(text):
    textblob_sentiment = TextBlob(text).sentiment
    analyzer = SentimentIntensityAnalyzer()
    vader_sentiment = analyzer.polarity_scores(text)

    combined_polarity = (textblob_sentiment.polarity + vader_sentiment['compound']) / 2
    combined_subjectivity = textblob_sentiment.subjectivity

    st.session_state['current_text'] = text  # Update session state with current text
    st.session_state['analyzed_texts'].append(text)  # Add text to analyzed texts

    return combined_polarity, combined_subjectivity

def main():
    st.title("Sentiment Analysis Web App")

    with st.sidebar:
        st.header("Recent Texts Analyzed")
        if st.session_state['analyzed_texts']:
            for idx, text in enumerate(reversed(st.session_state['analyzed_texts'][-5:]), 1):
                truncated_text = text if len(text) <= 40 else text[:40] + '...'
                if st.button(truncated_text, key=f"history_{idx}", help="Click to load this text"):
                    st.session_state['current_text'] = text
                    st.experimental_rerun()
        else:
            st.write("No texts analyzed yet.")

    with st.form(key='sentimentAnalysisForm'):
        user_input = st.text_area("Enter Text Here", height=150, placeholder="Type or paste text here...", value=st.session_state.get('current_text', ''))
        uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])
        analyze_button = st.form_submit_button(label='Analyze', type="primary")

    if uploaded_file is not None:
        user_input = uploaded_file.read().decode('utf-8')
        st.session_state['current_text'] = user_input

    if analyze_button:
        if user_input:
            col1, col2 = st.columns([3, 2])

            with col1:
                polarity, subjectivity = analyze_sentiment(user_input)

                if polarity > 0:
                    st.markdown("<h5 style='color: lightgreen;'>Positive 😊</h5>", unsafe_allow_html=True)
                elif polarity < 0:
                    st.markdown("<h5 style='color: red;'>Negative 😠</h5>", unsafe_allow_html=True)
                else:
                    st.markdown("<h5 style='color: gray;'>Neutral 😐</h5>", unsafe_allow_html=True)

                st.markdown(f"<h6>Polarity: {polarity}</h6>", unsafe_allow_html=True)
                st.markdown(f"<h6>Subjectivity: {subjectivity}</h6>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()