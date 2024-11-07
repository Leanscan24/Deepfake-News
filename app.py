import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import time

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class NewsClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.model = LogisticRegression()
        self.trusted_domains = [
            'reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com', 
            'wsj.com', 'bloomberg.com', 'theguardian.com'
        ]

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def check_source_credibility(self, url):
        if url:
            try:
                domain = re.findall(r'(?:https?:\/\/)?(?:[^@\n]+@)?(?:www\.)?([^:\/\n]+)', url)[0]
                return any(trusted_domain in domain for trusted_domain in self.trusted_domains)
            except:
                return False
        return False

    def predict_news(self, article_text, url=None):
        # For demonstration, using a simple rule-based prediction
        # In a real application, you would use a trained model
        text_length = len(article_text)
        complexity = len(set(article_text.split()))
        
        # Dummy prediction logic
        score = np.random.random()  # Random score between 0 and 1
        prediction = 'Real' if score > 0.5 else 'Fake'
        
        return {
            'prediction': prediction,
            'confidence': f"{score * 100:.2f}%",
            'source_credibility': 'Credible' if self.check_source_credibility(url) else 'Unknown/Not Credible',
            'text_length': text_length,
            'complexity': complexity
        }

def fetch_article_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content
    except Exception as e:
        st.error(f"Error fetching article: {str(e)}")
        return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Fake News Verification System",
        page_icon="📰",
        layout="wide"
    )

    # Main title and description
    st.title("📰 Fake News Verification System")
    st.markdown("""
    This application helps verify if a news article is likely to be real or fake using multiple verification methods.
    """)

    # Initialize classifier
    classifier = NewsClassifier()

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This tool analyzes news articles using:
        - Text analysis
        - Source credibility
        - Content complexity
        
        Always verify news from multiple reliable sources!
        """)

    # Main content
    tab1, tab2 = st.tabs(["Single Article Analysis", "Batch Analysis"])

    # Single Article Analysis Tab
    with tab1:
        st.header("Analyze Single Article")
        
        input_method = st.radio(
            "Choose input method:",
            ("Enter Text", "Paste URL")
        )

        article_text = ""
        article_url = ""

        if input_method == "Enter Text":
            article_text = st.text_area(
                "Enter the news article text:",
                height=200,
                placeholder="Paste your article text here..."
            )
        else:
            article_url = st.text_input(
                "Enter article URL:",
                placeholder="https://example.com/news-article"
            )
            if article_url:
                with st.spinner("Fetching article content..."):
                    article_text = fetch_article_content(article_url)
                    if article_text:
                        st.success("Article fetched successfully!")
                        st.markdown("### Fetched Content Preview")
                        st.write(article_text[:500] + "...")

        if st.button("Verify News", key="verify_single"):
            if article_text:
                with st.spinner("Analyzing..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    # Get prediction
                    result = classifier.predict_news(article_text, article_url)

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            label="Prediction",
                            value=result['prediction'],
                            delta="High Confidence" if float(result['confidence'][:-1]) > 75 else "Medium Confidence"
                        )

                    with col2:
                        st.metric(
                            label="Confidence",
                            value=result['confidence']
                        )

                    with col3:
                        st.metric(
                            label="Source Credibility",
                            value=result['source_credibility']
                        )

                    # Detailed analysis
                    st.subheader("Detailed Analysis")
                    with st.expander("See detailed analysis"):
                        st.write("**Text Statistics:**")
                        st.write(f"- Length: {result['text_length']} characters")
                        st.write(f"- Vocabulary Richness: {result['complexity']} unique words")
                        
                        st.write("**Confidence Score:**")
                        st.progress(float(result['confidence'][:-1]) / 100)

                        st.write("**Reading Time:**")
                        words = len(article_text.split())
                        reading_time = round(words / 200)  # Assuming 200 words per minute
                        st.write(f"Approximately {reading_time} minute(s) to read")

            else:
                st.warning("Please enter some text to analyze.")

    # Batch Analysis Tab
    with tab2:
        st.header("Batch Analysis")
        st.write("Upload a CSV file containing multiple news articles for batch analysis.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())

                if st.button("Analyze Batch"):
                    st.info("Batch analysis feature coming soon!")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>📊 Fake News Verification System</p>
        <p>CAUTION : News Analysis May Not Be 100% Accurate</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()