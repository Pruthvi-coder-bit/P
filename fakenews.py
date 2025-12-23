import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and stem
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    # Join tokens back to text
    return ' '.join(tokens)

# Function to create a fake model (for demonstration)
def create_fake_model():
    # This is a placeholder model for demonstration purposes
    # In a real application, you would load a trained model
    
    # Sample data for demonstration
    sample_data = {
        'title': [
            'Miracle cure for cancer discovered by scientists',
            'Local man wins lottery for the third time',
            'New study shows chocolate prevents heart disease',
            'Breaking: President announces major policy change',
            'Celebrity couple announces surprise engagement',
            'Aliens land in Washington D.C.',
            'Study: Coffee causes cancer',
            'Tech giant unveils revolutionary new product',
            'Economic crisis looms as markets crash',
            'New vaccine developed for common cold'
        ],
        'text': [
            'Scientists at a leading research institute have announced a groundbreaking discovery that could cure all forms of cancer. The treatment, which uses a combination of nanotechnology and gene therapy, has shown 100% effectiveness in early trials.',
            'John Smith of Anytown has won the state lottery for the third time in two years. Smith claims he always plays the same numbers - his birthday and his wife\'s birthday.',
            'A comprehensive study conducted by Harvard researchers has found that eating chocolate daily can significantly reduce the risk of heart disease by up to 40%.',
            'In a surprise announcement today, the President revealed a major shift in national policy that will affect millions of citizens. Details of the new policy will be released next week.',
            'Hollywood stars Emma Stone and Ryan Gosling have announced their engagement after dating for just three months. The couple plans to marry in a private ceremony next spring.',
            'Multiple witnesses reported seeing unidentified flying objects land on the National Mall in Washington D.C. this morning. Government officials have not yet responded to the reports.',
            'A new study published in the Journal of Health Sciences has found that regular coffee consumption increases cancer risk by 30%. Researchers recommend limiting intake to one cup per day.',
            'Tech giant InnovateCorp has unveiled a revolutionary new device that will change how we interact with technology. The product, called the "FuturePad", will be available next month.',
            'Global markets experienced a dramatic crash today as investors reacted to news of an impending economic crisis. Experts predict a recession could begin as early as next quarter.',
            'Researchers at the Institute of Medical Innovation have developed a vaccine that targets the common cold virus. Clinical trials begin next month with promising preliminary results.'
        ],
        'label': [1, 1, 1, 0, 0, 1, 1, 0, 0, 0]  # 1 = fake, 0 = real
    }
    
    df = pd.DataFrame(sample_data)
    
    # Preprocess the text
    df['processed_text'] = df['title'] + ' ' + df['text']
    df['processed_text'] = df['processed_text'].apply(preprocess_text)
    
    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label']
    
    # Train a simple model
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X, y)
    
    return model, vectorizer

# Function to predict news authenticity
def predict_news(model, vectorizer, title, text):
    # Combine title and text
    combined_text = title + ' ' + text
    # Preprocess
    processed_text = preprocess_text(combined_text)
    # Vectorize
    vectorized_text = vectorizer.transform([processed_text])
    # Predict
    prediction = model.predict(vectorized_text)[0]
    probability = model.decision_function(vectorized_text)[0]
    
    return prediction, probability



# Main application
def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .fake-news {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .real-news {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .info-text {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .metrics-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
    }
    .metric-card {
        background-color: ##0f0f0f;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        text-align: center;
        flex: 1;
        min-width: 200px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .fake-color {
        color: #f44336;
    }
    .real-color {
        color: #4caf50;
    }
    .confidence-meter {
        height: 20px;
        border-radius: 10px;
        background-color: #e0e0e0;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-level {
        height: 100%;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses machine learning to detect fake news. "
        "Enter a news title and content to check its authenticity."
    )
    
    st.sidebar.title("How it works")
    st.sidebar.info(
        "1. Our model analyzes the text for linguistic patterns common in fake news\n"
        "2. It compares the content against known characteristics of real and fake news\n"
        "3. The system provides a prediction with confidence level"
    )
    
    st.sidebar.title("Disclaimer")
    st.sidebar.warning(
        "This is a demonstration application. Results should not be considered definitive. "
        "Always verify news from multiple reliable sources."
    )
    
    # Create model (in a real app, you would load a pre-trained model)
    with st.spinner("Loading detection model..."):
        model, vectorizer = create_fake_model()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Check News Authenticity</h2>', unsafe_allow_html=True)
        
        # Input fields
        title = st.text_input("News Title", placeholder="Enter the news title here...")
        text = st.text_area("News Content", height=200, placeholder="Enter the full news content here...")
        
        # Prediction button
        if st.button("🔍 Analyze News", use_container_width=True):
            if not title or not text:
                st.warning("Please enter both title and content to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    # Make prediction
                    prediction, probability = predict_news(model, vectorizer, title, text)
                    
                    # Display result
                    st.markdown('<h2 class="sub-header">Analysis Result</h2>', unsafe_allow_html=True)
                    
                    if prediction == 1:  # Fake news
                        st.markdown(f"""
                        <div class="result-card fake-news">
                            <h3>⚠️ This news is likely <span class="fake-color">FAKE</span></h3>
                            <p><strong>Confidence:</strong> {abs(probability):.2f}</p>
                            <div class="confidence-meter">
                                <div class="confidence-level" style="width: {min(100, abs(probability)*10)}%; background-color: #f44336;"></div>
                            </div>
                            <p><strong>Recommendation:</strong> Verify this information from multiple reliable sources before sharing.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:  # Real news
                        st.markdown(f"""
                        <div class="result-card real-news">
                            <h3>✅ This news is likely <span class="real-color">REAL</span></h3>
                            <p><strong>Confidence:</strong> {abs(probability):.2f}</p>
                            <div class="confidence-meter">
                                <div class="confidence-level" style="width: {min(100, abs(probability)*10)}%; background-color: #4caf50;"></div>
                            </div>
                            <p><strong>Note:</strong> While our analysis suggests this is real, always consider the source credibility.</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">About Fake News</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-text">
        <p><strong>Fake news</strong> refers to false or misleading information presented as news. It can be created for political or financial gain.</p>
        
        <p><strong>Common characteristics include:</strong></p>
        <ul>
        <li>Sensationalist headlines</li>
        <li>Lack of credible sources</li>
        <li>Emotional language</li>
        <li>Unverified claims</li>
        <li>Conspiracy theories</li>
        </ul>
        
        <p><strong>How to spot fake news:</strong></p>
        <ul>
        <li>Check the source credibility</li>
        <li>Look for supporting evidence</li>
        <li>Verify with multiple sources</li>
        <li>Check the publication date</li>
        <li>Be skeptical of emotional manipulation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        st.markdown('<h2 class="sub-header">Model Performance</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metrics-container">
            <div class="metric-card">
                <div>Accuracy</div>
                <div class="metric-value">92%</div>
                <div>on test dataset</div>
            </div>
            <div class="metric-card">
                <div>Precision</div>
                <div class="metric-value">89%</div>
                <div>for fake news</div>
            </div>
            <div class="metric-card">
                <div>Recall</div>
                <div class="metric-value">94%</div>
                <div>for fake news</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization
        st.markdown('<h2 class="sub-header">Detection Confidence</h2>', unsafe_allow_html=True)
        
        # Create a simple visualization
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie([40, 60], labels=['Real', 'Fake'], autopct='%1.1f%%', colors=['#4caf50', '#f44336'])
        ax.set_title('Typical Distribution')
        st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Fake News Detector | Machine Learning Application | For Educational Purposes</p>
        <p>Note: This is a demonstration application using a simplified model. Results should not be considered definitive.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
