import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import re
import string
import random

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
)

def load_model():
    if os.path.exists('model/model.pkl') and os.path.exists('model/vectorizer.pkl'):
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('model/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load the preprocessed examples if available
        preprocessed_examples = {}
        if os.path.exists('model/preprocessed_examples.pkl'):
            with open('model/preprocessed_examples.pkl', 'rb') as f:
                preprocessed_examples = pickle.load(f)
        
        return model, vectorizer, preprocessed_examples
    else:
        return None, None, {}

def preprocess_text_simple(text):
    """
    A simplified version of preprocessing that doesn't use NLTK
    (for demo purposes only - real preprocessing was done during training)
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove whitespaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    else:
        return ""

def load_real_dataset(dataset_path=None):
    """
    Load sample data from the actual dataset used for training
    """
    # Default dataset path if not provided
    if dataset_path is None:
        dataset_path = r"News_dataset"  # Using the path from your train script

    try:
        # Load true and fake news datasets
        true_csv = os.path.join(dataset_path, 'True.csv')
        fake_csv = os.path.join(dataset_path, 'Fake.csv')
        
        true_df = pd.read_csv(true_csv)
        false_df = pd.read_csv(fake_csv)
        
        # Add labels
        true_df['label'] = 1  # Real
        false_df['label'] = 0  # Fake
        
        # Combine datasets
        df = pd.concat([true_df, false_df], ignore_index=True)
        
        # Determine which text column to use
        if 'text' in df.columns:
            df['content'] = df['text']
        elif 'title' in df.columns:
            df['content'] = df['title']
        else:
            raise ValueError("CSV files must contain either a 'text' or 'title' column")
        
        # Get title column if it exists
        if 'title' in df.columns:
            return df[['title', 'content', 'label']]
        else:
            # If no title column, use the first 50 chars of content as a title
            df['title'] = df['content'].apply(lambda x: str(x)[:50] + '...' if len(str(x)) > 50 else str(x))
            return df[['title', 'content', 'label']]
    
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Return the dummy dataset as fallback
        return create_sample_dataset()

def create_sample_dataset():
    """Fallback sample dataset in case the real one can't be loaded"""
    # Creating a sample dataset for fake news detection display
    fake_news = [
        {"title": "Shocking discovery: Water found to be harmful to humans", 
         "content": "Scientists have discovered that water is actually harmful to humans in the long term. This groundbreaking research suggests everyone should stop drinking water immediately.", 
         "label": 0},
        {"title": "Government secretly implanting microchips through vaccines", 
         "content": "Anonymous sources confirm that the government is using vaccines to implant tracking microchips into citizens. These microchips allow them to monitor your every move.", 
         "label": 0},
        {"title": "New study links chocolate consumption to immortality", 
         "content": "A controversial new study claims that people who eat chocolate daily may live forever. Scientists are baffled by these findings that challenge everything we know about aging.", 
         "label": 0},
        {"title": "Study shows coffee may reduce risk of certain diseases", 
         "content": "Recent research published in the Journal of Medical Science indicates that moderate coffee consumption may be associated with a reduced risk of liver disease and certain types of cancer.", 
         "label": 1},
        {"title": "New species of deep-sea fish discovered", 
         "content": "Marine biologists have identified a previously unknown species of fish living at depths of over 3,000 meters in the Pacific Ocean. The discovery highlights how much remains unknown about deep ocean ecosystems.", 
         "label": 1},
        {"title": "Local community opens new public library", 
         "content": "A new public library opened in the downtown area yesterday after three years of construction. The facility includes meeting rooms, computer labs, and a children's reading section.", 
         "label": 1}
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(fake_news)
    return df

# Streamlit UI components
def main():
    # Load the trained model, vectorizer, and preprocessed examples
    model, vectorizer, preprocessed_examples = load_model()
    
    if model is None or vectorizer is None:
        st.error("""
        No trained model found! 
        
        Please run the `train_model.py` script first to create the model files.
        
        ```
        python train_model.py
        ```
        """)
        st.stop()
    
    # Let's also try to load the real dataset
    dataset = load_real_dataset()
    
    st.title("📰 Fake News Detection System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Detection", "Dataset Examples", "About"])
    
    if page == "Home":
        show_home_page()
    elif page == "Detection":
        show_detection_page(model, vectorizer, preprocessed_examples)
    elif page == "Dataset Examples":
        show_examples_page(model, vectorizer, preprocessed_examples, dataset)
    elif page == "About":
        show_about_page()

def show_home_page():
    st.header("Welcome to the Fake News Detection System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is Fake News?
        
        Fake news refers to false or misleading information presented as legitimate news. 
        It's typically created to influence public opinion, generate clicks, or spread misinformation.
        
        ### How Does Our System Work?
        
        Our system uses machine learning to analyze text content and determine the likelihood
        that a news article is fake or legitimate. The system has been trained on a dataset of 
        labeled news articles to learn patterns that distinguish fake news from real news.
        
        ### Get Started
        
        - Use the **Detection** page to analyze news articles
        - See some **Dataset Examples** from our training data
        - Learn more about the system in the **About** section
        """)
    
    with col2:
        st.image("https://aipromptopus.com/wp-content/uploads/2024/01/OIG.aSnXm20oN14T40-1.jpg", 
                 caption="Detecting misinformation")

def show_detection_page(model, vectorizer, preprocessed_examples):
    st.header("Fake News Detection")
    
    st.markdown("""
    Enter a news article or paste some text to analyze whether it's likely to be fake news or real news.
    """)
    
    news_text = st.text_area("Enter news text:", height=200)
    
    if st.button("Analyze"):
        if not news_text:
            st.warning("Please enter some text to analyze.")
            return
        
        with st.spinner("Analyzing..."):
            # Check if we have this text pre-processed
            if news_text in preprocessed_examples:
                processed_text = preprocessed_examples[news_text]
            else:
                # Use a simplified preprocessing since we don't have NLTK
                processed_text = preprocess_text_simple(news_text)
            
            # Transform using the vectorizer
            text_vectorized = vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = model.predict(text_vectorized)[0]
            prediction_proba = abs(model.decision_function(text_vectorized)[0])
            
            # Scale the decision function to a confidence level (0-100%)
            confidence = min(100, prediction_proba * 20)  # Scaling factor can be adjusted
            
            # Display result
            st.subheader("Analysis Results")
            
            if prediction == 1:
                st.success(f"✅ This text is likely to be REAL news (Confidence: {confidence:.1f}%)")
            else:
                st.error(f"⚠️ This text is likely to be FAKE news (Confidence: {confidence:.1f}%)")
            
            # Show general explanation since we can't analyze word-level importance without full preprocessing
            st.subheader("General Analysis")
            
            if prediction == 1:
                st.markdown("""
                This content shows characteristics of legitimate news:
                - Uses more factual language
                - Contains fewer sensationalist claims
                - Presents information in a balanced way
                - Uses language patterns similar to verified news sources
                """)
            else:
                st.markdown("""
                This content shows some warning signs of fake news:
                - May use sensationalist language
                - Could contain exaggerated or unlikely claims
                - May use emotion-triggering phrases
                - Shows language patterns similar to known fake news
                """)

def show_examples_page(model, vectorizer, preprocessed_examples, dataset):
    st.header("Examples from Training Dataset")
    
    # Get data distribution
    real_count = len(dataset[dataset['label'] == 1])
    fake_count = len(dataset[dataset['label'] == 0])
    total_count = len(dataset)
    
    st.markdown(f"""
    Our dataset contains **{total_count}** news articles:
    - **{real_count}** real news articles ({real_count/total_count*100:.1f}%)
    - **{fake_count}** fake news articles ({fake_count/total_count*100:.1f}%)
    """)
    
    # Add option to sample from dataset
    sample_options = ["Random 5 Examples", "Top 5 Real News", "Top 5 Fake News", "Custom Selection"]
    sample_choice = st.radio("Select examples to display:", sample_options)
    
    if sample_choice == "Random 5 Examples":
        # Get 3 real and 2 fake randomly
        real_samples = dataset[dataset['label'] == 1].sample(3)
        fake_samples = dataset[dataset['label'] == 0].sample(2)
        samples = pd.concat([real_samples, fake_samples]).sample(frac=1)  # Shuffle
    
    elif sample_choice == "Top 5 Real News":
        samples = dataset[dataset['label'] == 1].head(5)
    
    elif sample_choice == "Top 5 Fake News":
        samples = dataset[dataset['label'] == 0].head(5)
    
    elif sample_choice == "Custom Selection":
        # Allow custom number selection
        num_real = st.slider("Number of real news examples:", 0, 5, 2)
        num_fake = st.slider("Number of fake news examples:", 0, 5, 2)
        
        real_samples = dataset[dataset['label'] == 1].sample(num_real)
        fake_samples = dataset[dataset['label'] == 0].sample(num_fake)
        samples = pd.concat([real_samples, fake_samples]).sample(frac=1)  # Shuffle
    
    # Display the selected samples
    st.subheader("Selected Examples")
    
    # Bulk analyze button
    if st.button("Analyze All Examples"):
        analyze_all = True
    else:
        analyze_all = False
    
    for idx, row in samples.iterrows():
        label = "FAKE" if row['label'] == 0 else "REAL"
        title_text = row['title'] if len(row['title']) < 100 else row['title'][:100] + "..."
        
        # Create an expander for each example
        with st.expander(f"**{label}**: {title_text}"):
            st.write(row['content'])
            
            # Option to analyze individually or in bulk
            should_analyze = analyze_all or st.button(f"Analyze this example", key=f"example_{idx}")
            
            if should_analyze:
                with st.spinner("Analyzing..."):
                    # Use simplified preprocessing
                    processed_text = preprocess_text_simple(row['content'])
                    
                    # Transform and predict
                    text_vectorized = vectorizer.transform([processed_text])
                    prediction = model.predict(text_vectorized)[0]
                    prediction_proba = abs(model.decision_function(text_vectorized)[0])
                    confidence = min(100, prediction_proba * 20)
                    
                    # Display result
                    actual_label = "REAL" if row['label'] == 1 else "FAKE"
                    predicted_label = "REAL" if prediction == 1 else "FAKE"
                    
                    # Create columns for better display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Actual label**: {actual_label}")
                    with col2:
                        st.markdown(f"**Predicted label**: {predicted_label}")
                    with col3:
                        st.markdown(f"**Confidence**: {confidence:.1f}%")
                    
                    if actual_label == predicted_label:
                        st.success("The model correctly classified this example!")
                    else:
                        st.error("The model misclassified this example.")

def show_about_page():
    st.header("About the Fake News Detection System")
    
    st.markdown("""
    ### How It Works
    
    This system uses Natural Language Processing (NLP) and Machine Learning techniques to classify news as real or fake:
    
    1. **Text Preprocessing**: Converting text to lowercase, removing punctuation and stopwords, stemming words
    
    2. **Feature Extraction**: Using TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features
    
    3. **Classification**: Using a Passive Aggressive Classifier to determine if news is real or fake
    
    ### Performance and Limitations
    
    - The system is trained on a limited dataset and may not catch all types of fake news
    - Context and subtlety can be challenging for the algorithm to detect
    - The model works best with text similar to its training data
    
    ### Best Practices
    
    - Always verify information from multiple reliable sources
    - Consider the source of the news article
    - Check for unusual claims, emotional language, or inconsistencies
    
    ### System Architecture
    
    The application uses Streamlit for the user interface and scikit-learn for machine learning components.
    """)
    
    # Create a simple architecture diagram using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define components
    components = ['User Input', 'Text Preprocessing', 'Feature Extraction', 
                 'ML Model', 'Prediction', 'Explanation']
    
    # Create positions for diagram
    y_pos = np.arange(len(components))
    x_pos = [0, 1, 2, 3, 4, 5]
    
    # Create horizontal bars
    ax.barh(y_pos, [0.7]*len(components), height=0.5, align='center', 
            color=['#6495ED', '#7FFF00', '#FF7F50', '#9370DB', '#20B2AA', '#FF6347'])
    
    # Add labels
    for i, (x, y) in enumerate(zip(x_pos, y_pos)):
        ax.text(x + 0.35, y, components[i], ha='center', va='center', 
                fontsize=12, fontweight='bold', color='black')
    
    # Add flow arrows
    for i in range(len(components)-1):
        ax.annotate('', xy=(x_pos[i+1]-0.15, y_pos[i+1]), 
                   xytext=(x_pos[i]+0.85, y_pos[i]),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # Remove axis
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-0.5, 6)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_title('Fake News Detection System Architecture', fontsize=14, fontweight='bold', pad=20)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
