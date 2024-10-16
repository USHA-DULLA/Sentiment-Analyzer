import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# Load the Hugging Face pre-trained sentiment-analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Set up the Streamlit page
st.set_page_config(page_title="Advanced Sentiment Analysis App", page_icon="😊", layout="centered")

# Add title and description
st.title("Sentiment Analyzer 😊")
st.write("Enter any text below, and the app will predict the sentiment with confidence levels. You can also upload a text file for analysis.")

# Text input from the user
user_input = st.text_area("Enter your text here:", "")
upload_file = st.file_uploader("Or upload a text file (supports .txt)", type="txt")

# Store results for multiple analyses
results = []

# Analyze text from user input
if st.button("Analyze Text"):
    if user_input:
        # Perform sentiment analysis
        result = sentiment_pipeline(user_input)
        results.append((user_input, result[0]))
        
        # Display the result
        sentiment = result[0]['label']
        score = result[0]['score']
        
        st.write(f"**Sentiment**: {sentiment}")
        st.write(f"**Confidence**: {score:.2f}")
    else:
        st.write("Please enter some text for analysis.")

# Analyze text from uploaded file
if st.button("Analyze File") and upload_file is not None:
    # Read the file contents
    file_content = upload_file.read().decode("utf-8")
    
    # Perform sentiment analysis
    result = sentiment_pipeline(file_content)
    results.append((file_content, result[0]))
    
    # Display the result
    sentiment = result[0]['label']
    score = result[0]['score']
    
    st.write(f"**File Sentiment**: {sentiment}")
    st.write(f"**Confidence**: {score:.2f}")

# Visualization of results
if results:
    # Create a DataFrame for visualization
    sentiments = [result[1]['label'] for result in results]
    confidence_scores = [result[1]['score'] for result in results]
    
    st.subheader("Sentiment Distribution")
    sentiment_counts = {label: sentiments.count(label) for label in set(sentiments)}
    
    # Bar chart for sentiment distribution
    plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red', 'orange'])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')
    st.pyplot(plt)

    # Display detailed results
    st.subheader("Detailed Results")
    for input_text, result in results:
        st.write(f"**Input Text**: {input_text}")
        st.write(f"**Sentiment**: {result['label']} - **Confidence**: {result['score']:.2f}")
    
# Clear Button
if st.button("Clear"):
    user_input = ""
    results.clear()
    st.experimental_rerun()

# Footer
st.write("Built using Hugging Face's Transformers and Streamlit.")
