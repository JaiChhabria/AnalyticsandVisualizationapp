import streamlit as st
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import openai
import os
from dotenv import load_dotenv
from summarizer import Summarizer
import docx2txt
from transformers import AlbertModel
from transformers import BartTokenizer, BartForConditionalGeneration, BertForSequenceClassification
from docx import Document 

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Read data from file
def read_data(file, file_type):
    if file_type == "xlsx":
        df = pd.read_excel(file)
    elif file_type == "csv":
        df = pd.read_csv(file)
    return df

# Generate automatic summary using BERT Extractive Summarizer
def generate_text_summary(text, word_limit=100):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    local_model_path = r"C:\Users\jaich\OneDrive\LLM_Project_Personal"
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    
    inputs = tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=word_limit, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Function to perform data summarization for CSV and Excel files
def generate_data_summary(data):
    # Add your data summarization logic here
    # Example: Calculate statistics, create charts, etc.
    summary = "Data summarization result goes here."
    return summary

# Redact personal information from text
def redact_personal_info(text):
    # Replace personal info with [REDACTED]
    redacted_text = text.replace("John Doe", "[REDACTED]","Vedika Chhabria","[REDACTED]")
    redacted_text = redacted_text.replace("vedikachharbia@gmail.com", "[REDACTED]")
    # Add more redactions as needed
    return redacted_text

# Create a word cloud from text
def create_word_cloud(text):
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    return plt

# Create a bar chart
def create_bar_chart(dataframe, x_column, y_column):
    plt.figure(figsize=(10, 6))
    plt.bar(dataframe[x_column], dataframe[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"Bar Chart: {y_column} vs {x_column}")
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Create a scatter plot
def create_scatter_plot(dataframe, x_column, y_column):
    plt.figure(figsize=(10, 6))
    plt.scatter(dataframe[x_column], dataframe[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"Scatter Plot: {y_column} vs {x_column}")
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Create a chat interaction with the model
def ask_question(question, df=None, doc_text=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides information based on the uploaded data."},
        {"role": "user", "content": f"Question: {question}"}
    ]
    if df is not None:
        messages.append({"role": "assistant", "content": f"Data: {df.to_string()}"})
    if doc_text is not None:
        messages.append({"role": "user", "content": f"Text: {doc_text}"})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    answer = response.choices[0].message['content']
    return answer

# Define the main function
def main():
    st.title("Data Analysis and Visualization App")
    st.sidebar.title("Options")
    
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["xlsx", "csv", "docx"])
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1]
        
        if file_type in ["xlsx", "csv"]:
            df = read_data(uploaded_file, file_type)
            
            st.subheader("Data Summary")
            automatic_summary = generate_data_summary(df)
            st.write(automatic_summary)
            
            st.subheader("Ask a Question")
            user_question = st.text_input("Ask a question about the data")
            
            if user_question:
                answer = ask_question(user_question, df)
                st.write("Answer:", answer)
            
            st.subheader("Chart Creation")
            chart_type = st.sidebar.selectbox("Select chart type", ["Bar", "Scatter"])
            
            if chart_type == "Bar":
                st.subheader("Bar Chart")
                x_column = st.sidebar.selectbox("Select X-axis column", df.columns)
                y_column = st.sidebar.selectbox("Select Y-axis column", df.columns)
                create_bar_chart(df, x_column, y_column)

            elif chart_type == "Scatter":
                st.subheader("Scatter Plot")
                x_column = st.sidebar.selectbox("Select X-axis column", df.columns)
                y_column = st.sidebar.selectbox("Select Y-axis column", df.columns)
                create_scatter_plot(df, x_column, y_column)
            
            # Include the code for other chart types (Line, Pie, Heat Map) here
        
        elif file_type == "docx":
            # Load text from docx file
            doc = Document(uploaded_file)
            doc_text = "\n".join([para.text for para in doc.paragraphs])
            
            st.subheader("Text Summarization")
            user_word_limit = st.number_input("Enter word limit for summary:", value=100)
            text_summary = generate_text_summary(doc_text, word_limit=user_word_limit)
            st.write(text_summary)
            
            st.subheader("Word Cloud")
            word_cloud_plot = create_word_cloud(text_summary)
            st.pyplot(word_cloud_plot)

# Run the app
if __name__ == "__main__":
    main()
