import streamlit as st
import pdfplumber
from groq import Groq
from dotenv import load_dotenv
import time
import os

# Set page configuration at the very beginning
st.set_page_config(
    page_title="MedLingo - User Query",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def convert_to_boolean_query(user_query):
    """Convert natural language query to boolean query using LLM"""
    prompt = f"""
    Convert the following natural language medical query into a Boolean search query using AND, OR, NOT operators.
    Make the query specific to medical test results and parameters.
    Format the output as a clear Boolean expression.
    
    User Query: {user_query}
    
    Example conversions:
    - "Show me diabetes related results" → "glucose AND HbA1c AND insulin"
    - "Check heart health markers" → "cholesterol AND triglycerides AND HDL OR LDL"
    - "Look for kidney function excluding liver tests" → "creatinine AND BUN NOT ALT NOT AST"
    """
    
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=256,
        stream=False
    )
    
    return completion.choices[0].message.content.strip()

def analyze_report(pdf_text, boolean_query):
    """Analyze the PDF content based on the boolean query"""
    prompt = f"""
    Analyze the following medical report text based on this Boolean query: {boolean_query}
    
    Focus on:
    1. Extracting relevant test results and values
    2. Highlighting abnormal values
    3. Organizing related parameters together
    4. Providing brief contextual information for the findings
    
    Report Text:
    {pdf_text}
    """
    
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        stream=True
    )
    
    return completion

def main():
    st.title("User Query")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader("Upload Medical Report (PDF)", type="pdf")
        
        # User query input
        user_query = st.text_input(
            "What would you like to know about the report?",
            placeholder="Example: Show me all diabetes related test results"
        )
    
    if uploaded_file and user_query:
        # Extract PDF text
        with pdfplumber.open(uploaded_file) as pdf:
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text() + "\n"
        
        # Convert user query to boolean query
        with st.spinner("Converting your query..."):
            boolean_query = convert_to_boolean_query(user_query)
            st.subheader("Generated Search Query")
            st.code(boolean_query, language="text")
        
        # Analyze report based on boolean query
        with st.spinner("Analyzing the report..."):
            completion = analyze_report(pdf_text, boolean_query)
            
            st.subheader("Analysis Results")
            result_container = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    result_container.markdown(full_response)
        
        # Show full PDF content in expander
        with st.expander("View Full Report Content"):
            st.text(pdf_text)
    
    elif not uploaded_file:
        st.info("Please upload a PDF report to begin.")
    elif not user_query:
        st.info("Please enter your query about the report.")

if __name__ == "__main__":
    main()