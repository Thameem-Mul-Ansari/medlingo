import streamlit as st
import os
import glob
import pdfplumber
from groq import Groq
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from styles import STYLE_CSS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
from styles import STYLE_CSS
import io
import fitz
from PIL import Image

# Set page configuration at the very beginning
st.set_page_config(
    page_title="MedLingo",
    page_icon="🏥",
    layout="wide",
)

st.markdown(f"<style>{STYLE_CSS}</style>", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
groq_client = Groq(api_key=groq_api_key)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "vectors" not in st.session_state:
        st.session_state.vectors = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "loader" not in st.session_state:
        st.session_state.loader = None
    if "docs" not in st.session_state:
        st.session_state.docs = None
    if "text_splitter" not in st.session_state:
        st.session_state.text_splitter = None
    if "final_documents" not in st.session_state:
        st.session_state.final_documents = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Research Agent"

# Sidebar setup
with st.sidebar:
    st.image("lingo.jpeg", use_container_width=True)
    st.subheader("Select your Option")

    # Sidebar buttons
    if st.button("Research Agent"):
        st.session_state.current_page = "Research Agent"
    if st.button("Clinical Trials"):
        st.session_state.current_page = "Clinical Trials"
    if st.button("User Queries"):  # Corrected here
        st.session_state.current_page = "User Queries"  # Corrected here
    if st.button("Summaries"):
        st.session_state.current_page = "Summaries"
    if st.button("Diagnosis Analysis"):
        st.session_state.current_page = "Diagnosis Analysis"

def get_pdf_files(directory):
    """Retrieve a list of PDF files in the specified directory."""
    return glob.glob(os.path.join(directory, '*.pdf'))

def vector_embedding():
    """Handle vector embedding of documents"""
    if st.session_state.vectors is None:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        save_processed_documents(st.session_state.final_documents)

def save_processed_documents(documents):
    """Save processed documents to a separate folder"""
    processed_dir = './processed_documents'
    os.makedirs(processed_dir, exist_ok=True)
    for idx, doc in enumerate(documents):
        with open(os.path.join(processed_dir, f'document_{idx + 1}.txt'), 'w', encoding='utf-8') as f:
            f.write(doc.page_content)

def convert_to_boolean_query(user_query):
    """Convert natural language query to boolean query using LLM"""
    prompt = f"""
    Convert the following natural language medical query into a Boolean search query using AND, OR, NOT operators.
    Make the query specific to medical test results and parameters.
    Format the output as a clear Boolean expression.
    
    User Query: {user_query}
    
    Example conversions:
    - "Show me diabetes related results" → "glucose AND HbA1c AND insulin"
    - " Check heart health markers" → "cholesterol AND triglycerides AND HDL OR LDL"
    - "Look for kidney function excluding liver tests" → "creatinine AND BUN NOT ALT NOT AST"
    """
    
    completion = groq_client.chat.completions.create(
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
    
    return groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        stream=True
    )

def research_agent_page():
    """Display the Research Agent page"""
    st.title("MedLingo - Your Research Assistant")
    
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert medical research assistant specializing in analyzing medical reports, research papers, and clinical documents. 
    Analyze the following medical context and provide detailed, accurate responses based on the provided information.
    
    Please ensure your responses:
    - Are evidence-based and grounded in the provided medical literature
    - Include relevant clinical findings and research outcomes
    - Highlight important medical terminology and concepts
    - Reference specific sections from the source documents when applicable
    - Maintain medical accuracy and precision
    - Avoid making diagnoses or providing medical advice
    - Clearly distinguish between established findings and preliminary research
    
    <context>
    {context}
    </context>

    Question:
    {input}
    """)
    
    processed_files = set(get_pdf_files('./papers'))
    new_files = processed_files - st.session_state.processed_files
    
    if new_files:
        vector_embedding()
        st.session_state.processed_files.update(new_files)
        st.write("New documents have been embedded and stored.")
    
    prompt1 = st.text_input("Enter Your Question From Documents")
    
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time:", time.process_time() - start)
        st.write(response['answer'])
        
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

def user_queries_page():
    """Display the User Queries page"""
    
    # Center-aligned title using markdown and custom CSS
    st.markdown("""
        <h1 style='text-align: center; padding: 20px;'>User Query</h1>
        <style>
            .upload-section {
                padding: 20px;
                margin-bottom: 20px;
            }
            .pdf-preview {
                border: 1px solid #ddd;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Add a container for the upload section
        with st.container():
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.header("Upload Your PDF")
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            st.markdown('</div>', unsafe_allow_html=True)

            # Initialize pdf_text
            pdf_text = ""
            
            if uploaded_file is not None:
                try:
                    # Extract text and display PDF using fitz
                    pdf_bytes = uploaded_file.read()
                    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
                        # Extract text from all pages
                        pdf_text = ""
                        num_pages = pdf_document.page_count
                        
                        # Create a container for PDF preview
                        st.subheader("PDF Preview")
                        
                        # Add page selection slider if document has multiple pages
                        if num_pages > 1:
                            selected_page = st.slider("Select Page", 1, num_pages, 1) - 1
                        else:
                            selected_page = 0
                        
                        # Display selected page
                        page = pdf_document.load_page(selected_page)
                        
                        # Extract text from current page
                        pdf_text += page.get_text()
                        
                        # Convert page to image and display
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Zoom factor of 2 for better quality
                        img_bytes = pix.tobytes()
                        st.image(img_bytes, caption=f"Page {selected_page + 1}", use_container_width=True)
                        
                        # Display page text in an expander
                        with st.expander(f"Page {selected_page + 1} Text"):
                            st.text(page.get_text())
                            
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Rewind the file for potential reuse
                    uploaded_file.seek(0)
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    pdf_text = ""

    with col2:
        if pdf_text.strip():  # Only proceed if we have actual text content
            st.header("Analysis")
            user_query = st.text_input("Enter your medical query")

            if st.button("Analyze") and user_query:
                # Generate boolean query
                boolean_query = convert_to_boolean_query(user_query)
                st.subheader("Boolean Query")
                st.code(boolean_query, language="text")

                # Analyze PDF content based on boolean query
                st.subheader("Analysis Results")
                with st.spinner("Analyzing document..."):
                    analysis_stream = analyze_report(pdf_text, boolean_query)
                    analysis_response = ""
                    analysis_placeholder = st.empty()
                    
                    for chunk in analysis_stream:
                        if chunk.choices[0].delta.content is not None:
                            analysis_response += chunk.choices[0].delta.content
                            analysis_placeholder.markdown(analysis_response)

                # Generate suggested questions
                st.subheader("Suggested Questions")
                question_prompt = f"""
                Based on the medical report content below, generate 5 relevant questions that could be asked to better understand the medical context. 
                Focus on key findings, abnormal values, and important medical parameters mentioned in the text.
                Format as a bullet point list.

                Medical Report:
                {pdf_text[:2000]}  # Limiting text length for API constraints
                """
                
                questions_response = groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": question_prompt}],
                    temperature=0.7,
                    max_tokens=512,
                    stream=False
                )
                
                st.markdown(questions_response.choices[0].message.content)

                # Export functionality
                export_text = f"""
                Medical Document Analysis
                ========================
                
                Original Query: {user_query}
                Boolean Query: {boolean_query}
                
                Analysis Results:
                ----------------
                {analysis_response}
                
                Suggested Questions:
                -------------------
                {questions_response.choices[0].message.content}
                """
                
                st.download_button(
                    label="Download Analysis",
                    data=export_text,
                    file_name="medical_analysis.txt",
                    mime="text/plain"
                )
        elif uploaded_file is not None:
            st.warning("No text could be extracted from the uploaded PDF. Please ensure the PDF contains readable text.")

def diagnosis_analysis_page():
    st.title("Diagnosis Analysis")

    # Text Inputs
    gender = st.selectbox('Select Gender', ('Male', 'Female', 'Other'))
    age = st.number_input('Enter Age', min_value=0, max_value=120, value=25)
    symptoms = st.text_area('Enter Symptoms', 'e.g., fever, cough, headache')
    medical_history = st.text_area('Enter Medical History', 'e.g., diabetes, hypertension')

    if st.button("Get Diagnosis and Treatment Plan"):
        # Placeholder for further processing
        st.write("Generating recommendations...")
        # Here you would call your diagnostic functions or logic

def generate_summary(pdf_text, summary_type):
    """Generate summary based on the selected type"""
    prompts = {
        "Clinical": """
        Create a clinical summary of the following medical document. Focus on:
        - Key clinical findings
        - Diagnostic results
        - Treatment recommendations
        - Follow-up instructions
        Present the information in a clear, professional medical format.
        
        Document text:
        {text}
        """,
        
        "Research": """
        Create a research-oriented summary of the following medical document. Focus on:
        - Study methodology
        - Key research findings
        - Statistical significance
        - Research implications
        - Future research recommendations
        Format as a structured research summary.
        
        Document text:
        {text}
        """,
        
        "Patient": """
        Create a patient-friendly summary of the following medical document. Focus on:
        - Main health findings in simple terms
        - Treatment plan in clear language
        - Lifestyle recommendations
        - Next steps for the patient
        Use plain language and avoid medical jargon where possible.
        
        Document text:
        {text}
        """,
        
        "Administrative": """
        Create an administrative summary of the following medical document. Focus on:
        - Key administrative details
        - Billing codes and categories
        - Insurance-relevant information
        - Follow-up scheduling needs
        - Documentation requirements
        Format for healthcare administrative purposes.
        
        Document text:
        {text}
        """,
        
        "Meta-Analysis": """
        Create a summary suitable for a meta-analysis review of the following medical document. Focus on:
        - Inclusion and exclusion criteria used for studies
        - Key pooled statistical results
        - Heterogeneity among study outcomes
        - Overall effect size and confidence intervals
        - Limitations of the meta-analysis
        Present the summary in a format appropriate for systematic synthesis and interpretation.
        
        Document text:
        {text}
        """,
        
        "Systematic Review": """
        Create a summary tailored for a systematic review of the following medical document. Focus on:
        - Research question and objectives
        - Comprehensive literature search strategy
        - Quality assessment of included studies
        - Synthesis of main findings and evidence quality
        - Key gaps identified for future research
        Use a format suitable for a systematic and rigorous evaluation.
        
        Document text:
        {text}
        """
    }

    summary_response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{
            "role": "user",
            "content": prompts[summary_type].format(text=pdf_text)
        }],
        temperature=0.7,
        max_tokens=1024,
        stream=True
    )
    
    return summary_response

def summaries_page():
    """Display the Summaries page"""
    
    # Center-aligned title with custom CSS
    st.markdown("""
        <h1 style='text-align: center; padding: 20px;'>Medical Document Summaries</h1>
        <style>
            .summary-button {
                margin: 10px 0;
                width: 100%;
            }
            .pdf-preview {
                border: 1px solid #ddd;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
            }
            .summary-section {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .stButton {
                display: inline-block;
                width: auto !important;
                margin: 0 10px 10px 0 !important;
            }
            .button-container {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                justify-content: start;
                padding: 10px 0;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    
    # Initialize session state for storing the PDF text
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'current_summary' not in st.session_state:
        st.session_state.current_summary = ""
    
    with col1:
        st.header("Upload Medical Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            try:
                # Process PDF with PyMuPDF
                pdf_bytes = uploaded_file.read()
                with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
                    # Extract text from all pages
                    st.session_state.pdf_text = ""
                    num_pages = pdf_document.page_count
                    st.write(f"Total Pages: {num_pages}")
                    
                    # Create PDF preview container
                    st.markdown('<div class="pdf-preview">', unsafe_allow_html=True)
                    st.subheader("Document Preview")
                    
                    # Add page selection if multiple pages
                    if num_pages > 1:
                        selected_page = st.slider("Select Page", 1, num_pages, 1) - 1
                    else:
                        selected_page = 0
                    
                    # Display selected page
                    page = pdf_document.load_page(selected_page)
                    st.session_state.pdf_text += page.get_text()
                    
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_bytes = pix.tobytes()
                    st.image(img_bytes, caption=f"Page {selected_page + 1}", use_container_width=True)
                    
                    # Show page text in expander
                    with st.expander(f"Page {selected_page + 1} Text"):
                        st.text(page.get_text())
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                uploaded_file.seek(0)
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.session_state.pdf_text = ""
    
    with col2:
        st.header("Generate Summary")
        
        if st.session_state.pdf_text.strip():
            # Summary type buttons
            st.markdown('<div class="summary-section">', unsafe_allow_html=True)
            
            # Define summary types with their respective labels and types
            summary_types = {
                "Clinical Summary": "Clinical",
                "Research Summary": "Research",
                "Patient-Friendly": "Patient",
                "Administrative": "Administrative",
                "Meta-Analysis": "Meta-Analysis",
                "Systematic Review": "Systematic Review"
            }
            
            # Create a container for buttons
            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            
            # Create columns for button layout
            button_cols = st.columns(3)
            
            # Distribute buttons across columns
            for idx, (button_text, summary_type) in enumerate(summary_types.items()):
                col_idx = idx % 3
                with button_cols[col_idx]:
                    if st.button(button_text, key=f"btn_{summary_type}", help=f"Generate {button_text}"):
                        st.session_state.current_summary = ""  # Clear previous summary
                        with st.spinner(f"Generating {button_text}..."):
                            summary_stream = generate_summary(st.session_state.pdf_text, summary_type)
                            summary_placeholder = st.empty()
                            
                            # Stream the summary
                            for chunk in summary_stream:
                                if chunk.choices[0].delta.content is not None:
                                    st.session_state.current_summary += chunk.choices[0].delta.content
                                    summary_placeholder.markdown(st.session_state.current_summary)
                            
                            # Add download button for the summary
                            st.download_button(
                                label=f"Download {button_text}",
                                data=st.session_state.current_summary,
                                file_name=f"medical_summary_{summary_type.lower()}.txt",
                                mime="text/plain"
                            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.info("Please upload a PDF document to generate summaries.")

def clinical_trials_page():
    """Display the Clinical Trials page"""
    st.title("Clinical Trials")

def main():
    """Main function to run the application"""
    init_session_state()
    
    pages = {
    "Research Agent": research_agent_page,
    "Diagnosis Analysis": diagnosis_analysis_page,
    "User Queries": user_queries_page,  # Corrected here
    "Summaries": summaries_page,
    "Clinical Trials": clinical_trials_page
    }
    
    pages[st.session_state.current_page]()

if __name__ == "__main__":
    main()