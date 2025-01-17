import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import pandas as pd
import tempfile

# Function to read and combine text from all pages in a PDF
def read_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    combined_text = ""
    for doc in docs:
        combined_text += doc.page_content + "\n"

    temp_file.close()
    return combined_text

# Streamlit app title and instructions
st.title("AI Based Automatic RFP Generator")
st.markdown("""
    <style>
    .title {
        font-size: 3em;
        color: #4CAF50;
        text-align: center;
    }
    .instructions {
        font-size: 1.2em;
        color: #555;
        text-align: center;
        margin-bottom: 2em;
    }
    .file-uploader {
        margin-bottom: 2em;
    }
    .question-input {
        margin-bottom: 2em;
    }
    .answer {
        font-size: 1.2em;
        color: #333;
        background-color: #f9f9f9;
        padding: 1em;
        border-radius: 10px;
        margin-bottom: 2em;
    }
    .rfp-document {
        font-size: 1.2em;
        color: #333;
        background-color: #e9e9e9;
        padding: 1em;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="instructions">Transform Your Procurement Process with Intelligent Automation: Instant, Accurate, and Effortless RFP Generation via upload a simple sample PDF document.</p>', unsafe_allow_html=True)

# File upload widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="file-uploader")

# Text input for the question
question = st.text_input("Enter your question based on the invoice:", key="question-input")

# Process the uploaded file only when a question is entered
if uploaded_file is not None and question:
    pdf_text = read_pdf(uploaded_file)

    openai_api_key = "Enter you Open API Key"
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
    PROMPT_TEMPLATE = """
You are an advanced RFP Generator powered by Qult AI. Your task is to generate a comprehensive Request for Proposal (RFP) document based on the provided context and user query.

Context:
{context}

Instructions:
1. Analyze the context thoroughly to understand the requirements and objectives.
2. Generate the RFP document with all necessary headings and sections as outlined in the uploaded PDF.
3. Ensure the RFP is structured, detailed, and addresses all aspects mentioned in the context and user query.
4. Maintain a professional tone and format throughout the document.
5. If the user query is not related to RFP generation, respond with "This query is not related to RFP generation and is outside my functionality."

User Query:
{question}

Output the complete RFP document below:
"""


#     PROMPT_TEMPLATE = """
# You are tasked with generating a Request for Proposal (RFP) document based on the provided context.

# Context:
# {context}

# The RFP should be structured under the following sections:

# 1. **Introduction and Overview**
#    - 1.1 **Statement of Confidentiality**: Generate a statement explaining the confidentiality requirements.
#    - 1.2 **Company - Organisation Overview**: Provide a summary of stc's organization, its mission, vision, and objectives.
#    - 1.3 **Purpose of the RFP**: Describe the purpose of the RFP and why it is being issued.
#    - 1.4 **Company Business Requirements**: Specify the business needs and objectives that this RFP is addressing.

# 2. **Project Description**
#    - 2.1 **Scope and Approach**: Define the scope of the project and the approach to be taken.
#    - 2.2 **Deliverables**: List the expected deliverables for the project.
#    - 2.3 **Assumptions and Constraints**: Detail any assumptions made and constraints that apply to the project.
#    - 2.4 **Anticipated Project Time Frame**: Provide a timeline for the completion of the project.
#    - 2.5 **Selection Process and Evaluation Criteria**: Outline the process for vendor selection and the criteria for evaluation.
#    - 2.6 **General Terms and Conditions**: List the general terms and conditions that apply to the RFP.
#    - 2.7 **Project Specific Terms and Conditions**: Include any project-specific terms and conditions.

# 3. **Vendor Response**
#    - 3.1 **Proposal Submission Requirements**: Explain the submission requirements for the proposal.
#    - 3.2 **Questions Relating to the RFP & Submission of Proposal**: Provide guidance on how questions related to the RFP should be handled and how the proposal should be submitted.

# 4. **stc Requirements**
#    - 4.1 **General Requirements**: Detail the general requirements for the project.
#    - 4.2 **Requirements**: Provide any specific requirements for the project.
#    - 4.3 **Performance and Workload Requirements**: Include any performance and workload expectations.
#    - 4.4 **Reliability, Availability Requirements**: Outline the reliability and availability expectations.
#    - 4.5 **Updates and Maintenance**: Define expectations around updates and maintenance after the project completion.
#    - 4.6 **Documentation and stc Software & Hardware Ownership**: Specify documentation requirements and the ownership of software and hardware.

# 5. **Attached Documents**: Mention any documents attached to the RFP that are relevant for the vendor to review.

# ---

# Answer the question based on the above context: {question}
# """

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    llmchain = LLMChain(llm=llm, prompt=prompt)

    response = llmchain.invoke({"context": pdf_text, "question": question})
    answer = response["text"]

    st.markdown('<p class="answer">Answer to your question:</p>', unsafe_allow_html=True)
    st.write(answer)

    st.markdown('<p class="rfp-document">Generated RFP Document Based on Invoice Content:</p>', unsafe_allow_html=True)
    rfp_response = llmchain.invoke({"context": pdf_text, "question": "Generate an RFP based on the invoice"})
    rfp_data = rfp_response["text"]
    st.write(rfp_data)

else:
    if uploaded_file:
        st.write("Please enter a question based on the invoice to get an answer.")
    else:
        st.write("Please upload a PDF invoice and enter your question.")
