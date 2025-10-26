import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

st.title("Document Summarizer and QnA")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'tmp_file_path' not in st.session_state:
    st.session_state.tmp_file_path = None

# LOADING PDF DOCUMENT
pdf = st.file_uploader("Upload file", type="pdf")

if pdf is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf.read())
        tmp_file_path = tmp_file.name

    pdf_loader = PyPDFLoader(tmp_file_path)
    pdf_docs = pdf_loader.load()

    documents = ""
    for i in range(len(pdf_docs)):
        documents += pdf_docs[i].page_content + "\n"

    # Store in session state
    st.session_state.documents = documents
    st.session_state.tmp_file_path = tmp_file_path

    st.success(f"✅ PDF loaded successfully! ({len(pdf_docs)} pages)")
else:
    st.session_state.documents = None


# INITIALIZING THE LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

# Only show features if document is loaded
if st.session_state.documents:

    # Create tabs for Summary and Q&A
    tab1, tab2 = st.tabs(["📄 Summary", "💬 Ask Questions"])

    # TAB 1: SUMMARY
    with tab1:
        st.markdown("### Generate Document Summary")

        # CREATING THE PROMPT TEMPLATE FOR SUMMARY
        summary_prompt = ChatPromptTemplate.from_template(
            "Summarize the following document in a clear and concise manner: {documents}"
        )

        # CREATING THE CHAIN FOR SUMMARY
        summary_chain = summary_prompt | llm | StrOutputParser()

        if st.button("Generate Summary", key="summary_btn"):
            with st.spinner("Generating summary..."):
                result = summary_chain.invoke(
                    {"documents": st.session_state.documents})
                st.subheader("Document Summary:")
                st.write(result)

    # TAB 2: Q&A
    with tab2:
        st.markdown("### Ask Questions About Your Document")

        # Question input
        user_question = st.text_input(
            "Enter your question:", placeholder="What is this document about?")

        if st.button("Get Answer", key="qa_btn") and user_question:
            with st.spinner("Finding answer..."):

                # CREATING THE PROMPT TEMPLATE FOR Q&A
                qa_prompt = ChatPromptTemplate.from_template(
                    """
                    Based on the following document, answer the question. If the answer cannot be found in the document, say "I cannot find the answer in the provided document."           
                    Document:
                    {documents}

                    Question: {question}

                    Answer:
                    """
                )

                # CREATING THE CHAIN FOR Q&A
                qa_chain = qa_prompt | llm | StrOutputParser()

                answer = qa_chain.invoke({
                    "documents": st.session_state.documents,
                    "question": user_question
                })

                st.subheader("Answer:")
                st.write(answer)

        # Display chat history if needed
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Store Q&A in session
        if user_question and st.session_state.get('last_answer'):
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": st.session_state.last_answer
            })

else:
    st.info("👆 Please upload a PDF document to get started.")

# Cleanup on session end (optional)
# Note: Streamlit doesn't have a perfect session end hook,
# so temporary files might accumulate. Consider periodic cleanup.
