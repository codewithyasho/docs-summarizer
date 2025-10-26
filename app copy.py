import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import tempfile


load_dotenv()

st.title("Document Summarizer and QnA")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'tmp_file_path' not in st.session_state:
    st.session_state.tmp_file_path = None
if 'pdf_docs' not in st.session_state:
    st.session_state.pdf_docs = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# LOADING PDF DOCUMENT
pdf = st.file_uploader("Upload file", type="pdf")

if pdf is not None:
    with st.spinner("Loading..."):
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_file_path = tmp_file.name

        pdf_loader = PyPDFLoader(tmp_file_path)
        pdf_docs = pdf_loader.load()

        # Split documents into chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(pdf_docs)

        # Create vector store for RAG (Q&A)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Store in session state
        st.session_state.pdf_docs = pdf_docs
        st.session_state.chunks = chunks
        st.session_state.vectorstore = vectorstore
        st.session_state.tmp_file_path = tmp_file_path

        st.success(
            f"✅ PDF loaded successfully! ({len(pdf_docs)} pages, {len(chunks)} chunks)")
else:
    st.session_state.documents = None
    st.session_state.pdf_docs = None
    st.session_state.vectorstore = None


# INITIALIZING THE LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

# Only show features if document is loaded
if st.session_state.pdf_docs:

    # Create tabs for Summary and Q&A
    tab1, tab2 = st.tabs(["📄 Summary", "💬 Ask Questions"])

    # TAB 1: SUMMARY (MapReduce Strategy)
    with tab1:
        st.markdown("### Generate Document Summary")

        summary_type = st.radio(
            "Select summary type:",
            ["Quick Summary (Faster)", "Detailed Summary"],
            horizontal=True
        )

        if st.button("Generate Summary", key="summary_btn"):
            with st.spinner("Generating summary... This may take a moment for large documents."):
                try:
                    if summary_type == "Quick Summary (Faster)":
                        # OPTIMIZED: Use only 5-7 chunks and process in ONE call
                        chunks_to_process = st.session_state.chunks[:7]
                        
                        # Combine chunks into one text
                        combined_text = "\n\n".join([chunk.page_content for chunk in chunks_to_process])
                        
                        # Single API call for quick summary
                        quick_prompt = ChatPromptTemplate.from_template(
                            """
                            Provide a concise summary of the following document excerpt. 
                            Focus on the main points and key information:

                            {text}

                            Summary:
                            """
                        )
                        quick_chain = quick_prompt | llm | StrOutputParser()
                        result = quick_chain.invoke({"text": combined_text})
                        
                        st.subheader("Document Summary:")
                        st.write(result)
                        st.caption(f"⚡ Quick summary from first {len(chunks_to_process)} chunks ({len(st.session_state.pdf_docs)} total pages)")
                    
                    else:
                        # Detailed Summary - MapReduce approach
                        chunks_to_process = st.session_state.chunks
                        
                        # Map phase: Summarize each chunk
                        chunk_summaries = []
                        progress_bar = st.progress(0)

                        for i, chunk in enumerate(chunks_to_process):
                            summary_prompt = ChatPromptTemplate.from_template(
                                "Provide a concise summary of the following text:\n\n{text}"
                            )
                            chain = summary_prompt | llm | StrOutputParser()
                            summary = chain.invoke({"text": chunk.page_content})
                            chunk_summaries.append(summary)
                            progress_bar.progress((i + 1) / len(chunks_to_process))

                        # Reduce phase: Combine all summaries
                        st.text("Combining summaries...")
                        combined_summary_text = "\n\n".join(chunk_summaries)

                        final_prompt = ChatPromptTemplate.from_template(
                            """
                            Create a comprehensive summary from these section summaries:

                            {summaries}

                            Provide a well-structured final summary:
                            """
                        )
                        final_chain = final_prompt | llm | StrOutputParser()
                        result = final_chain.invoke({"summaries": combined_summary_text})

                        progress_bar.empty()
                        st.subheader("Document Summary:")
                        st.write(result)
                        st.caption(f"📊 Processed {len(chunks_to_process)} chunks from {len(st.session_state.pdf_docs)} pages")

                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")

    # TAB 2: Q&A (RAG Strategy)
    with tab2:
        st.markdown("### Ask Questions About Your Document")

        # Question input
        user_question = st.text_input(
            "Enter your question:",
            placeholder="What is this document about?",
            key="question_input"
        )

        if st.button("Get Answer", key="qa_btn") and user_question:
            with st.spinner("Finding answer in document..."):
                try:
                    # Retrieve relevant chunks using vector similarity
                    relevant_docs = st.session_state.vectorstore.similarity_search(
                        user_question,
                        k=4  # Get top 4 most relevant chunks
                    )

                    # Combine relevant chunks
                    context = "\n\n".join(
                        [doc.page_content for doc in relevant_docs])

                    # CREATING THE PROMPT TEMPLATE FOR Q&A
                    qa_prompt = ChatPromptTemplate.from_template(
                        """
                        Based on the following context from the document, answer the question. 
                        If the answer cannot be found in the context, say "I cannot find the answer in the provided document."

                        Context:
                        {context}

                        Question: {question}

                        Answer:
                        """
                    )

                    # CREATING THE CHAIN FOR Q&A
                    qa_chain = qa_prompt | llm | StrOutputParser()

                    answer = qa_chain.invoke({
                        "context": context,
                        "question": user_question
                    })

                    st.subheader("Answer:")
                    st.write(answer)

                except Exception as e:
                    st.error(f"Error answering question: {str(e)}")

else:
    st.info("👆 Please upload a PDF document to get started.")


# footer
st.markdown("---")
st.markdown("*Powered by LangChain & Groq*")

