# 📄 Document Summarizer and QnA

An intelligent document processing application that leverages LangChain, Groq LLM, and FAISS vector store to provide document summarization and question-answering capabilities for PDF files.

## 🌟 Features

### 1. **PDF Document Upload**
- Easy drag-and-drop PDF file upload
- Automatic document processing and chunking
- Support for multi-page documents

### 2. **Smart Summarization**
- **Quick Summary**: Fast summarization using the first few chunks (ideal for quick insights)
- **Detailed Summary**: Comprehensive MapReduce-based summarization of the entire document
  - Map phase: Individual chunk summarization
  - Reduce phase: Intelligent consolidation of all summaries
- Progress tracking for detailed summaries

### 3. **Question & Answer (RAG)**
- Ask natural language questions about your document
- Retrieval-Augmented Generation (RAG) for accurate answers
- Context-aware responses based on document content
- Vector similarity search to find relevant information

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Groq (openai/gpt-oss-120b model)
- **Framework**: LangChain
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Document Processing**: PyPDFLoader, RecursiveCharacterTextSplitter

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key
- Internet connection for downloading embeddings model (first-time setup)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd project6-docs-summarizer
```

2. **Create a virtual environment**
```bash
python -m venv .venv
```

3. **Activate the virtual environment**

On Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:
```bash
source .venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Set up environment variables**

Create a `.env` file in the project root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

To get a Groq API key:
- Visit [Groq Console](https://console.groq.com)
- Sign up for a free account
- Generate an API key from the dashboard

## 💻 Usage

1. **Start the application**
```bash
streamlit run app.py
```

2. **Upload a PDF**
   - Click the "Upload file" button
   - Select a PDF document from your computer
   - Wait for the document to be processed

3. **Generate Summary**
   - Navigate to the "📄 Summary" tab
   - Choose between Quick or Detailed summary
   - Click "Generate Summary"

4. **Ask Questions**
   - Navigate to the "💬 Ask Questions" tab
   - Type your question in the input field
   - Click "Get Answer" to receive a response

## 🏗️ Architecture

### Document Processing Pipeline
```
PDF Upload → PyPDFLoader → Text Splitting → Embeddings → FAISS Vector Store
```

### Summarization (MapReduce)
```
Document Chunks → Map (Individual Summaries) → Reduce (Final Summary)
```

### Q&A (RAG)
```
User Question → Vector Similarity Search → Context Retrieval → LLM → Answer
```

## 📊 Configuration

### Text Splitting Parameters
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters

### RAG Parameters
- **Top K Results**: 4 most relevant chunks

### LLM Settings
- **Model**: openai/gpt-oss-120b (via Groq)
- **Temperature**: 0.1 (for consistent, factual responses)

## 🎯 Use Cases

- **Academic Research**: Quickly summarize research papers and extract key information
- **Business Documents**: Analyze reports, contracts, and proposals
- **Legal Documents**: Review and query legal documents efficiently
- **Educational Material**: Create summaries of textbooks and study materials
- **Technical Documentation**: Navigate and understand complex technical docs

## 🔧 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your `.env` file exists and contains a valid `GROQ_API_KEY`
   - Verify the API key is active in the Groq console

2. **Memory Issues with Large PDFs**
   - Use "Quick Summary" for faster processing
   - Consider splitting very large PDFs into smaller files

3. **Embeddings Download**
   - First-time setup downloads the HuggingFace model (~80MB)
   - Requires internet connection

## 📝 Project Structure

```
project6-docs-summarizer/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
├── README.md                 # Project documentation
└── .venv/                    # Virtual environment (after setup)
```

## 🔐 Security Notes

- Never commit your `.env` file to version control
- Keep your API keys confidential
- Add `.env` to your `.gitignore` file

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the amazing framework
- [Groq](https://groq.com/) for fast LLM inference
- [Streamlit](https://streamlit.io/) for the intuitive UI framework
- [HuggingFace](https://huggingface.co/) for embeddings models

## 📧 Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.

---

*Built with ❤️ using LangChain & Groq*
