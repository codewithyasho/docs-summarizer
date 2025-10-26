# document summarizer

from langchain_groq import ChatGroq
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()


# LOADING PDF DOCUMENT
pdf_loader = PyPDFLoader("resume.pdf")
pdf_docs = pdf_loader.load()

documents = ""
for i in range(len(pdf_docs)):
    documents += pdf_docs[i].page_content + "\n"
# print(documents)


# INITIALIZING THE LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

# CREATING THE PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_template(
    "Summarize the following document: {documents}."
)


# CREATING THE CHAIN
chain = prompt| llm | StrOutputParser()

## DISPLAYING THE SUMMARIES FOR EACH CHUNK
result = chain.invoke({"documents": documents})

print("Document Summary and Key Insights:")
print(result)

