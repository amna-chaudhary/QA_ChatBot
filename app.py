import gradio as gr
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4.1-mini")

# Initialize embedding model
embeddings = OpenAIEmbeddings()

# Global retriever
retriever = None


def upload_pdf(pdf_path):
    """
    Upload and process the PDF.
    """
    global retriever

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Create vector database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever()

    return "✅ PDF uploaded and processed successfully!"


def chat(question):
    """
    Answer questions from the uploaded PDF.
    """
    global retriever

    if retriever is None:
        return "Please upload a PDF first."

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.

Answer the user's question using ONLY the context below.

Context:
{context}

Question:
{question}
"""
    )

    # QA Chain
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return answer


# ---------------------- Gradio UI ---------------------- #

with gr.Blocks(title="PDF Question Answering Bot") as demo:

    gr.Markdown("# 📄 PDF Question Answering Bot")

    pdf = gr.File(
        label="Upload PDF",
        file_types=[".pdf"],
        type="filepath"      # IMPORTANT
    )

    upload_btn = gr.Button("Upload PDF")

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    upload_btn.click(
        fn=upload_pdf,
        inputs=pdf,
        outputs=status
    )

    question = gr.Textbox(
        label="Ask a Question",
        placeholder="Example: What this paper is talking about?"
    )

    ask_btn = gr.Button("Ask")

    answer = gr.Textbox(
        label="Answer",
        lines=10
    )

    ask_btn.click(
        fn=chat,
        inputs=question,
        outputs=answer
    )

demo.launch()