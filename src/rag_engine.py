import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGEngine:
    def __init__(self, chroma_dir="./chroma_db", model_name="llama3"):
        self.chroma_dir = chroma_dir
        self.model_name = model_name
        
        # Incializimi i modelit te Embeddings nga Hugging Face
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Inicializimi i LLM lokal (Ollama)
        self.llm = Ollama(model=self.model_name)
        
        # Lidhja me ChromaDB
        self.vector_store = Chroma(persist_directory=self.chroma_dir, embedding_function=self.embeddings)
        
    def ingest_pdf(self, file_path):
        # Leximi i PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Ndarja ne copa me overlap per te mos humbur kontekstin
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Ruajtja ne ChromaDB
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def query(self, question):
        # Krijimi i retriever (merr 3 copat me te ngjashme)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Prompti per t'i thene modelit te bazohet VETEM te dokumenti
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context. If you don't know the answer, say that you don't know.\n\nContext: {context}\n\nQuestion: {input}\n\nAnswer:"
        )
        
        # Krijimi i zinxhirit LangChain
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        # Ekzekutimi
        response = retrieval_chain.invoke({"input": question})
        return response["answer"]