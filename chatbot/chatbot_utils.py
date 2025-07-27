import os
import pandas as pd
import tiktoken
from langchain_openai import OpenAIEmbeddings, OpenAI
import numpy as np
import pathlib
from scipy.spatial.distance import pdist, squareform
from typing import Any
from langchain_community.document_loaders import WikipediaLoader, TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredEmailLoader, UnstructuredCSVLoader
from langchain_community.retrievers import KNNRetriever, PubMedRetriever
from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, FlareChain, OpenAIModerationChain, SequentialChain
from langchain.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chains.base import Chain
import tempfile
import logging

# Environment setup
os.environ["OPENAI_API_KEY"] = "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"

# Logging setup
logging.basicConfig(encoding="utf-8", level=logging.INFO)
LOGGER = logging.getLogger()

# Initialize LLM
LLM = OpenAI()

# Prompt Template for CSV Queries
PROMPT = (
    "If you do not know the answer, say you don't know.\n"
    "Think step by step.\n"
    "\n"
    "Below is the query.\n"
    "Query: {query}\n"
)

# Document Loader Exception
class DocumentLoaderException(Exception):
    pass

# Document Loader Utility
class DocumentLoader(object):
    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".xlsx": UnstructuredEmailLoader,
        ".csv": UnstructuredCSVLoader,
        ".doc": UnstructuredWordDocumentLoader
    }

def load_document(temp_filepath: str) -> list[Document]:
    ext = pathlib.Path(temp_filepath).suffix
    loader = DocumentLoader.supported_extensions.get(ext)
    if not loader:
        raise DocumentLoaderException(f"Invalid Extension {ext}, not trained for that... Sorry")

    loader = loader(temp_filepath)
    docs = loader.load()
    logging.info(docs)
    return docs

# Retriever Configuration
def configure_retriever(docs: list[Document], use_compression: bool = False) -> BaseRetriever:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 7, "include_metadata": True})
    if not use_compression:
        return retriever

    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.2)
    return ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)

# Memory Initialization
def init_memory():
    return ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

MEMORY = init_memory()

# Chain Configuration for Document Chatbot
def configure_chain(retriever: BaseRetriever, use_flare: bool = True) -> Chain:
    output_key = 'response' if use_flare else 'answer'
    MEMORY.output_key = output_key
    params = dict(llm=LLM, retriever=retriever, memory=MEMORY, verbose=True, max_tokens_limit=4000)
    if use_flare:
        return FlareChain.from_llm(**params)
    return ConversationalRetrievalChain.from_llm(**params)

# Retrieval Chain for Document Chatbot
def configure_retrieval_chain(uploaded_files, use_compression: bool = False, use_flare: bool = False, use_moderation: bool = False) -> Chain:
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))

    retriever = configure_retriever(docs=docs, use_compression=use_compression)
    chain = configure_chain(retriever=retriever, use_flare=use_flare)
    if not use_moderation:
        return chain

    input_variables = ["user_input"] if use_flare else ["chat_history", "question"]
    moderation_input = "response" if use_flare else "answer"
    moderation_chain = OpenAIModerationChain(input_key=moderation_input)
    return SequentialChain(chains=[chain, moderation_chain], input_variables=input_variables)

# Agent for CSV-based Queries
def create_agent(csv_file: str) -> AgentExecutor:
    df = pd.read_csv(csv_file)
    return create_pandas_dataframe_agent(LLM, df, verbose=True, allow_dangerous_code=True)

def query_agent(agent: AgentExecutor, query: str) -> str:
    prompt = PromptTemplate(template=PROMPT, input_variables=["query"])
    return agent.run(prompt.format(query=query))
