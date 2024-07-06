import os
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



def process_single_document(doc_path, model_type, user_prompt):
    # Load and split the document
    loader = PDFPlumberLoader(doc_path)
    docs = loader.load_and_split()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Create the vector database from chunks
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name=f"local-rag-{os.path.basename(doc_path)}"
    )

    cached_llm = Ollama(model=model_type)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        cached_llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | cached_llm
        | StrOutputParser()
    )

    return chain.invoke(user_prompt)



