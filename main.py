import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader


def load_document(file: str):
    print(f"Loading {file} ...", end=" ")
    name, extension = os.path.splitext(file)
    if extension == ".pdf":
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
    elif extension == ".docx":
        loader = Docx2txtLoader(file)
        pages = loader.load()
    elif extension == ".txt":
        loader = TextLoader(file)
        pages = loader.load()
    else:
        print("Document format is not supported!")
        return None
    print("Done")
    return pages


def create_chunks(data, chunk_size=256, chunk_overlap=20) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    data = text_splitter.split_documents(data)
    return data


def calculate_embedding_cost(chunks: list):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in chunks])
    return total_tokens, total_tokens / 1000 * 0.0004


def create_embeddings(chunks: list):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, question: str, k=3):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(question)
    return answer


if __name__ == "__main__":
    st.image("./assets/img.png")
    st.subheader("LLM Question-Answering Application 🤖")
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "docx", "txt"])
        chunk_size = st.number_input("Chunk size:", min_value=100, max_value=2048, value=512)
        k = st.number_input("k", min_value=1, max_value=20, value=3)
        add_data = st.button("Add Data")

        if uploaded_file and add_data:
            with st.spinner("Reading, chunking and embedding file ..."):
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./files", uploaded_file.name)
                with open(file_name, "wb") as f:
                    f.write(bytes_data)
                data = load_document(file_name)
                chunks = create_chunks(data, chunk_size=chunk_size)
                st.write(f"Chunk size: {chunk_size}, Chunks: {len(chunks)}")

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f"Embedding cost: ${embedding_cost:.4f}")
                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success("File Uploaded, chunked and embedded successfully.")

    q = st.text_input("Ask a question about the content of your file")
    if q:
        if "vs" in st.session_state:
            vector_store = st.session_state.vs
            st.write(f"k: {k}")
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area("LLM Answer:", value=answer)
