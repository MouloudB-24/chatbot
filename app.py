import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain



OPENAI_API_KEY = ""

# Use interface
st.header("Intelligent Chatbot with generative IA")
with st.sidebar:
    st.title("Your documents")
    files = st.file_uploader("Download a PDF files", type="pdf", accept_multiple_files=True)

if files:
    st.write("File(s) uploaded successfully!")
    for file in files:
        st.write(f"- {file.name}")
else:
    st.write("Please upload a PDF file to get started!")


if files:
    # Extract text from PDF
    text = ""
    for file in files:
        pdf_reader = PdfReader(file)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                st.warning("One or more pages of the document cannot be extract!")

    # Text division into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Genera te embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create FAISS database
    vector_db = FAISS.from_texts(chunks, embeddings)

    # Champ pour poser la question
    user_query = st.text_input("Ask your question here:")

    # Search for chunks similar to th user's question
    if user_query:
        results = vector_db.similarity_search(user_query, top_k=2)

        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=100,
            model_name="gpt-3.5-turbo"
        )

        # Loading the question-and-answer chain
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_document=results, question=user_query)

        # Display response
        st.write("Réponse générée:")
        st.write(response)


        # Download response
        st.download_button(
            label="Download response",
            data=response,
            file_name="response.txt",
            mime="text/plain"
        )





