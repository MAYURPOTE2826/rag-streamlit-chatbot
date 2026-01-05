import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import DocArrayInMemorySearch

from langchain_google_genai import ChatGoogleGenerativeAI


# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Multi-PDF RAG Chatbot", layout="wide")
st.title("📚 Multi-PDF RAG Chatbot")


# -----------------------------
# Read Gemini API Key
# -----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# -----------------------------
# Utils
# -----------------------------
def format_chat_history(chat_history):
    text = ""
    for msg in chat_history:
        if msg["role"] == "user":
            text += f"You: {msg['content']}\n"
        else:
            text += f"AI: {msg['content']}\n"
    return text


def answer_with_gemini(question, retriever, llm):
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(
        [doc.page_content[:1500] for doc in docs[:3]]
    )

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content, docs


# -----------------------------
# Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# -----------------------------
# Sidebar: Upload PDFs
# -----------------------------
st.sidebar.header("📄 Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.success("Processing PDFs...")

    all_docs = []

    for uploaded_file in uploaded_files:
        file_path = f"temp_{uploaded_file.name}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

    # -----------------------------
    # Split Text
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(all_docs)
    texts = [doc.page_content for doc in chunks]

    # -----------------------------
    # Local Embeddings
    # -----------------------------
    model = SentenceTransformer("all-MiniLM-L6-v2")

    class LocalEmbeddingFunction:
        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts):
            return self.model.encode(texts).tolist()

        def embed_query(self, text):
            return self.model.encode([text]).tolist()[0]

    embeddings = LocalEmbeddingFunction(model)

    vectorstore = DocArrayInMemorySearch.from_texts(
        texts=texts,
        embedding=embeddings
    )

    # -----------------------------
    # Gemini LLM (SAFE)
    # -----------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    st.session_state.llm = llm
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.sidebar.success("✅ All PDFs processed!")


# -----------------------------
# Chat Interface
# -----------------------------
st.subheader("Ask questions across ALL PDFs")

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"🧑 **You:** {chat['content']}")
    else:
        st.markdown(f"🤖 **AI:** {chat['content']}")

user_input = st.text_input("Ask a question:")

if user_input and st.session_state.retriever:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    with st.spinner("Thinking..."):
        answer, sources = answer_with_gemini(
            user_input,
            st.session_state.retriever,
            st.session_state.llm
        )

    st.session_state.chat_history.append(
        {"role": "ai", "content": answer}
    )

    st.markdown("### 🤖 Answer")
    st.write(answer)

    st.markdown("### 📌 Sources")
    for i, doc in enumerate(sources[:2]):
        with st.expander(f"Source {i + 1}"):
            st.write(doc.page_content)

    st.rerun()


# -----------------------------
# Clear Chat
# -----------------------------
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()


# -----------------------------
# Save Chat History
# -----------------------------
st.markdown("---")
st.subheader("💾 Save Chat History")

if st.session_state.chat_history:
    chat_text = format_chat_history(st.session_state.chat_history)

    st.download_button(
        label="⬇️ Download Chat History",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )
else:
    st.info("No chat history to save yet.")
