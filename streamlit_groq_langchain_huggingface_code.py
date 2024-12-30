import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.api_config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = st.secrets.api_config.LANGCHAIN_API_KEY

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.secrets.api_config.GROQ_API_KEY

# Initialize LLM and embeddings
llm = ChatGroq(model="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)

# Text splitter and prompt
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
prompt = hub.pull("rlm/rag-prompt")

# Define state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    # Retrieve the conversation history
    chat_history = st.session_state.question_history
    formatted_history = "\n".join(
        f"User: {q}\nAssistant: {a}" for q, a in chat_history
    )
    
    # Extract document content
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Create a formatted prompt
    formatted_prompt = (
        "You are an intelligent assistant helping a user based on their questions and context extracted from documents. "
        "Here is the conversation so far:\n\n"
        f"{formatted_history}\n\n"
        "And here is the relevant context from the document:\n\n"
        f"{docs_content}\n\n"
        "Based on the above, answer the user's next question in a concise and informative manner. "
        "If the user's question is already addressed in the conversation, refer to the prior response without repeating the full answer."
        "\n\n"
        f"User's Question: {state['question']}\n\n"
        "Answer:"
    )
    
    # Invoke the LLM with the prompt
    response = llm.invoke(formatted_prompt)
    return {"answer": response.content}

# Compile the application state graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Ensure storage directory exists
PDF_STORAGE_DIR = "pdf_storage"
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# Streamlit UI
st.title("PDF Q&A App")
st.sidebar.title("PDF Management and Q&A")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    file_path = os.path.join(PDF_STORAGE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"File '{uploaded_file.name}' saved!")

# List saved PDFs
saved_pdfs = [f for f in os.listdir(PDF_STORAGE_DIR) if f.endswith(".pdf")]
selected_pdf = st.sidebar.selectbox("Select a PDF to process:", ["None"] + saved_pdfs)

if selected_pdf != "None":
    st.text(f"{selected_pdf} file selected.")
    file_path = os.path.join(PDF_STORAGE_DIR, selected_pdf)
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)
    st.sidebar.success(f"PDF '{selected_pdf}' processed and indexed!")

# Delete a PDF
st.sidebar.write("---")
delete_pdf = st.sidebar.selectbox("Select a PDF to delete:", ["None"] + saved_pdfs)

if delete_pdf != "None":
    if st.sidebar.button("Delete Selected PDF"):
        os.remove(os.path.join(PDF_STORAGE_DIR, delete_pdf))
        st.sidebar.success(f"PDF '{delete_pdf}' deleted!")
        # Refresh the list of PDFs
        saved_pdfs = [f for f in os.listdir(PDF_STORAGE_DIR) if f.endswith(".pdf")]
        
# Q&A Section
if "question_history" not in st.session_state:
    st.session_state.question_history = []

if selected_pdf != "None":
    question = st.text_input("Ask a question:")
    if question:
        try:
            response = graph.invoke({"question": question})
            answer = response["answer"]
            st.session_state.question_history.append((question, answer))
            st.success(f"""Answer: {answer}""")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display question history
if st.session_state.question_history:
    st.subheader("Question History")
    for idx, (q, a) in enumerate(st.session_state.question_history, start=1):
        st.write(f"**Q{idx}: {q}**")
        st.write(f"**A{idx}: {a}**")
