import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment variables setup
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

# Initialize LLM and embeddings
llm = ChatGroq(model="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)

# Text splitter and prompt
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def create_prompt(question: str, context: str) -> str:
    return f"""
You are an intelligent CSV data assistant. Your task is to answer questions based on the given CSV data context. Follow these guidelines:

1. Read the provided context carefully.
2. If the question requires calculations, perform them with accuracy.
3. If the question involves data visualization, suggest an appropriate graph type (e.g., bar chart, line chart, pie chart, etc.) and describe it clearly.
4. Provide a concise and accurate answer to the question.

Context:
{context}

Question: {question}

Answer:"""

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
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    formatted_prompt = create_prompt(state["question"], docs_content)
    response = llm.invoke(formatted_prompt)
    answer = response.content

    # Check if graph needs to be generated
    if "graph" in answer.lower():
        try:
            csv_path = os.path.join(CSV_STORAGE_DIR, selected_csv)
            df = pd.read_csv(csv_path)

            # Generate a sample graph
            fig, ax = plt.subplots()
            df.plot(kind="bar", ax=ax)  # Modify graph type based on need
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating graph: {e}")
    
    return {"answer": answer}

# Compile the application state graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Ensure storage directory exists
CSV_STORAGE_DIR = "csv_storage"
os.makedirs(CSV_STORAGE_DIR, exist_ok=True)

# Streamlit UI
st.title("CSV Q&A App with Visualization")
st.sidebar.title("CSV Management and Q&A")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
if uploaded_file:
    file_path = os.path.join(CSV_STORAGE_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"File '{uploaded_file.name}' saved!")

# List saved CSVs
saved_csvs = [f for f in os.listdir(CSV_STORAGE_DIR) if f.endswith(".csv")]
selected_csv = st.sidebar.selectbox("Select a CSV to process:", ["None"] + saved_csvs)

if selected_csv != "None":
    st.text(f"{selected_csv} file selected.")
    file_path = os.path.join(CSV_STORAGE_DIR, selected_csv)
    loader = CSVLoader(file_path)
    docs = loader.load()
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)
    st.sidebar.success(f"CSV '{selected_csv}' processed and indexed!")

# Q&A Section
if "question_history" not in st.session_state:
    st.session_state.question_history = []

if selected_csv != "None":
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
