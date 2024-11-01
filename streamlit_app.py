import streamlit as st

import os
from typing import Sequence

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain.tools.retriever import create_retriever_tool
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFaceEndpoint
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

# Show title and description.
st.title("📄 FSC 111 Slide revision")

tavily_api_key="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"
os.environ["GROQ_API_KEY"] = "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"
os.environ['TAVILY_API_KEY'] = tavily_api_key

llm = ChatGroq(model="llama3-8b-8192")

huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

### Construct retriever ###
loader = PyPDFLoader("Biogeochemical_cycles.pdf")
documents1 = loader.load()
loader = PyPDFLoader("Properties_of_life-Order_and_Metabolism.pdf")
documents2 = loader.load()
loader = PyPDFLoader("Classification and Evolution of Living things 1.pdf")
documents3 = loader.load()
loader = PyPDFLoader("Classification and Evolution of Living things 2.pdf")
documents4 = loader.load()
loader = PyPDFLoader("FSC 111 Mitosis and Meiosis Class 9th.pptx.pdf")
documents5 = loader.load()
loader = PyPDFLoader("FSC111 -  Unifying Themes in Biology.pdf")
documents6 = loader.load()
loader = PyPDFLoader("FSC111 - Cell Structure and Function (1) 4th.pdf")
documents7 = loader.load()
loader = PyPDFLoader("FSC111 - Cell Structure and Function 4th.pdf")
documents8 = loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, add_start_index=True)
splits1=text_splitter.split_documents(documents1)
splits2=text_splitter.split_documents(documents2)
splits3=text_splitter.split_documents(documents3)
splits4=text_splitter.split_documents(documents4)
splits5=text_splitter.split_documents(documents5)
splits6=text_splitter.split_documents(documents6)
splits7=text_splitter.split_documents(documents7)
splits8=text_splitter.split_documents(documents8)
all_splits=splits1+splits2+splits3+splits4+splits5+splits6+splits7+splits8

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(all_splits)
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=huggingface_embeddings
)
retriever = vectorstore.as_retriever()

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Be concise"
    "When providing answer to a question from the user, use the retrieved context and outside pieces of information to provide a good answer to the question"
    "Always double-check information with multiple sources to ensure accuracy"

    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }


workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "abc123"}}


### Chat Interface ###

st.title("FSC111 Slide revision")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

response = app.invoke(
    {"input": prompt},
    config=config,
)
# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response["answer"])
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
