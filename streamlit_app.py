
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


# Show title and description.
st.title("📄 FSC 111 Slide revision")
huggingface_api_key="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"

import os
os.environ['HUGGINGFACEHUB_API_TOKEN']=huggingface_api_key

### Construct retriever ###
loader = PyPDFLoader("/content/drive/MyDrive/Project/Training set/Biogeochemical_cycles.pdf")
documents1 = loader.load()
loader = PyPDFLoader("/content/drive/MyDrive/Project/Training set/Properties_of_life-Order_and_Metabolism.pdf")
documents2 = loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, add_start_index=True)
splits1=text_splitter.split_documents(documents1)
splits2=text_splitter.split_documents(documents2)
all_splits=splits1+splits2

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
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
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

def rag_output(question):
    result = app.invoke(
    {"input": question},
    config=config,
    )
    return result

def chat_history():
    chat_history = app.get_state(config).values["chat_history"]
    for message in chat_history:
        message.pretty_print()

# Ask the user for a question via `st.text_area`.
query=st.text_area(
    "Now ask a question about FSC 111",
)

submit_button=st.button("Ask")
if submit_button:
    st.write(rag_output(question)['answer']

history_button=st.button("Chat History")
if history_button:
    st.write(chat_history())
