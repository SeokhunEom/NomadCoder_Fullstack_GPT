import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
import os

os.makedirs('./.cache/files/', exist_ok=True)
os.makedirs('./.cache/embeddings/', exist_ok=True)


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None

if "memory" not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(return_messages=True, memory_key="history")
memory = st.session_state['memory']


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message_box = None
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, openai_api_key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = UnstructuredFileLoader(file_path)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = loader.load_and_split(text_splitter=text_splitter)

    underlying_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    store = LocalFileStore(f"./.cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, store)

    db = FAISS.from_documents(documents, cached_embeddings)
    retriever = db.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def load_memory(_):
    return memory.load_memory_variables({})['history']


def save_memory(question, response):
    memory.save_context(
        {"input": question},
        {"output": response},
    )


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI that provides answer using only the following document. If you don't know the answer just say 'I don't know'. Don't forget to say 'I don't know' if you don't know the answer. Here is the document:
        ---
        {document}
        """
    ),
    ("system", "Now conversation started."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


with st.sidebar:
    file = st.file_uploader("Upload a file. (.txt, .pdf, .md, or .docx)", type=["txt", "pdf", "md", "docx"])
    st.session_state["openai_api_key"] = st.text_input("Enter a OpenAI API key")
    st.divider()
    st.code(
        '''
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None

if "memory" not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(return_messages=True, memory_key="history")
memory = st.session_state['memory']


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message_box = None
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, openai_api_key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = UnstructuredFileLoader(file_path)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = loader.load_and_split(text_splitter=text_splitter)

    underlying_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    store = LocalFileStore(f"./.cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, store)

    db = FAISS.from_documents(documents, cached_embeddings)
    retriever = db.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def load_memory(_):
    return memory.load_memory_variables({})['history']


def save_memory(question, response):
    memory.save_context(
        {"input": question},
        {"output": response},
    )


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI that provides answer using only the following document. If you don't know the answer just say 'I don't know'. Don't forget to say 'I don't know' if you don't know the answer. Here is the document:
        ---
        {document}
        """
    ),
    ("system", "Now conversation started."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


with st.sidebar:
    file = st.file_uploader("Upload a file. (.txt, .pdf, .md, or .docx)", type=["txt", "pdf", "md", "docx"])
    st.session_state["openai_api_key"] = st.text_input("Enter a OpenAI API key")
    st.divider()
    st.code(
        """
        print(hi)
        """,
        language="python",
        line_numbers=True,
    )
    st.caption("This is a github repository: https://github.com/SeokhunEom/NomadCoder_Fullstack_GPT")


st.title("DocumentGPT")

st.markdown(
    """
    Welcome!

    Use this chatbot to ask questions to an AI about your files!

    Upload your **file** and add your **OpenAI API key** on the sidebar.
    """
)


if file and st.session_state["openai_api_key"]:
    retriever = embed_file(file, st.session_state["openai_api_key"])
    llm = ChatOpenAI(temperature=0.1, streaming=True, openai_api_key=st.session_state["openai_api_key"], callbacks=[ChatCallbackHandler()])

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")
        chain = (
                {"document": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough(),}
                | RunnablePassthrough.assign(history=load_memory) | prompt | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
            save_memory(message, response.content)
else:
    st.session_state["messages"] = []
        ''',
        language="python",
        line_numbers=True,
    )
    st.caption("This is a github repository: https://github.com/SeokhunEom/NomadCoder_Fullstack_GPT")


st.title("DocumentGPT")

st.markdown(
    """
    Welcome!

    Use this chatbot to ask questions to an AI about your files!

    Upload your **file** and add your **OpenAI API key** on the sidebar.
    """
)


if file and st.session_state["openai_api_key"]:
    retriever = embed_file(file, st.session_state["openai_api_key"])
    llm = ChatOpenAI(temperature=0.1, streaming=True, openai_api_key=st.session_state["openai_api_key"], callbacks=[ChatCallbackHandler()])

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")
        chain = (
                {"document": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough(),}
                | RunnablePassthrough.assign(history=load_memory) | prompt | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
            save_memory(message, response.content)
else:
    st.session_state["messages"] = []