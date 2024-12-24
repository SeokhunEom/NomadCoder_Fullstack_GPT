import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
import os


os.makedirs('./.cache/cloudflare_embeddings/', exist_ok=True)


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


def parse_page(soup):
    main = soup.find("main")
    text = main.get_text() if main else soup.get_text()
    return text.replace("\n", " ").replace("\xa0", " ")


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


answers_prompt = ChatPromptTemplate.from_template(
    """
    Use ONLY the following context to answer the user's question.
    If you can't answer from the context, just say "I don't know".
    Do not fabricate any information.

    Then, provide a score from 0 to 5 for your answer:
      - 5: Very accurate answer
      - 0: No answer at all or complete irrelevance

    Include the score at the end of your response, even if it's 0.

    Context:
    {context}

    Question: {question}

    Examples:
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
    """
)


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to respond to the user's question.
            Select the answers with the highest score and prefer the most recent ones.
            Cite sources exactly as they appear. Do not modify them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]

    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        openai_api_key=st.session_state["openai_api_key"],
    )
    answers_chain = answers_prompt | llm

    answers = []
    for doc in docs:
        result = answers_chain.invoke(
            {"question": question, "context": doc.page_content}
        )
        answers.append(
            {
                "answer": result.content,
                "source": doc.metadata["source"],
                "date": doc.metadata.get("lastmod", "No date"),
            }
        )

    return {
        "question": question,
        "answers": answers,
    }


def choose_answer(inputs):
    question = inputs["question"]
    answers = inputs["answers"]

    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        openai_api_key=st.session_state["openai_api_key"],
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )
    choose_chain = choose_prompt | llm

    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


@st.cache_data(show_spinner="Cloudflare 문서 로딩중...")
def load_docs():
    loader = SitemapLoader(
        "https://developers.cloudflare.com/sitemap-0.xml",
        filter_urls=[
            'https://developers.cloudflare.com/ai-gateway',
            'https://developers.cloudflare.com/vectorize',
            'https://developers.cloudflare.com/workers-ai',
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = loader.load_and_split(text_splitter=splitter)

    underlying_embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["openai_api_key"])
    store = LocalFileStore(f"./.cache/cloudflare_embeddings")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, store)

    db = FAISS.from_documents(documents, cached_embeddings)
    retriever = db.as_retriever()
    return retriever



if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.title("Cloudflare SiteGPT")

st.markdown(
    """
    이 앱은 Cloudflare의 AI Gateway, Vectorize, Workers AI 제품에 대한 질문을 할 수 있는 챗봇입니다.
    
    왼쪽 사이드바에 OpenAI API key를 입력하고 질문을 입력하면 챗봇이 답변을 제공합니다.
    """
)


with st.sidebar:
    st.session_state["openai_api_key"] = st.text_input("OpenAI API key", type="password")
    st.divider()
    st.code(
        '''
        import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
import os


os.makedirs('./.cache/cloudflare_embeddings/', exist_ok=True)


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


def parse_page(soup):
    main = soup.find("main")
    text = main.get_text() if main else soup.get_text()
    return text.replace("\n", " ").replace("\xa0", " ")


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


answers_prompt = ChatPromptTemplate.from_template(
    """
    Use ONLY the following context to answer the user's question.
    If you can't answer from the context, just say "I don't know".
    Do not fabricate any information.

    Then, provide a score from 0 to 5 for your answer:
      - 5: Very accurate answer
      - 0: No answer at all or complete irrelevance

    Include the score at the end of your response, even if it's 0.

    Context:
    {context}

    Question: {question}

    Examples:
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
    """
)


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to respond to the user's question.
            Select the answers with the highest score and prefer the most recent ones.
            Cite sources exactly as they appear. Do not modify them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]

    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        openai_api_key=st.session_state["openai_api_key"],
    )
    answers_chain = answers_prompt | llm

    answers = []
    for doc in docs:
        result = answers_chain.invoke(
            {"question": question, "context": doc.page_content}
        )
        answers.append(
            {
                "answer": result.content,
                "source": doc.metadata["source"],
                "date": doc.metadata.get("lastmod", "No date"),
            }
        )

    return {
        "question": question,
        "answers": answers,
    }


def choose_answer(inputs):
    question = inputs["question"]
    answers = inputs["answers"]

    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        openai_api_key=st.session_state["openai_api_key"],
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )
    choose_chain = choose_prompt | llm

    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


@st.cache_data(show_spinner="Cloudflare 문서 로딩중...")
def load_docs():
    loader = SitemapLoader(
        "https://developers.cloudflare.com/sitemap-0.xml",
        filter_urls=[
            'https://developers.cloudflare.com/ai-gateway',
            'https://developers.cloudflare.com/vectorize',
            'https://developers.cloudflare.com/workers-ai',
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = loader.load_and_split(text_splitter=splitter)

    underlying_embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["openai_api_key"])
    store = LocalFileStore(f"./.cache/cloudflare_embeddings")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, store)

    db = FAISS.from_documents(documents, cached_embeddings)
    retriever = db.as_retriever()
    return retriever



if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.title("Cloudflare SiteGPT")

st.markdown(
    """
    이 앱은 Cloudflare의 AI Gateway, Vectorize, Workers AI 제품에 대한 질문을 할 수 있는 챗봇입니다.
    
    왼쪽 사이드바에 OpenAI API key를 입력하고 질문을 입력하면 챗봇이 답변을 제공합니다.
    """
)


with st.sidebar:
    st.session_state["openai_api_key"] = st.text_input("OpenAI API key", type="password")
    st.divider()
    st.code(
        """
        print("SiteGPT")
        """
    )
    st.caption("This is a github repository: https://github.com/SeokhunEom/NomadCoder_Fullstack_GPT")


if st.session_state["openai_api_key"]:
    retriever = load_docs()

    send_message("Cloudflare의 AI Gateway, Vectorize, Workers AI 제품에 대한 질문을 해보세요!.", "ai", save=False)
    paint_history()
    message = st.chat_input("궁금한것을 질문하세요.")

    if message:
        send_message(message, "human")
        chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []

        '''
    )
    st.caption("This is a github repository: https://github.com/SeokhunEom/NomadCoder_Fullstack_GPT")


if st.session_state["openai_api_key"]:
    retriever = load_docs()

    send_message("Cloudflare의 AI Gateway, Vectorize, Workers AI 제품에 대한 질문을 해보세요!.", "ai", save=False)
    paint_history()
    message = st.chat_input("궁금한것을 질문하세요.")

    if message:
        send_message(message, "human")
        chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []
