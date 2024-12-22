import streamlit as st
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.schema.runnable import RunnablePassthrough


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

create_quiz_function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


if "quiz_data" not in st.session_state:
    st.session_state["quiz_data"] = None
if "is_submitted" not in st.session_state:
    st.session_state["is_submitted"] = False
if "user_score" not in st.session_state:
    st.session_state["user_score"] = 0
if "total_score" not in st.session_state:
    st.session_state["total_score"] = 0


prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    Difficulty: {difficulty}
    
    Quiz Language: {language}
    
    Use the following context to create the questions:

    ---
    Context: {context}
    ---
    """
)


def reset_quiz():
    st.session_state["quiz_data"] = None
    st.session_state["is_submitted"] = False
    st.session_state["user_score"] = 0
    st.session_state["total_score"] = 0


def foramt_document(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="관련 위키피디아 검색 중...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


@st.cache_data(show_spinner="퀴즈 생성 중..")
def make_quiz(topic, difficulty, language, openai_api_key):
    docs = wiki_search(topic)

    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        openai_api_key=openai_api_key,
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[create_quiz_function],
    )

    chain = {"context": foramt_document} | RunnablePassthrough.assign(difficulty=lambda x: difficulty) | RunnablePassthrough.assign(language=lambda x: language) | prompt | llm
    response = chain.invoke(docs)

    arguments = json.loads(response.additional_kwargs["function_call"]["arguments"])
    return arguments


with st.sidebar:
    st.title("QuizGPT 설정")
    topic = st.text_input("퀴즈 주제")
    difficulty = st.selectbox("퀴즈 난이도", ["쉬움", "어려움"])
    language = st.selectbox("퀴즈 언어", ["한국어", "영어"])
    openai_api_key = st.text_input("OpenAI API key", type="password")
    st.divider()
    st.code(
        '''
        import streamlit as st
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.schema.runnable import RunnablePassthrough


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

create_quiz_function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


if "quiz_data" not in st.session_state:
    st.session_state["quiz_data"] = None
if "is_submitted" not in st.session_state:
    st.session_state["is_submitted"] = False
if "user_score" not in st.session_state:
    st.session_state["user_score"] = 0
if "total_score" not in st.session_state:
    st.session_state["total_score"] = 0


prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    Difficulty: {difficulty}
    
    Quiz Language: {language}
    
    Use the following context to create the questions:

    ---
    Context: {context}
    ---
    """
)


def reset_quiz():
    st.session_state["quiz_data"] = None
    st.session_state["is_submitted"] = False
    st.session_state["user_score"] = 0
    st.session_state["total_score"] = 0


def foramt_document(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="관련 위키피디아 검색 중...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


@st.cache_data(show_spinner="퀴즈 생성 중..")
def make_quiz(topic, difficulty, language, openai_api_key):
    docs = wiki_search(topic)

    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        openai_api_key=openai_api_key,
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[create_quiz_function],
    )

    chain = {"context": foramt_document} | RunnablePassthrough.assign(difficulty=lambda x: difficulty) | RunnablePassthrough.assign(language=lambda x: language) | prompt | llm
    response = chain.invoke(docs)

    arguments = json.loads(response.additional_kwargs["function_call"]["arguments"])
    return arguments


with st.sidebar:
    st.title("QuizGPT 설정")
    topic = st.text_input("퀴즈 주제")
    difficulty = st.selectbox("퀴즈 난이도", ["쉬움", "어려움"])
    language = st.selectbox("퀴즈 언어", ["한국어", "영어"])
    openai_api_key = st.text_input("OpenAI API key", type="password")
    st.divider()
    st.code(
        """
        print("QuizGPT")
        """)
    st.caption("This is a github repository: https://github.com/SeokhunEom/NomadCoder_Fullstack_GPT")


st.title("QuizGPT")


st.markdown(
    """
    QuizGPT은 위키백과에서 정보를 가져와서 퀴즈를 생성하는 서비스입니다.
    
    퀴즈를 생성하려면 왼쪽 사이드바에 퀴즈 주제와 난이도를 선택하고, OpenAI API key를 입력해주세요.
    """
)


if st.button("Generate Quiz"):
    reset_quiz()
    if topic and openai_api_key:
        quiz_data = make_quiz(topic, difficulty, language, openai_api_key)
        if quiz_data is not None:
            st.session_state["quiz_data"] = quiz_data
        else:
            st.write("해당 주제에 대한 정보를 찾을 수 없습니다.")
    else:
        st.write("주제와 API 키를 모두 입력해주세요.")


if st.session_state["quiz_data"]:
    with st.form("quiz_form"):
        questions = st.session_state["quiz_data"]["questions"]
        st.session_state["total_score"] = len(questions)

        for i, q in enumerate(questions):
            st.write(f"**문제 {i+1}.** {q['question']}")
            selected_answer = st.radio(
                f"문제 {i+1}에 대한 답을 선택하세요",
                [ans["answer"] for ans in q["answers"]],
                key=f"question_{i}",
            )
        submit_button = st.form_submit_button("제출하기")
        if submit_button:
            st.session_state["is_submitted"] = True
            score = 0
            for i, q in enumerate(questions):
                selected_answer = st.session_state[f"question_{i}"]
                for ans in q["answers"]:
                    if ans["answer"] == selected_answer and ans["correct"]:
                        score += 1
            st.session_state["user_score"] = score

    if st.session_state["is_submitted"]:
        st.write(f'점수: {st.session_state["user_score"]} / {st.session_state["total_score"]}')
        if st.session_state["user_score"] == st.session_state["total_score"]:
            st.balloons()
            st.success("축하합니다! 만점입니다.")
        else:
            if st.button("Try Again"):
                st.session_state["is_submitted"] = False
                st.session_state["user_score"] = 0
                for i in range(st.session_state["total_score"]):
                    st.session_state[f"question_{i}"] = None
        ''')
    st.caption("This is a github repository: https://github.com/SeokhunEom/NomadCoder_Fullstack_GPT")


st.title("QuizGPT")


st.markdown(
    """
    QuizGPT은 위키백과에서 정보를 가져와서 퀴즈를 생성하는 서비스입니다.
    
    퀴즈를 생성하려면 왼쪽 사이드바에 퀴즈 주제와 난이도를 선택하고, OpenAI API key를 입력해주세요.
    """
)


if st.button("Generate Quiz"):
    reset_quiz()
    if topic and openai_api_key:
        quiz_data = make_quiz(topic, difficulty, language, openai_api_key)
        if quiz_data is not None:
            st.session_state["quiz_data"] = quiz_data
        else:
            st.write("해당 주제에 대한 정보를 찾을 수 없습니다.")
    else:
        st.write("주제와 API 키를 모두 입력해주세요.")


if st.session_state["quiz_data"]:
    with st.form("quiz_form"):
        questions = st.session_state["quiz_data"]["questions"]
        st.session_state["total_score"] = len(questions)

        for i, q in enumerate(questions):
            st.write(f"**문제 {i+1}.** {q['question']}")
            selected_answer = st.radio(
                f"문제 {i+1}에 대한 답을 선택하세요",
                [ans["answer"] for ans in q["answers"]],
                key=f"question_{i}",
            )
        submit_button = st.form_submit_button("제출하기")
        if submit_button:
            st.session_state["is_submitted"] = True
            score = 0
            for i, q in enumerate(questions):
                selected_answer = st.session_state[f"question_{i}"]
                for ans in q["answers"]:
                    if ans["answer"] == selected_answer and ans["correct"]:
                        score += 1
            st.session_state["user_score"] = score

    if st.session_state["is_submitted"]:
        st.write(f'점수: {st.session_state["user_score"]} / {st.session_state["total_score"]}')
        if st.session_state["user_score"] == st.session_state["total_score"]:
            st.balloons()
            st.success("축하합니다! 만점입니다.")
        else:
            if st.button("Try Again"):
                st.session_state["is_submitted"] = False
                st.session_state["user_score"] = 0
                for i in range(st.session_state["total_score"]):
                    st.session_state[f"question_{i}"] = None