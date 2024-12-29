import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
import json
from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper


class EventHandler(AssistantEventHandler):
    message = ""

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    @override
    def on_text_delta(self, delta, snapshot):
        self.message += delta.value
        self.message_box.markdown(self.message.replace("$", "\$"))

    def on_event(self, event):
        if event.event == "thread.run.requires_action":
            submit_tool_outputs(event.data.id, event.data.thread_id)


def wikipediaSearchTool(arg):
    query = arg['query']
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)


def duckDuckGoSearchTool(arg):
    query = arg['query']
    ddg = DuckDuckGoSearchAPIWrapper()
    return ddg.run(query)


def webScrapeTool(arg):
    url = arg['url']
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()


functions_map = {
    "WikipediaSearchTool": wikipediaSearchTool,
    "DuckDuckGoSearchTool": duckDuckGoSearchTool,
    "WebScrapeTool": webScrapeTool,
}


def get_run(run_id, thread_id):
    return st.session_state["client"].beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return st.session_state["client"].beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = st.session_state["client"].beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    with st.session_state["client"].beta.threads.runs.submit_tool_outputs_stream(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


def insert_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def paint_history(thread_id):
    messages = get_messages(thread_id)
    for message in messages:
        insert_message(
            message.content[0].text.value,
            message.role,
        )


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None

if "client" not in st.session_state:
    st.session_state["client"] = None

if "thread" not in st.session_state:
    st.session_state["thread"] = None

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
)

st.title("AI Research Assistant")

st.markdown(
    """
    이 앱은 Wikipedia와 웹에서 정보를 검색하는 챗봇입니다.

    왼쪽 사이드바에 OpenAI API key를 입력하고 질문을 입력하면 챗봇이 답변을 제공합니다.
    """
)


with st.sidebar:
    st.session_state["openai_api_key"] = st.text_input("OpenAI API key", type="password")
    st.divider()
    st.code(
        '''
import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
import json
from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper


class EventHandler(AssistantEventHandler):
    message = ""

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    @override
    def on_text_delta(self, delta, snapshot):
        self.message += delta.value
        self.message_box.markdown(self.message.replace("$", "\$"))

    def on_event(self, event):
        if event.event == "thread.run.requires_action":
            submit_tool_outputs(event.data.id, event.data.thread_id)


def wikipediaSearchTool(arg):
    query = arg['query']
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)


def duckDuckGoSearchTool(arg):
    query = arg['query']
    ddg = DuckDuckGoSearchAPIWrapper()
    return ddg.run(query)


def webScrapeTool(arg):
    url = arg['url']
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()


functions_map = {
    "WikipediaSearchTool": wikipediaSearchTool,
    "DuckDuckGoSearchTool": duckDuckGoSearchTool,
    "WebScrapeTool": webScrapeTool,
}


def get_run(run_id, thread_id):
    return st.session_state["client"].beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return st.session_state["client"].beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = st.session_state["client"].beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    with st.session_state["client"].beta.threads.runs.submit_tool_outputs_stream(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


def insert_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def paint_history(thread_id):
    messages = get_messages(thread_id)
    for message in messages:
        insert_message(
            message.content[0].text.value,
            message.role,
        )


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None

if "client" not in st.session_state:
    st.session_state["client"] = None

if "thread" not in st.session_state:
    st.session_state["thread"] = None

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
)

st.title("AI Research Assistant")

st.markdown(
    """
    이 앱은 Wikipedia와 웹에서 정보를 검색하는 챗봇입니다.

    왼쪽 사이드바에 OpenAI API key를 입력하고 질문을 입력하면 챗봇이 답변을 제공합니다.
    """
)


with st.sidebar:
    st.session_state["openai_api_key"] = st.text_input("OpenAI API key", type="password")
    st.divider()
    st.code(
        """
        print("Hello, World!")
        """
    )
    st.caption("This is a github repository: https://github.com/SeokhunEom/NomadCoder_Fullstack_GPT")


if st.session_state["openai_api_key"]:
    if st.session_state["client"] is None:
        client = OpenAI(api_key=st.session_state["openai_api_key"])
        st.session_state["client"] = client

    if st.session_state["thread"] is None:
        st.session_state["thread"] = st.session_state["client"].beta.threads.create()

    paint_history(st.session_state["thread"].id)
    message = st.chat_input("질문을 입력하세요.")

    if message:
        send_message(st.session_state["thread"].id, message)
        insert_message(message, "user")

        with st.chat_message("assistant"):
            with st.session_state["client"].beta.threads.runs.stream(
                    thread_id=st.session_state["thread"].id,
                    assistant_id="asst_yQ2yXePUChKu0hcIjOzrr1dT",
                    event_handler=EventHandler(),
            ) as stream:
                stream.until_done()
else:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.session_state["messages"] = []
    st.session_state["client"] = None
    st.session_state["thread"] = None

        '''
    )
    st.caption("This is a github repository: https://github.com/SeokhunEom/NomadCoder_Fullstack_GPT")


if st.session_state["openai_api_key"]:
    if st.session_state["client"] is None:
        client = OpenAI(api_key=st.session_state["openai_api_key"])
        st.session_state["client"] = client

    if st.session_state["thread"] is None:
        st.session_state["thread"] = st.session_state["client"].beta.threads.create()

    paint_history(st.session_state["thread"].id)
    message = st.chat_input("질문을 입력하세요.")

    if message:
        send_message(st.session_state["thread"].id, message)
        insert_message(message, "user")

        with st.chat_message("assistant"):
            with st.session_state["client"].beta.threads.runs.stream(
                    thread_id=st.session_state["thread"].id,
                    assistant_id="asst_yQ2yXePUChKu0hcIjOzrr1dT",
                    event_handler=EventHandler(),
            ) as stream:
                stream.until_done()
else:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.session_state["messages"] = []
    st.session_state["client"] = None
    st.session_state["thread"] = None
