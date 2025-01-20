import streamlit as st
import json
import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

###STRUCTURES
class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content = json.dumps(tool_result),
                    name = tool_call["name"],
                    tool_call_id = tool_call["id"],
                )
            )
        return {"messages": outputs}


###>>>---001 - Simple Assistant---<<<###
class SimpleAssistant:
    def __init__(self, framework, temperature_filter, model_name, shared_memory):
        self.shared_memory = shared_memory
        self.config = {
            "configurable": {"thread_id": "1"},
            "callbacks": [StreamlitCallbackHandler(st.container())]}
        self.llm_framework = {
            "Groq": ChatGroq,
            "Ollama": ChatOllama,
            "Google Generative AI": ChatGoogleGenerativeAI,
            "SambaNova": ChatSambaNovaCloud,
            "Scaleway": ChatOpenAI
        }
        self.llm_model = self.llm_framework[framework]
        if framework == "Scaleway":
            self.llm = ChatOpenAI(
                base_url = os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
                api_key = os.getenv("SCW_SECRET_KEY"),
                model = model_name,
                temperature =  temperature_filter
            )
        else:
            self.llm = self.llm_model(
                model = model_name,
                temperature = temperature_filter,
            )
        self.tool = DuckDuckGoSearchResults(output_format = "list")
        self.tools = [self.tool]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def load_model(self):
        ###GRAPH
        self.graph_builder = StateGraph(State)
        self.graph_builder.add_node("chatbot", self.chatbot)
        #self.tool_node = BasicToolNode(tools = [self.tool])
        self.tool_node = ToolNode(tools = [self.tool])
        self.graph_builder.add_node(
            "tools",
            self.tool_node
            )
        self.graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        self.graph_builder.add_edge("tools", "chatbot")
        self.graph_builder.add_edge(START, "chatbot")
        #self.graph_builder.add_edge("chatbot", END)
        self.graph = self.graph_builder.compile(
            checkpointer = self.shared_memory
        )

    def route_tools(
        self,
        state: State,
    ):
        """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    def chatbot(self, state: State):
        return {"messages": [self.llm.invoke(state["messages"])]}
    
    def stream_graph_updates(self, user_input: str):
        # The config is the **second positional argument** to stream() or invoke()!
        events = self.graph.stream(
            {"messages": [("user", user_input)]}, 
            self.config, 
            stream_mode = "values"
        )
        for i, event in enumerate(events):
            if i == 0:
                pass
            else:
                st.chat_message(
                    event["messages"][-1].type
                ).markdown(
                    event["messages"][-1].content
                )

