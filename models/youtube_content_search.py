import streamlit as st
import stqdm
import os
import numpy as np
from dotenv import load_dotenv
from typing import List, Annotated, Dict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START


load_dotenv()
#------------------------------------------------
###STRUCTURES
class SearchQueries(BaseModel):
    search_queries: List = Field(
        "A list of accurate search queries for YouTube videos.",
        title = "Search Queries",
        description = "Search queries for YouTube videos.",
        example = "How to make a cake"
    )

class State(TypedDict):
    error: str
    messages: List
    streamlit_actions: List
    user_input: str
    queries_results: List
    unique_videos: List
#------------------------------------------------

class YouTubeContentSearch:
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

    def load_model(self, max_results):
        self.max_results = max_results
        self.youtube_search_agent = self.build_youtube_search_agent()
        self.workflow = StateGraph(State)
        ###NODES
        self.workflow.add_node("search_youtube_videos", self.search_youtube_videos)
        ###EDGES
        self.workflow.add_edge(START, "search_youtube_videos")
        self.workflow.add_edge("search_youtube_videos", END)
        self.graph = self.workflow.compile(
            checkpointer = st.session_state["shared_memory"]#self.shared_memory
        )
    
    ###AGENTS
    def build_youtube_search_agent(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a YouTube search agent.\n
                    Based on the following prompt:\n\n
                    {user_input}\n\n
                    You must take this user input and transform it into 
                    the most efficient youtube search queries you can.\n
                    It must be in a list format.\n
                    """,
                ),
                ("placeholder", "{messages}"),
            ]
        )
        chain = prompt | self.llm.with_structured_output(SearchQueries)
        return chain

    #------------------------------------------------
    ###TOOLS

    #------------------------------------------------
    ###NODES
    def search_youtube_videos(self, state: State):
        messages = state["messages"]
        streamlit_actions = state["streamlit_actions"]
        user_input = state["user_input"]
        streamlit_action = []
        youtube_search_queries = self.youtube_search_agent.invoke({
            "user_input": user_input
        })
        search_queries = youtube_search_queries.search_queries
        queries_results = {}
        for query in stqdm.stqdm(search_queries, desc = "Searching YouTube videos"):
            results = YoutubeSearch(
                query, 
                max_results = self.max_results).to_dict()
            results = [{
                "title": video["title"], 
                "id": video["id"], 
                "publish_time": video["publish_time"],
                "duration": video["duration"],
                "views": video["views"],
                "channel": video["channel"]}
                for video 
                in results]
            queries_results[query] = results
        messages += [
            (
                "assistant",
                list(queries_results.keys())
            )
        ]
        streamlit_action += [(
            "json", 
            {"body": messages[-1][1], "expanded": False},
            ("Youtube search queries", False),
            messages[-1][0],
            )]
        unique_videos = []
        videos = [item for sublist in queries_results.values() for item in sublist]
        for video in videos:
            if video not in unique_videos:
                unique_videos.append(video)
        messages += [
            (
                "assistant",
                unique_videos
            )
        ]
        streamlit_action += [(
            "json", 
            {"body": messages[-1][1], "expanded": False},
            ("Youtube videos searched", False),
            messages[-1][0],
            )]
        streamlit_actions += [streamlit_action]
        return {
            "messages": messages,
            "streamlit_actions": streamlit_actions,
            "queries_results": queries_results,
            "unique_videos": unique_videos
        }

    #------------------------------------------------
    def stream_graph_updates(self, user_input):
        # The config is the **second positional argument** to stream() or invoke()!
        events = self.graph.stream(
            {
                "messages": [("user", user_input)], 
                "streamlit_actions": [[(
                    "markdown", 
                    {"body": user_input},
                    ("User request", True),
                    "user"
                    )]],
                "error": "",
                "user_input": user_input,
                "queries_results": {}
                },
            self.config, 
            stream_mode = "values"
        )
        for event in events:
            actions = event["streamlit_actions"][-1]
            if actions != []:
                for action in actions:
                    st.chat_message(
                        action[3]
                    ).expander(
                        action[2][0], 
                        expanded = action[2][1]
                    ).__getattribute__(
                        action[0]
                    )(
                        **action[1]
                    )