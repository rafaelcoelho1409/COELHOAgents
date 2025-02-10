import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
import os
import json
from pathlib import Path
from functions import (
    check_model_and_temperature,
    initialize_shared_memory,
    view_application_graph
)
from models.youtube_content_search import YouTubeContentSearch, YouTubeChatbot


initialize_shared_memory()


model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()


with st.sidebar.form("Project Settings"):
    max_results = st.number_input(
        label = "Maximum Results",
        min_value = 1,
        #max_value = 10,
        step = 1,
        value = 1
    )
    context_to_search = st.text_area(
        label = "Context to search",
        placeholder = "Provide the context to be searched on YouTube",
    )
    submit_project_settings = st.form_submit_button(
        "Set and search",
        use_container_width = True)
if submit_project_settings:
    st.session_state["max_results"] = max_results
    st.session_state["context_to_search"] = context_to_search
    st.session_state["youtube_content_search_agent"] = YouTubeContentSearch(
        st.session_state["framework"],
        st.session_state["temperature_filter"], 
        st.session_state["model_name"],
        st.session_state["shared_memory"]
    )
    st.session_state["youtube_content_search_agent"].load_model(st.session_state["max_results"])
    if st.session_state["memory_filter"] == False:
        st.session_state["shared_memory"] = MemorySaver()
    st.session_state["youtube_content_search_agent"].stream_graph_updates(
        context_to_search)
    st.session_state["snapshot"] = st.session_state["youtube_content_search_agent"].graph.get_state(
        st.session_state["youtube_content_search_agent"].config)


try:
    max_results = st.session_state["max_results"]
    context_to_search = st.session_state["context_to_search"]
except:
    st.info("Set a maximum number of results and a context for YouTube searches.")
    st.stop()


chatbot_agent = YouTubeChatbot(
    st.session_state["framework"],
    st.session_state["temperature_filter"],
    st.session_state["model_name"],
    st.session_state["shared_memory"]
)
chatbot_agent.load_model(st.session_state["youtube_content_search_agent"].rag_chain)


view_graph = st.session_state["view_graph_button_container"].button(
    label = "View application graph",
    use_container_width = True,
)
if view_graph:
    view_application_graph(st.session_state["youtube_content_search_agent"].graph)


st.session_state["snapshot"] += chatbot_agent.graph.get_state(chatbot_agent.config)
messages_block = [x for i, x in enumerate(st.session_state["snapshot"]) if i % 7 == 0]
if not submit_project_settings:
    try:
        for actions in messages_block[-1]["streamlit_actions"]:
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
    except:
        pass


if prompt := st.chat_input():
    if st.session_state["memory_filter"] == False:
        st.session_state["shared_memory"] = MemorySaver()
    chatbot_agent.stream_graph_updates(
        prompt)