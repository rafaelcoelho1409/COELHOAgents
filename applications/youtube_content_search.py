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
from models.youtube_content_search import YouTubeContentSearch


initialize_shared_memory()


model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()


with st.sidebar.form("Project Settings"):
    max_results = st.number_input(
        label = "Maximum Results",
        min_value = 1,
        max_value = 10,
        value = 5
    )
    submit_project_settings = st.form_submit_button(
        "Set maximum results",
        use_container_width = True)
if submit_project_settings:
    st.session_state["max_results"] = max_results


try:
    max_results = st.session_state["max_results"]
except:
    st.info("Set a maximum number of results for YouTube searches.")
    st.stop()


role = YouTubeContentSearch(
    st.session_state["framework"],
    st.session_state["temperature_filter"], 
    st.session_state["model_name"],
    st.session_state["shared_memory"]
)
role.load_model(st.session_state["max_results"])


view_graph = st.session_state["view_graph_button_container"].button(
    label = "View application graph",
    use_container_width = True,
)
if view_graph:
    view_application_graph(role.graph)


snapshot = role.graph.get_state(role.config)
#for msg in st.session_state["history"].messages:
try:
    for msg in snapshot.values["messages"]:
    #for msg in st.session_state["history"].messages:
        st.chat_message(msg.type).write(msg.content)
except:
    pass


if prompt := st.chat_input():
    if st.session_state["memory_filter"] == False:
        st.session_state["shared_memory"] = MemorySaver()
    role.stream_graph_updates(
        prompt)