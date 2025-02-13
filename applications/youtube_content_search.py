import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
import uuid
import numpy as np
from functions import (
    check_model_and_temperature,
    initialize_shared_memory,
    view_application_graphs,
    view_neo4j_context_graph
)
from models.youtube_content_search import (
    YouTubeContentSearch, 
    YouTubeChatbot
)


initialize_shared_memory()


model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()


search_type_filter = st.sidebar.selectbox(
    label = "Search type",
    options = [
        "Search",
        "Video", 
        "Channel", 
        "Playlist"
        ],
    index = 0
)

if search_type_filter == "Search":
    with st.sidebar.form(f"Project Settings - {search_type_filter}"):
        context_to_search = st.text_area(
            label = "Context to search",
            placeholder = "Provide the context to be searched on YouTube",
        )
        max_results = st.number_input(
            label = "Maximum videos to search",
            min_value = 1,
            max_value = 20,
            step = 1,
            value = 1
        )
        upload_date = st.selectbox(
            label = "Upload date",
            options = [
                None,
                "Last Hour",
                "Today",
                "This Week",
                "This Month",
                "This Year"],
            index = 0
        )
        duration = st.selectbox(
            label = "Duration",
            options = [
                None,
                "Under 4 minutes",
                "Over 20 minutes",
                "4 - 20 minutes"],
            index = 0
        )
        features = st.multiselect(
            label = "Features",
            options = [
                "Live",
                "4K",
                "HD",
                "Subtitles/CC",
                "Creative Commons",
                "360",
                "VR180",
                "3D",
                "HDR",
                "Location",
                "Purchased"
            ],
            default = []
        )
        sort_by = st.selectbox(
            label = "Sort by",
            options = [
                None,
                "Relevance",
                "Upload Date",
                "View count",
                "Rating"],
            index = 2
        )
        submit_project_settings = st.form_submit_button(
            "Set and search",
            use_container_width = True)
elif search_type_filter == "Video":
    with st.sidebar.form(f"Project Settings - {search_type_filter}"):
        context_to_search = st.text_area(
            label = "Context to search",
            placeholder = "Provide the context to be searched on YouTube",
        )
        video_url = st.text_input(
            label = "Video URL",
            placeholder = "Provide the URL of the video to search",
        )
        submit_project_settings = st.form_submit_button(
            "Set and search",
            use_container_width = True)
elif search_type_filter == "Channel":
    with st.sidebar.form(f"Project Settings - {search_type_filter}"):
        context_to_search = st.text_area(
            label = "Context to search",
            placeholder = "Provide the context to be searched on this YouTube Channel",
        )
        channel_url = st.text_input(
            label = "Channel URL",
            placeholder = "Provide the URL of the channel to search",
        )
        max_results = st.number_input(
            label = "Maximum videos to search",
            min_value = 1,
            step = 1,
            value = 5
        )
        submit_project_settings = st.form_submit_button(
            "Set and search",
            use_container_width = True)
elif search_type_filter == "Playlist":
    with st.sidebar.form(f"Project Settings - {search_type_filter}"):
        context_to_search = st.text_area(
            label = "Context to search",
            placeholder = "Provide the context to be searched on this YouTube playlist",
        )
        playlist_url = st.text_input(
            label = "Playlist URL",
            placeholder = "Provide the URL of the playlist to search",
        )
        max_results = st.number_input(
            label = "Maximum videos to search",
            min_value = 1,
            step = 1,
            value = 5
        )
        submit_project_settings = st.form_submit_button(
            "Set and search",
            use_container_width = True)


if submit_project_settings:
    if search_type_filter == "Search":
        st.session_state["max_results"] = max_results
        st.session_state["context_to_search"] = context_to_search
    elif search_type_filter == "Video":
        st.session_state["context_to_search"] = context_to_search
        st.session_state["video_url"] = video_url
    elif search_type_filter == "Channel":
        st.session_state["context_to_search"] = context_to_search
        st.session_state["max_results"] = max_results
        st.session_state["channel_url"] = channel_url
    elif search_type_filter == "Playlist":
        st.session_state["context_to_search"] = context_to_search
        st.session_state["max_results"] = max_results
        st.session_state["playlist_url"] = playlist_url
    st.session_state["youtube_content_search_agent"] = YouTubeContentSearch(
        st.session_state["framework"],
        st.session_state["temperature_filter"], 
        st.session_state["model_name"],
        st.session_state["shared_memory"]
    )
    if search_type_filter == "Search":
        kwargs = {
            "max_results": st.session_state["max_results"],
            "search_type": search_type_filter,
            "upload_date": upload_date,
            "duration": duration,
            "features": features,
            "sort_by": sort_by
        }
    elif search_type_filter == "Video":
        kwargs = {
            "video_url": video_url,
            "search_type": search_type_filter
        }
    elif search_type_filter == "Channel":
        kwargs = {
            "channel_url": channel_url,
            "search_type": search_type_filter,
            "max_results": max_results
        }
    elif search_type_filter == "Playlist":
        kwargs = {
            "playlist_url": playlist_url,
            "search_type": search_type_filter,
            "max_results": max_results
        }
    st.session_state["youtube_content_search_agent"].load_model(**kwargs)
    st.session_state["youtube_content_search_agent"].stream_graph_updates(
        context_to_search)
    st.session_state["snapshot"] = st.session_state["youtube_content_search_agent"].graph.get_state(
        st.session_state["youtube_content_search_agent"].config)

if not "snapshot" in st.session_state:
    st.session_state["snapshot"] = []

try:
    context_to_search = st.session_state["context_to_search"]
except:
    st.info("Please, provide a context for YouTube searches.")
    st.stop()


chatbot_agent = YouTubeChatbot(
    st.session_state["framework"],
    st.session_state["temperature_filter"],
    st.session_state["model_name"],
    st.session_state["shared_memory"]
)
chatbot_agent.load_model(st.session_state["youtube_content_search_agent"].rag_chain)


view_app_graph = st.session_state["view_graph_button_container"].button(
    label = "View application graphs",
    use_container_width = True,
)
if view_app_graph:
    view_application_graphs(
        {
            "YouTube Content Search": st.session_state["youtube_content_search_agent"].graph,
            "YouTube Chatbot": chatbot_agent.graph})
    

view_neo4j_graph = st.sidebar.button(
    label = "View Neo4j context graph",
    use_container_width = True,
)
if view_neo4j_graph:
    view_neo4j_context_graph()


st.session_state["snapshot"] += chatbot_agent.graph.get_state(chatbot_agent.config)
messages_blocks_ = [
    x 
    for i, x 
    in enumerate(st.session_state["snapshot"])
    if i % 7 == 0
    ]
messages_blocks = []
for item in messages_blocks_:
    if item not in messages_blocks:
        messages_blocks.append(item)
streamlit_actions = []
for item in messages_blocks:
    if item not in streamlit_actions:
        streamlit_actions += item["streamlit_actions"]
#if not submit_project_settings:
try:
    index_start_messages = np.where(np.array([x[0][3] for x in streamlit_actions]) == "user")[0][1]
    streamlit_actions = streamlit_actions[index_start_messages:]
except:
    pass
for actions in streamlit_actions:
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
#else:
#    pass


if prompt := st.chat_input():
    chatbot_agent.stream_graph_updates(
        prompt)