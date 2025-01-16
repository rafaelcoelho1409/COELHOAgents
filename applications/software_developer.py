import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from functions import (
    check_model_and_temperature,
    initialize_shared_memory,
    view_application_graph
)
from models.software_developer import SoftwareDeveloper


initialize_shared_memory()

model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()

language_option = st.sidebar.selectbox(
    label = "Language",
    options = [
        "Python"
    ]
)

role = SoftwareDeveloper(
    st.session_state["framework"],
    st.session_state["temperature_filter"], 
    st.session_state["model_name"],
    st.session_state["shared_memory"]
)
role.load_model(language_option)

#snapshot = role.graph.get_state(role.config)
#for msg in st.session_state["history"].messages:
try:
    #for msg in snapshot.values["messages"]:
    for msg in st.session_state["history"].messages:
        st.chat_message(msg.type).write(msg.content)
except:
    pass

if prompt := st.chat_input():
    if st.session_state["memory_filter"] == False:
        st.session_state["shared_memory"] = MemorySaver()
    role.stream_graph_updates_test(language_option, prompt)