import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from functions import (
    check_model_and_temperature,
    initialize_shared_memory,
    view_application_graph
)
from models.simple_assistant import SimpleAssistant

initialize_shared_memory()

model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()

role = SimpleAssistant(
    st.session_state["framework"],
    st.session_state["temperature_filter"], 
    st.session_state["model_name"],
    st.session_state["shared_memory"]
)
role.load_model()

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
        st.chat_message(msg.type).write(msg.content)
except:
    pass

#st.session_state["shared_memory"] = MemorySaver()

if prompt := st.chat_input():
    #if st.session_state["memory_filter"] == False:
    #    st.session_state["shared_memory"] = MemorySaver()
    role.stream_graph_updates(prompt)