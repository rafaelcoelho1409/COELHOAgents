import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
import os
from pathlib import Path
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


with st.sidebar.form("Project Settings"):
    technology_option = st.selectbox(
        label = "Technology",
        options = [
            "Python",
            "Java",
            "Go",
            "C++",
            "C#",
            ".NET",
            "MySQL",
            "PostgreSQL"
        ]
    )
    project_name = st.text_input(
        label = "Project Name",    
        )
    submit_project_settings = st.form_submit_button(
        "Create/Load Project",
        use_container_width = True)
if submit_project_settings:
    if project_name == "":
        st.sidebar.info("Please enter a project name.")
        st.stop()
    else:
        home_folder = Path.home()
        main_folder = home_folder / "COELHOAgentsProjects"
        project_folder = main_folder / project_name
        os.makedirs(project_folder, exist_ok = True)
        st.session_state["technology_option"] = technology_option
        st.session_state["project_name"] = project_name
        st.session_state["project_folder"] = project_folder
        st.sidebar.success(f"Project {project_name} created/loaded successfully.")
try:
    project_name = st.session_state["project_name"]
    st.sidebar.info(f"**Project Name:** {project_name}")
except:
    st.sidebar.info("Please create/load a project.")
    st.stop()


role = SoftwareDeveloper(
    st.session_state["framework"],
    st.session_state["temperature_filter"], 
    st.session_state["model_name"],
    st.session_state["shared_memory"]
)
role.load_model(
    technology_option,
    st.session_state["project_folder"],
    )

view_graph = st.sidebar.button(
    label = "View application graph",
    use_container_width = True
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
        technology_option, 
        st.session_state["project_name"], 
        prompt)