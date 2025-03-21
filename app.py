import streamlit as st
from streamlit_extras.grid import grid
from langgraph.checkpoint.memory import MemorySaver
from functions import (
    settings,
    check_model_and_temperature,
    initialize_shared_memory
)


st.set_page_config(
    page_title = "COELHO Agents", 
    page_icon = ":material/home:",
    layout = "wide")

pages_dict = {
    #"Home": "applications/home.py",
    "Simple Assistant": "applications/simple_assistant.py",
    "Software Developer": "applications/software_developer.py",
    "YouTube Content Search": "applications/youtube_content_search.py",
}
pages = {
    "Home": st.Page("applications/home.py", title = "Home", icon = ":material/home:")} | {
    name: st.Page(path, title = name, icon = ":material/edit:")
    for name, path 
    in pages_dict.items()}


pg = st.navigation({
    "COELHO Agents by Rafael Coelho": [
        pages["Home"]],
    "Applications": [
        pages["Simple Assistant"],
        pages["Software Developer"],
        pages["YouTube Content Search"]
    ],

})

with open("style.css") as css:
    st.html(f"<style>{css.read()}</style>")

with st.container(key = "app_title"):
    st.title(("$$\\textbf{" + pg.title + "}$$").replace("&", "\&"))

if pg.title in [
    "Home",
    "Simple Assistant",
    "Software Developer"]:
    grid_buttons = st.sidebar.columns(2)
    settings_button = grid_buttons[0].button(
        label = "Settings",
        use_container_width = True
    )
    if settings_button:
        settings()
    clear_memory_button = grid_buttons[1].button(
        label = "Clear memory",
        use_container_width = True
    )
else:
    settings_button = st.sidebar.button(
        label = "Settings",
        use_container_width = True
    )
    if settings_button:
        settings()
st.session_state["view_graph_button_container"] = st.sidebar.container()


pg.run()

model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    if pg.title != "Home":
        st.info("Choose model and temperature to start running COELHO Agents models.")
    st.stop()


with st.sidebar.expander("**Informations**", expanded = True):
    st.markdown(f"**Framework:** {st.session_state['framework']}")
    st.markdown(f"**Model:** {st.session_state["model_name"]}")
    st.markdown(f"**Temperature:** {st.session_state["temperature_filter"]}")
    #reload_active_models()


initialize_shared_memory()
try:
    if clear_memory_button:
        st.session_state["shared_memory"] = MemorySaver()
except:
    pass