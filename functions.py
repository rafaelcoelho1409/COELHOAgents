import streamlit as st
import base64
import ollama
import subprocess
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver


###>>>---LOCAL FUNCTIONS---<<<###
def image_border_radius(image_path, border_radius, width, height, page_object = None, is_html = False):
    if is_html == False:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        # Create HTML string with the image
        img_html = f'<img src="data:image/jpeg;base64,{img_base64}" style="border-radius: {border_radius}px; width: {width}%; height: {height}%">'
        # Display the HTML string in Streamlit
        if page_object == None:
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            page_object.markdown(img_html, unsafe_allow_html=True)
    else:
        # Create HTML string with the image
        img_html = f'<img src="{image_path}" style="border-radius: {border_radius}px; width: 300px;">'
        # Display the HTML string in Streamlit
        if page_object == None:
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            page_object.markdown(img_html, unsafe_allow_html=True)

def reload_active_models():
    active_models_container = st.sidebar.container()
    active_models_text = "**Active model:** "
    if st.session_state["framework"] == "Ollama":
        if ollama.ps()["models"] != []:
            for model_name in ollama.ps()["models"]:
                active_models_text += f"* {model_name['model']}\n"
        else:
            active_models_text += "No active models."
    elif st.session_state["framework"] == "Groq":
        active_models_text += st.session_state["model_name"]
    active_models_container.info(active_models_text)

def check_model_and_temperature():
    return all([x in st.session_state.keys() for x in ["model_name", "temperature_filter"]])

def initialize_shared_memory():
    # Initialize shared memory
    if "history" not in st.session_state:
        st.session_state["history"] = StreamlitChatMessageHistory(key = "chat_history")
    if "shared_memory" not in st.session_state:
        st.session_state["shared_memory"] = MemorySaver()


###>>>---STREAMLIT FUNCTIONS---<<<###
@st.dialog("Settings")
def settings():
    framework_option = st.selectbox(
        label = "Framework",
        options = [
            "Groq",
            "Ollama"
        ]
    )
    st.session_state["framework"] = framework_option
    if framework_option == "Ollama":
        with st.form("Settings Ollama"):
            models_options = sorted(
                [x["model"] for x in ollama.list()["models"]])
            if ollama.ps()["models"] != []:
                active_models = [x["model"] for x in ollama.ps()["models"]]
                models_filter = st.selectbox(
                    label = "Ollama Models",
                    options = models_options,
                    index = models_options.index(active_models[0])
                )
            else:
                try:
                    #try to get the last model used, if exists
                    models_filter = st.selectbox(
                        label = "Ollama Models",
                        options = models_options,
                        index = models_options.index(st.session_state["model_name"])
                    )
                except:
                    models_filter = st.selectbox(
                        label = "Ollama Models",
                        options = sorted([x["model"] for x in ollama.list()["models"]])
                    )
            temperature_filter = st.slider(
                label = "Temperature",
                min_value = 0.00,
                max_value = 1.00,
                value = 0.00,
                step = 0.01
            )
            toggle_filters = st.columns(3)
            try:
                memory_filter = toggle_filters[0].toggle(
                    label = "Memory",
                    value = st.session_state["memory_filter"]
                )
            except:
                memory_filter = toggle_filters[0].toggle(
                    label = "Memory",
                    value = True
                )
            try:
                vector_database_filter = toggle_filters[1].toggle(
                    label = "Vector database",
                    value = st.session_state["vector_database_filter"]
                )
            except:
                vector_database_filter = toggle_filters[1].toggle(
                    label = "Vector database",
                )
            try:
                rag_filter = toggle_filters[2].toggle(
                    label = "RAG",
                    value = st.session_state["rag_filter"]
                )
            except:
                rag_filter = toggle_filters[2].toggle(
                    label = "RAG",
                )
            submit_button = st.form_submit_button(
                    label = "Run model",
                    use_container_width = True
                )
            if submit_button:
                if "model_name" in st.session_state:
                    if st.session_state["model_name"] != models_filter:
                        subprocess.check_call([
                            "ollama",
                            "stop",
                            st.session_state["model_name"]
                        ],
                        )
                else:
                    subprocess.check_call([
                        "ollama",
                        "stop",
                        models_filter
                    ],
                    )
                st.session_state["model_name"] = models_filter
                st.session_state["temperature_filter"] = temperature_filter
                st.session_state["memory_filter"] = memory_filter
                st.session_state["vector_database_filter"] = vector_database_filter
                st.session_state["rag_filter"] = rag_filter
                st.rerun()
    elif framework_option == "Groq":
        with st.form("Settings Groq"):
            models_option = st.selectbox(
                label = "Groq Models", 
                options = [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant"
                ])
            temperature_filter = st.slider(
                label = "Temperature",
                min_value = 0.00,
                max_value = 1.00,
                value = 0.00,
                step = 0.01
            )
            toggle_filters = st.columns(3)
            try:
                memory_filter = toggle_filters[0].toggle(
                    label = "Memory",
                    value = st.session_state["memory_filter"]
                )
            except:
                memory_filter = toggle_filters[0].toggle(
                    label = "Memory",
                    value = True
                )
            try:
                vector_database_filter = toggle_filters[1].toggle(
                    label = "Vector database",
                    value = st.session_state["vector_database_filter"]
                )
            except:
                vector_database_filter = toggle_filters[1].toggle(
                    label = "Vector database",
                )
            try:
                rag_filter = toggle_filters[2].toggle(
                    label = "RAG",
                    value = st.session_state["rag_filter"]
                )
            except:
                rag_filter = toggle_filters[2].toggle(
                    label = "RAG",
                )
            submit_button = st.form_submit_button(
                    label = "Run model",
                    use_container_width = True
                )
            if submit_button:
                st.session_state["model_name"] = models_option
                st.session_state["temperature_filter"] = temperature_filter
                st.session_state["memory_filter"] = memory_filter
                st.session_state["vector_database_filter"] = vector_database_filter
                st.session_state["rag_filter"] = rag_filter
                st.rerun()


@st.dialog("Application graph")
def view_application_graph(graph):
    st.image(graph.get_graph().draw_mermaid_png())
