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


initialize_shared_memory()


model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()