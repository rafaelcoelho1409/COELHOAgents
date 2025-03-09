import streamlit as st
from streamlit_extras.grid import grid
from langgraph.checkpoint.memory import MemorySaver
from functions import (
    image_border_radius,
)


grid_logo = grid([0.15, 0.7, 0.15], vertical_align = True)
grid_logo.container()
image_border_radius("assets/coelho_agents_logo.png", 20, 100, 100, grid_logo)

st.session_state["shared_memory"] = MemorySaver()