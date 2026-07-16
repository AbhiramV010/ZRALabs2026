import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# apply the css for centering & rounding of elements
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
