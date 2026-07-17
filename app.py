import streamlit as st
from PIL import Image
import time
import random

# remove all of streamlit's bloat
st.set_page_config(
    page_title="ZRA Labs: Railway Object Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# header footer etc, all gone for now
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            #stDecoration {display: none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# apply style.css file
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.title("ZRA Labs: Railway Object Detection")
st.write("Upload an image to identify railway assets")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# TODO: remove test class, this is only for frontend mockup
CLASSES = {
    "Train": "Rail vehicles, use tracks for movement",
    "Track": "The steel rails and sleepers that guide and support trains.",
    "Signal": "Device used to control train movements and traffic flow.",
    "Platform": "The passenger boarding area, parallel to the tracks at a station.",
    "Overhead Wire": "Lines that hang above a train, supplying electrical power.",
    "Crossing Gate": "A safety barrier that blocks road & foot traffic when a train is passing.",
    "Test Class": "This is for testing only: Lorem ipsum dolor sit amet."
}

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # image analysis here
    with st.spinner("Processing ..."):
        time.sleep(1.0)  
        
        detected_label = "Test Class"
        confidence_score = random.uniform(0.50, 1.00)
        explanation = CLASSES[detected_label]

    st.success("Analysis has been completed")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Detected Object", value=detected_label)
    with col2:
        st.metric(label="Confidence Score", value=f"{confidence_score:.2%}")
        
    st.info(f"Classification >> {explanation}")