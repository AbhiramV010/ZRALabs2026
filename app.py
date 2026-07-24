import streamlit as st
from PIL import Image
import time
import random

class DetectionResult:
    def __init__(self, label, description, confidence, rect_1=(0, 0), rect_2=(0, 0)):
        self.label = label
        self.description = description
        self.confidence = confidence
        self.rect_1 = rect_1
        self.rect_2 = rect_2

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

# remove all of streamlit's bloat
st.set_page_config(
    page_title="ZRA Labs Summer Project",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
st.markdown("Upload an image to identify railway assets <br><br><br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # image analysis here
    with st.spinner("Processing ..."):
        time.sleep(1.0)  
        
        detections = []
        mock_labels = ["Test Class", "Track"]
        
        for label in mock_labels:
            res = DetectionResult()
            res.label = label
            res.confidence = random.uniform(0.50, 1.00)
            res.description = CLASSES[label]
            res.rect_1 = (0, 0)
            res.rect_2 = (0, 0)
            detections.append(res)

    st.success("Analysis has been completed")
    
    for res in detections:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Detected Object", value=res.label)
        with col2:
            st.metric(label="Confidence Score", value=f"{res.confidence:.2%}")
            
        st.info(f"Classification >> {res.description}")