import streamlit as st
from PIL import Image
from labeller import *
import numpy as np
from ultralytics import YOLO


class DetectionResult:
    COLOR_MAP = {
        "train": "#FF0000",
        "track": "#00FF00",
        "signal": "#FFFF00",
        "platform": "#00FFFF",
        "overhead wire": "#FF00FF",
        "crossing gate": "#FFA500",
   }

    def __init__(
        self,
        label="",
        description="",
        confidence=0.0,
        rect_1=(0, 0),
        rect_2=(0, 0)
    ):
        self.label = label
        self.description = description
        self.confidence = confidence
        self.rect_1 = rect_1
        self.rect_2 = rect_2

    def get_color(self):
        return self.COLOR_MAP.get(
            self.label.lower(),
            "#FF00FF"
        )


# TODO: remove test class, this is only for frontend mockup
CLASSES = {
    "Train": "Rail vehicles, use tracks for movement",
    "Track": "The steel rails and sleepers that guide and support trains.",
    "Signal": "Device used to control train movements and traffic flow.",
    "Platform": "The passenger boarding area, parallel to the tracks at a station.",
    "Overhead Wire": "Lines that hang above a train, supplying electrical power.",
    "Crossing Gate": "A safety barrier that blocks road & foot traffic when a train is passing.",
}


@st.cache_resource
def get_model():
    return YOLO("yolov8n.pt")


model = get_model()

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
</style>
"""

st.markdown(
    hide_streamlit_style,
    unsafe_allow_html=True
)

# apply style.css file
try:
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
except FileNotFoundError:
    pass

st.title("ZRA Labs: Railway Object Detection")

st.markdown(
    "Upload an image to identify railway assets",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# XXX: THIS BLOCK CONTAINS 'DUMMY' CODE AS OF NOW, LINK YOLO OR ANOTHER MODEL
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    with st.spinner("Processing ..."):

        results = model.predict(
            source=np.array(image),
            conf=0.25
        )

        detections = []

        if len(results) > 0 and len(results[0].boxes) > 0:

            for box in results[0].boxes:

                cls_idx = int(box.cls[0].item())

                label = results[0].names[cls_idx].title()

                desc = CLASSES.get(
                    label,
                    "Detected asset"
                )

                conf = float(box.conf[0].item())

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                res = DetectionResult(
                    label=label,
                    description=desc,
                    confidence=conf,
                    rect_1=(int(x1), int(y1)),
                    rect_2=(int(x2), int(y2))
                )

                detections.append(res)

    labelled_image = labelImage(
        detections,
        image.copy()
    )

    st.image(
        labelled_image,
        caption="Detected Objects",
        use_container_width=True
    )

    st.success("Analysis has been completed")

    for res in detections:

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Detected Object",
                value=res.label
            )

        with col2:
            st.metric(
                label="Confidence Score",
                value=f"{res.confidence:.2%}"
            )

        st.info(f"Classification >> {res.description}")