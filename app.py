import streamlit as st
from PIL import Image
from labeller import *
from result import DetectionResult
import numpy as np
from ultralytics import YOLO


# TODO: remove test classes, these are only for the frontend mockup
CLASSES = {
    "Train": "Rail vehicles, use tracks for movement",
    "Track": "The steel rails and sleepers that guide and support trains.",
    "Signal": "Device used to control train movements and traffic flow.",
    "Platform": "The passenger boarding area, parallel to the tracks at a station.",
    "Overhead Wire": "Lines that hang above a train, supplying electrical power.",
    "Crossing Gate": "A safety barrier that blocks road & foot traffic when a train is passing.",
}

# the look of the app lives in .streamlit/config.toml, not in here
st.set_page_config(
    page_title="ZRA Railway Detection",
    page_icon=":material/radar:",
    layout="centered",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def get_model():
    return YOLO("yolov8n.pt")


model = get_model()

# style.css only holds what the theme can't express (image shadow etc.)
try:
    with open("style.css") as f:
        st.html(f"<style>{f.read()}</style>")
except FileNotFoundError:
    pass


def render_header():
    st.badge(
        "Computer vision",
        icon=":material/network_node:",
        color="primary"
    )

    st.title("Railway object detection")

    st.caption(
        "Upload a railway-related photo, and the model marks up the "
        "assets it recognises, with a confidence score for each."
    )


def render_class_legend():
    """Shown before an upload, so the page isn't an empty box."""
    st.subheader("What it looks for")

    for label, description in CLASSES.items():

        with st.container(border=True, gap=None):

            probe = DetectionResult(label=label)

            st.badge(
                label,
                color=probe.get_badge_color()
            )

            st.caption(description)


def render_summary(detections):
    labels = [res.label for res in detections]

    top = max(detections, key=lambda r: r.confidence)

    cols = st.columns(3)

    cols[0].metric(
        "Objects found",
        len(detections),
        border=True
    )

    cols[1].metric(
        "Distinct classes",
        len(set(labels)),
        border=True
    )

    cols[2].metric(
        "Best match",
        top.label,
        delta_description=f"{top.confidence:.0%} confident",
        border=True
    )


def render_detection(res):
    with st.container(border=True):

        with st.container(horizontal=True, vertical_alignment="center"):

            st.badge(
                res.label,
                color=res.get_badge_color()
            )

            st.caption(f"{res.confidence:.1%} confidence")

        st.progress(min(max(res.confidence, 0.0), 1.0))

        st.caption(res.description)


# XXX: THIS BLOCK CONTAINS 'DUMMY' CODE AS OF NOW, LINK YOLO OR ANOTHER MODEL
def detect(image):
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

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            res = DetectionResult(
                label=label,
                description=desc,
                confidence=conf,
                rect_1=(int(x1), int(y1)),
                rect_2=(int(x2), int(y2))
            )

            detections.append(res)

    return detections


render_header()

st.space("medium")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="JPG/JPEG/PNG"
)

if uploaded_file is None:
    st.space("medium")
    render_class_legend()

else:
    image = Image.open(uploaded_file)

    with st.spinner("Scanning image ...", show_time=True):
        detections = detect(image)

    st.space("medium")

    if not detections:
        st.warning(
            "No railway assets recognised in this image.",
            icon=":material/search_off:"
        )

    else:
        render_summary(detections)

        st.space("small")

        st.image(
            labelImage(detections, image.copy()),
            caption="Detected objects",
            width="stretch"
        )

        st.space("medium")

        st.subheader("Detections")

        for res in detections:
            render_detection(res)
