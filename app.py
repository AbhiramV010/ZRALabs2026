import streamlit as st
from pathlib import Path
from PIL import Image
from labeller import *
from result import DetectionResult
from model.predict import RailwayClassifier

ASSETS = Path(__file__).resolve().parent / "assets"

# the six classes the checkpoint was trained on
CLASSES = {
    "Train": "Rail vehicles, use tracks for movement",
    "Track": "The steel rails and sleepers that guide and support trains.",
    "Signal": "Device used to control train movements and traffic flow.",
    "Platform": "The passenger boarding area, parallel to the tracks at a station.",
    "Crossing Gate": "A safety barrier that blocks road & foot traffic when a train is passing.",
    "Overhead Wire": "The catenary and contact wire strung above the track "
                     "that electric trains draw current from.",
}

# a length of track, animated in style.css
RAIL = '<div class="rail"></div>'
RAIL_SCANNING = (
    '<div class="rail rail-live">'
    '<span class="lamp"></span><span class="lamp"></span>'
    '</div>'
)

st.set_page_config(
    page_title="ZRA Railway Detection",
    page_icon=str(ASSETS / "icon-192.png"),
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.logo(
    str(ASSETS / "logo.png"),
    size="large",
    link="https://railwayacademy.org/",
    icon_image=str(ASSETS / "icon-192.png")
)


@st.cache_resource
def get_model():
    return RailwayClassifier()


model = get_model()

try:
    with open("style.css") as f:
        st.html(f"<style>{f.read()}</style>")
except FileNotFoundError:
    pass


def render_header():
    with st.container(key="app-header"):

        st.badge(
            "Computer vision",
            icon=":material/network_node:",
            color="primary"
        )

        st.title("Railway Object Detection")

        st.caption(
            "Upload a railway-related photo, and the model marks up the "
            "assets it recognises, with a confidence score for each."
        )

    st.html(RAIL)


def render_class_legend():
    st.subheader("What it looks for")

    for index, (label, description) in enumerate(CLASSES.items()):

        with st.container(border=True, gap=None, key=f"legend-{index}"):

            probe = DetectionResult(label=label)

            st.badge(
                label,
                color=probe.get_badge_color()
            )

            st.caption(description)


# st.metric truncates long values, these cards don't
def render_card(index, label, value, note=None):
    with st.container(border=True, height="stretch", key=f"summary-{index}"):

        st.caption(label)

        st.header(value, anchor=False)

        st.caption(note or "")


def render_summary(detections):
    top = detections[0]

    labels = {res.label for res in detections}

    cols = st.columns(3, gap="small")

    with cols[0]:
        render_card(
            0,
            "Objects found",
            str(len(detections)),
            "across the frame"
        )

    with cols[1]:
        render_card(
            1,
            "Distinct classes",
            str(len(labels)),
            f"of {len(CLASSES)} the model knows"
        )

    with cols[2]:
        render_card(
            2,
            "Best match",
            top.label,
            f"{top.confidence:.0%} confident"
        )


def render_detection(res, key=None):
    with st.container(border=True, key=key):

        with st.container(horizontal=True, vertical_alignment="center"):

            st.badge(
                res.label,
                color=res.get_badge_color()
            )

            st.caption(f"{res.confidence:.1%} confidence")

        st.progress(min(max(res.confidence, 0.0), 1.0))

        st.caption(res.description)


def describe(label):
    return CLASSES.get(label, "Railway asset")


def detect(image):
    """Everything the model finds in the frame, strongest first."""
    found = list(model.detect(image))

    results = [
        DetectionResult(
            label=label,
            description=describe(label),
            confidence=confidence,
            rect_1=(x1, y1),
            rect_2=(x2, y2)
        )
        for label, confidence, (x1, y1, x2, y2) in found
    ]

    return sorted(results, key=lambda res: -res.confidence)


def classify(image):
    """The whole-frame verdict, one entry per class, best first."""
    ranked = list(model.classify(image))

    return [
        DetectionResult(
            label=label,
            description=describe(label),
            confidence=confidence
        )
        for label, confidence in sorted(ranked, key=lambda pair: -pair[1])
    ]


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

    # the running signal light is cleared once the scan finishes
    scanner = st.empty()
    scanner.html(RAIL_SCANNING)

    with st.spinner("Scanning image ...", show_time=True):
        detections = detect(image)
        ranked = classify(image)

    scanner.empty()

    st.space("medium")

    if not detections:
        st.warning(
            "No railway assets recognised in this frame. The whole-frame "
            "ranking below is shown anyway.",
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

        for index, res in enumerate(detections):
            render_detection(res, key=f"detection-{index}")

    st.space("medium")

    with st.expander("Whole-frame class confidence"):

        for index, res in enumerate(ranked):
            render_detection(res, key=f"ranked-{index}")
