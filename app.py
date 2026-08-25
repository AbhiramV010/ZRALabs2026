import os

import requests
import streamlit as st
from pathlib import Path
from PIL import Image
from labeller import labelImage
from result import DetectionResult
from model.predict import RailwayClassifier
from store import CaptureStore

ASSETS = Path(__file__).resolve().parent / "assets"

# set this and the interface stops holding a model of its own, calling
# api.py instead. Unset, it scans in process exactly as it always did
API_URL = os.environ.get("ZRA_API_URL")

API_TIMEOUT = 120

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


@st.cache_resource
def get_store():
    return CaptureStore()

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
            "Upload railway-related photos, and the model marks up the "
            "assets it recognises, with a confidence score for each."
        )

    st.html(RAIL)


def render_class_legend():
    st.subheader("What it looks for")

    st.caption(
        "Each class is drawn in its own colour on the marked-up image."
    )

    st.space("small")

    items = list(CLASSES.items())

    # two per row, so the landing page isn't one long column of cards
    for start in range(0, len(items), 2):
        cols = st.columns(2, gap="small")

        for offset, (label, description) in enumerate(items[start:start + 2]):

            index = start + offset

            with cols[offset]:
                # stretch keeps both cards in a row the same height
                with st.container(
                    border=True,
                    gap=None,
                    height="stretch",
                    key=f"legend-{index}"
                ):

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
            # the class colour, so the card points at its own box
            f":{top.get_badge_color()}[{top.label}]",
            f"{top.confidence:.0%} confident"
        )


# st.progress paints every bar the theme's primary. These rules repaint each
# one in its class colour, so a row reads as the box it came from. The style
# block is collapsed by style.css, it takes no room in the layout.
def paint_confidence_bars(prefix, results):
    rules = "\n".join(
        f'.st-key-{prefix}-{index} [data-testid="stProgressBarTrack"] > div '
        f'{{ background-color: {res.get_color()}; }}'
        for index, res in enumerate(results)
    )

    st.html(f"<style>{rules}</style>")


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


def shift_carousel(delta, total):
    st.session_state.carousel_index = min(
        max(st.session_state.carousel_index + delta, 0), total - 1
    )


def render_carousel_nav(current_file, index, total):
    cols = st.columns([1, 3, 1], vertical_alignment="center", gap="small")

    with cols[0]:
        st.button(
            "Previous",
            icon=":material/chevron_left:",
            disabled=index == 0,
            width="stretch",
            on_click=shift_carousel,
            args=(-1, total)
        )

    with cols[1]:
        st.caption(f"Image {index + 1} of {total} · {current_file.name}")

    with cols[2]:
        st.button(
            "Next",
            icon=":material/chevron_right:",
            disabled=index == total - 1,
            width="stretch",
            on_click=shift_carousel,
            args=(1, total)
        )


def detect_via_api(upload):
    """Ask the server what is in this frame.

    The descriptions are filled in here rather than sent - they are
    wording for this screen, and there is no reason to put them on a
    wire that might be a radio link.
    """
    assert API_URL is not None

    response = requests.post(
        f"{API_URL.rstrip('/')}/v1/classify",
        files={"files": (upload.name, upload.getvalue(), upload.type)},
        timeout=API_TIMEOUT,
    )

    response.raise_for_status()

    entry = response.json()["results"][0]

    if entry.get("error"):
        raise ValueError(entry["error"])

    results = [
        DetectionResult.from_dict(hit) for hit in entry["detections"]
    ]

    for res in results:
        res.description = describe(res.label)

    return results


def detect_locally(image):
    found = list(get_model().detect(image))

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


def detect(image, upload=None):
    """Everything the model finds in the frame, strongest first."""
    if API_URL and upload is not None:

        try:
            return detect_via_api(upload)

        # a server that is down should degrade to a slower interface,
        # not a broken one. The whole point of this running offline is
        # that the network is the part expected to fail
        except Exception as error:
            st.warning(
                f"Server unreachable, scanning on this machine instead. ({error})",
                icon=":material/cloud_off:"
            )

    return detect_locally(image)


def scan(image, upload):
    """Detect, and keep a record of it on this device."""
    results = detect(image, upload)

    try:
        get_store().record(
            image,
            results,
            source_name=upload.name,
            model_name="railway_classifier",
            model_version="api" if API_URL else get_model().backend.runtime,
        )

    # a full card or a read-only disk should not lose the user a scan
    # they can still see on screen
    except Exception as error:
        st.warning(
            f"Could not record this capture locally. ({error})",
            icon=":material/save_as:"
        )

    return results


def render_device_panel():
    """What this machine is holding, and where it sends it."""
    with st.sidebar:

        st.subheader("On this device")

        try:
            stats = get_store().stats()

        except Exception as error:
            st.caption(f"No capture store here. ({error})")

            return

        st.metric("Captures kept", stats["captures"])

        st.metric("Waiting to upload", stats["pending"])

        st.caption(
            f"{stats['detections']} detections, "
            f"{stats['thumb_bytes'] / 1_000_000:.1f} MB of thumbnails"
        )

        st.caption(
            f"Scanning on {API_URL}" if API_URL
            else "Scanning on this machine"
        )


render_header()

st.space("medium")

uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    help="JPG/JPEG/PNG",
    accept_multiple_files=True
)

if not uploaded_files:
    st.space("medium")
    render_class_legend()

else:
    ids = [f.file_id for f in uploaded_files]

    # a fresh upload set starts the carousel over and drops stale results
    if st.session_state.get("carousel_ids") != ids:
        st.session_state.carousel_ids = ids
        st.session_state.carousel_index = 0
        st.session_state.carousel_cache = {}

    total = len(uploaded_files)
    carousel_index = st.session_state.carousel_index
    current_file = uploaded_files[carousel_index]
    image = Image.open(current_file)

    st.space("medium")

    render_carousel_nav(current_file, carousel_index, total)

    # each image is scanned once per session, Previous/Next just replays it
    if current_file.file_id not in st.session_state.carousel_cache:
        scanner = st.empty()
        scanner.html(RAIL_SCANNING)

        with st.spinner("Scanning image ...", show_time=True):
            st.session_state.carousel_cache[current_file.file_id] = scan(
                image, current_file
            )

        scanner.empty()

    detections = st.session_state.carousel_cache[current_file.file_id]

    st.space("medium")

    if not detections:
        st.warning(
            "No railway assets recognised in this frame.",
            icon=":material/search_off:"
        )

    else:
        render_summary(detections)

        st.space("small")

        st.image(
            labelImage(detections, image.copy()),
            caption="Box colours match the badges below",
            width="stretch"
        )

        st.space("medium")

        st.subheader("Detections")

        paint_confidence_bars("detection", detections)

        for index, res in enumerate(detections):
            render_detection(res, key=f"detection-{index}")


# last, not first. st.sidebar writes to the sidebar wherever it is
# called from, and calling it up at the top would count the captures as
# they were before this run recorded one, so the panel would always be
# showing the answer to the previous upload
render_device_panel()
