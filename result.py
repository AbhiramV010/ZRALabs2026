from typing import Literal

BadgeColor = Literal[
    "red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"
]


class DetectionResult:
    # box colors, matched to the palette in .streamlit/config.toml
    COLOR_MAP = {
        "train": "#F87171",
        "track": "#60A5FA",
        "signal": "#FACC15",
        "platform": "#4ADE80",
        "overhead wire": "#C084FC",
        "crossing gate": "#FB923C",
    }

    # the same classes as streamlit badge color names
    BADGE_MAP: dict[str, BadgeColor] = {
        "train": "red",
        "track": "blue",
        "signal": "yellow",
        "platform": "green",
        "overhead wire": "violet",
        "crossing gate": "orange",
    }

    def __init__(self, label="", description="", confidence=0.0, rect_1=(0, 0), rect_2=(0, 0)):
        self.label = label
        self.description = description
        self.confidence = confidence
        self.rect_1 = rect_1
        self.rect_2 = rect_2

    def get_color(self):
        return self.COLOR_MAP.get(self.label.lower(), "#94A3B8")

    def get_badge_color(self) -> BadgeColor:
        return self.BADGE_MAP.get(self.label.lower(), "gray")

    # the colors above are for this screen, these two are for everywhere
    # else - the local store writes these fields, and a sync client puts
    # them on the wire. Corners go out as lists so a value survives a
    # round trip through JSON as the same thing it went in as.
    def to_dict(self):
        return {
            "label": self.label,
            "description": self.description,
            "confidence": self.confidence,
            "rect_1": list(self.rect_1),
            "rect_2": list(self.rect_2),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            label=data.get("label", ""),
            description=data.get("description", ""),
            confidence=data.get("confidence", 0.0),
            rect_1=tuple(data.get("rect_1") or (0, 0)),
            rect_2=tuple(data.get("rect_2") or (0, 0)),
        )
