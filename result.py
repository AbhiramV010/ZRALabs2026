class DetectionResult:
    COLOR_MAP = {
        "train": "#FF0000",
        "track": "#00FF00",
        "signal": "#FFFF00",
        "platform": "#00FFFF",
        "overhead wire": "#FF00FF",
        "crossing gate": "#FFA500",
        "test class": "#FFFFFF"
    }

    def __init__(self, label="", description="", confidence=0.0, rect_1=(0, 0), rect_2=(0, 0)):
        self.label = label
        self.description = description
        self.confidence = confidence
        self.rect_1 = rect_1
        self.rect_2 = rect_2

    def get_color(self):
        return self.COLOR_MAP.get(self.label.lower(), "#FF00FF")
