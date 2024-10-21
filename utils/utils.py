import pandas as pd


def sanitize_text(text):
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
