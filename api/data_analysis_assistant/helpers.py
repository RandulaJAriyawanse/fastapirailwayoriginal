import json
import io
import re
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from utils.utils import sanitize_text
import pandas as pd
import base64
import numpy as np


def get_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_string = buffer.getvalue()
    return info_string


def filter_json_schema(json_schema, filter):
    """
    Retrieves the snippet of the json schema to be used as a reference for
    generating data analysis code
    """
    keys = filter.split("::")
    for key in keys:
        try:
            json_schema = json_schema["properties"]
        except KeyError:
            try:
                json_schema = json_schema["patternProperties"]
            except KeyError:
                pass
        json_schema = json_schema[key]
    return json_schema


# Temporary
def retrieve_data(file_name, filter):
    """
    Temporary function in place of API response from EODHD
    """
    file_path = os.path.join(os.path.dirname(__file__), file_name)

    with open(file_path, "r") as file:
        data = json.load(file)
    keys = filter.split("::")
    for key in keys:
        data = data[key]
    return data


def extract_python_code(text):
    code_block = re.findall(r"```python(.*)```", text, re.DOTALL)

    if code_block:
        cleaned_code = code_block[0].strip().strip('"""')
        return cleaned_code
    return None


def serialize_messages(messages):
    message_dicts = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            message_dicts.append({"type": "AIMessage", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            message_dicts.append({"type": "HumanMessage", "content": msg.content})

    messages_json = json.dumps(message_dicts)
    return messages_json


def dataframe_to_json_serializable(df):
    df = df.applymap(
        lambda x: (
            x.isoformat()
            if isinstance(x, pd.Timestamp)
            else None if x is None else x  # Explicitly check for None
        )
    )
    df.replace({np.nan: None}, inplace=True)
    json_df = df.to_dict(orient="records")
    return json_df


def png_to_base64(plt):
    img_str = None
    if plt:
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.clf()
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_str
