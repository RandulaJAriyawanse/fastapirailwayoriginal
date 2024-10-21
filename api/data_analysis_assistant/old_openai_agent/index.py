import os
import json
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from utils.prompt import ClientMessage
from utils.tools import get_current_weather
from utils.utils import sanitize_text
import time
from api.data_analysis_assistant.old_openai_agent.steps import DataAnalysisAssistant
from api.data_analysis_assistant.old_openai_agent.helpers import (
    retrieve_data,
    execute_code_with_fallback,
    capture_parse_output,
    capture_output,
    extract_python_code,
    capture_output_temp,
)
import pandas as pd
import uuid


import io
import re
import traceback
import sys
import json
import os


load_dotenv(".env.local")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

available_tools = {
    "get_current_weather": get_current_weather,
}


def stream_text(messages: List[ClientMessage], protocol: str = "data"):
    assistant = DataAnalysisAssistant()

    translate_gen = assistant.translate_question(
        "Can you show me income statement for CBA projected over the next 5 years based on the last 5 years"
    )

    output_1 = None
    output_2 = None

    for result in translate_gen:
        yield result  # Continue yielding the stream data as is
        out = capture_parse_output(result)
        if out and out["output"]:
            output_1 = out["output"]
            break

    # Step 2: Pass question_1 from translate_result to generate_filters
    if output_1:
        question_1 = output_1["question_1"]  # Extract question_1 from the parsed result

        generate_filters_gen = assistant.generate_filters(
            question_1
        )  # Call generate_filters with question_1
        for result in generate_filters_gen:
            yield result
            out = capture_parse_output(result)
            if out and out["output"]:
                output_2 = out["output"]
                break

    data = retrieve_data(output_2)

    generate_filter_code_gen = assistant.generate_filter_code(output_2, data)
    for result in generate_filter_code_gen:
        yield result
        out = capture_parse_output(result)
        if out and out["output"]:
            if out["toolName"] == "dataframe":
                json_df = out
                df = pd.DataFrame(json_df)
                break
            else:
                code = out["output"]
                # break

    print("question 2 df \n", df)

    if output_1["question_2"]:
        question_2 = output_1["question_2"]
        analyse_data_gen = assistant.analyse_data(question_2, df)
        for result in analyse_data_gen:
            yield result
            out = capture_output_temp(result)
            if out and out["output"]:
                if out["toolName"] == "dataframe":
                    json_df = out["output"]
                    df = pd.DataFrame(json_df)
                    break

    print("question 3 df \n", df)
    if output_1["question_3"]:
        question_3 = output_1["question_3"]
        generate_plot_gen = assistant.plot_data(question_3, df)
        for result in generate_plot_gen:
            yield result
