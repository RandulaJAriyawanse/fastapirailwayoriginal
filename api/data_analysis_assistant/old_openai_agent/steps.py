from pydantic import BaseModel
from datetime import datetime as dt
from typing import Optional, List
import json
from api.data_analysis_assistant.old_openai_agent.helpers import (
    store_dataframe_info,
    filter_json_schema,
    extract_python_code,
    capture_parse_output,
    capture_output_temp,
)
from api.data_analysis_assistant.streaming_helpers import (
    call_chat_parsed_api,
    call_api,
)
from api.data_analysis_assistant.json_schema_raw import json_schema_raw

import io
import re
import traceback
import sys
import json
import os
from fastapi.responses import JSONResponse
import uuid
import pandas as pd


class DataAnalysisAssistant:
    def __init__(self):
        """
        Initializes the DataAnalysisAssistant with required configurations
        """
        self.json_schema = json_schema_raw

    def call_api_and_execute_with_fallback(
        self,
        name,
        messages,
        response_format=None,
        data=None,
        df=None,
        plot=None,
        max_attempts=3,  # need to make data more generic
    ):
        max_attempts = max_attempts
        attempts = 0

        while attempts < max_attempts:
            try:
                global_vars = globals()
                if data:
                    global_vars["data"] = data

                print("running call_api_and_execute_with_fallback messages")
                print(messages)

                result = call_api(name, messages, response_format=response_format)
                output = ""
                for res in result:
                    yield res
                    if response_format:
                        out = capture_parse_output(res)
                        if out and out["output"]:
                            output = capture_parse_output(res)["output"]
                    else:
                        out = capture_output_temp(res)
                        # print("out", out)
                        if out:
                            output = capture_output_temp(res)["output"]

                if response_format:
                    code = output["code"]
                elif extract_python_code(output):
                    print("Extracting python code")
                    code = extract_python_code(output)
                else:
                    code = output

                try:
                    print("\nCode: ", code)
                    exec(code, global_vars)
                except Exception as exec_error:
                    tb = sys.exc_info()[2]
                    extracted_tb = traceback.extract_tb(tb)
                    for frame in extracted_tb:
                        if frame.filename == "<string>":
                            print(
                                f"Error in dynamically executed code at line {frame.lineno}:\n{frame.line}"
                            )

                    raise exec_error  # Re-raise the error to be caught by the outer try-except
                if "df" in global_vars:
                    print("Code executed successfully")
                    df = global_vars["df"].applymap(
                        lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x
                    )
                    tool_uuid = str(uuid.uuid4())
                    yield '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
                        id=tool_uuid,
                        name="dataframe",
                        args=json.dumps({"name": "dataframe"}),
                    )
                    print("\ndf")
                    print(df.to_dict(orient="records"))
                    yield 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
                        id=tool_uuid,
                        name="dataframe",
                        args=json.dumps({"name": "dataframe"}),
                        result=json.dumps(
                            {
                                "output": df.to_dict(orient="records"),
                            }
                        ),
                    )
                    break
                else:
                    raise NameError(
                        "The variable 'df' was not created in the generated code."
                    )
            except Exception as e:
                error_traceback = traceback.format_exc()
                print(f"Attempt {attempts + 1} failed. Error:\n{e}")

                attempts += 1
                if attempts >= max_attempts:
                    print(f"Max attempts reached ({max_attempts}).")
                    raise RuntimeError(
                        f"Failed to execute code after {max_attempts} attempts."
                    )

                error_message = {
                    "role": "user",
                    "content": f"An error occurred on the previous code you generated for the same question. Here is the code:\n{code} \n\n Here id the error: {e}",
                }
                print("ERROR MESSAGE ")
                messages.append(error_message)

    def translate_question(self, question):
        """
        Translates complex questions into parts for multi-query data analysis
        """
        print("running translate_question")

        class QuestionTranslation(BaseModel):
            question_1: str
            question_2: Optional[str]
            question_3: Optional[str]

        messages = [
            {
                "role": "system",
                "content": "Do a simple translation of the user's query into 3 parts."
                "\nQuestion 1: Question to retrieve data to perform data analysis or answer the user's question"
                "\nQuestion 2: Question if an data analysis is required"
                "\nQuestion 3: Question to plot the data"
                "\n\nNote, Question 2 and Question 3 are optional and may not be requested, in which case just return and empty string",
            },
            {"role": "user", "content": question},
        ]

        yield from call_chat_parsed_api(
            "translate_question", messages, QuestionTranslation
        )

    def generate_filters(self, event, retry_messages=None):
        json_schema = json.loads(json_schema_raw)

        class Filters(BaseModel):
            data_filter: str
            metric_filter: Optional[List[str]]
            start_date: Optional[str]
            end_date: Optional[str]

        messages = [
            {
                "role": "system",
                "content": "Here is the output JSON schema for an API reponse:"
                f"\n{json_schema}"
                "\n\nBased on the schema, separate out the request into the following filters"
                "\nData_Filter: filter string for the requested data that separates the separate json layers with '::' such as 'Financials::Income_Statement::yearly'."
                "\nMetric_Filter: Optional list of strings that contains specific metrics the user may have requested. Return an empty list if the user has not specified metrics."
                "\nStart_Date: Optional start date if the user has requests a date range."
                "\nEnd_Date: Optional end date if the user has requests a date range."
                f"\n\nFor reference, the date today is {dt.today()}",
            },
            {"role": "user", "content": event},
        ]
        if retry_messages:
            messages.append(retry_messages)

        yield from call_chat_parsed_api("generate_filters", messages, Filters)

    def generate_filter_code(self, event, data, retry_messages=None):

        json_schema = json.loads(json_schema_raw)
        filtered_json_schema = filter_json_schema(json_schema, event["data_filter"])

        class FilterCode(BaseModel):
            code: str

        messages = [
            {
                "role": "system",
                "content": "You have access to some json called 'data' that follows the following schema:"
                f"\n{filtered_json_schema}"
                "\n\nReturn the code to convert the json into a dataframe and filter it based on the provided filters. "
                "\nAssume you have access to the 'data' variable. Do not create a sample dataset"
                "\nThe final dataframe you output should be called 'df' and created globally"
                "",
            },
            {
                "role": "user",
                "content": f"Here are the metrics to filter: {event['metric_filter']}, and here is the date range: {event['start_date']} {event['end_date']}",
            },
        ]
        if retry_messages:
            messages.extend(retry_messages)

        yield from self.call_api_and_execute_with_fallback(
            "generate_filter_code", messages, FilterCode, data=data
        )

    def analyse_data(self, question, df, retry_messages=None):

        print("running analyse_data")

        info_string = store_dataframe_info(df)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates python code to analyze a dataframe called 'df'."
                "\nHere is the df.info():"
                f"\n{info_string}"
                "\n\nConduct the analysis of the dataframe 'df' as if you already have it. Do not make a sample dataframe"
                "\nThe final dataframe you output should also be called 'df'",
            },
            {"role": "user", "content": question},
        ]
        if retry_messages:
            messages.append(retry_messages)

        yield from self.call_api_and_execute_with_fallback(
            "analyse_data", messages, df=df
        )

    def plot_data(self, question, df, retry_messages=None):

        info_string = store_dataframe_info(df)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates python code to plot a dataframe called 'df'."
                "\nHere is the df.info():"
                f"\n{info_string}"
                "\n\nPlot the dataframe 'df' as if you already have it. Do not make a sample dataframe",
            },
            {"role": "user", "content": question},
        ]
        if retry_messages:
            messages.extend(retry_messages)

        yield from self.call_api_and_execute_with_fallback("plot_data", messages, df=df)
