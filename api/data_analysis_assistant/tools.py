from pydantic import BaseModel
import pandas as pd
from datetime import datetime as dt
from dotenv import load_dotenv
import traceback
from typing import Optional, List
import json
import sys
from typing import List
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated
import traceback
import json
from typing import List
import os
from pydantic import BaseModel
import pandas as pd
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from api.data_analysis_assistant.json_schema_raw import json_schema_raw

# from api.data_analysis_assistant.json_schema_historical_price import (
#     json_schema_historical_price,
# )
from langchain_openai import ChatOpenAI
from api.data_analysis_assistant.helpers import (
    get_info,
    filter_json_schema,
    retrieve_data,
    extract_python_code,
    png_to_base64,
    dataframe_to_json_serializable,
)
import math
import numpy as np
import io
import base64
from typing import Literal


load_dotenv(".env.local")

llm = ChatOpenAI(temperature=0, model="gpt-4o", stream_usage=True)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
        df: Dataframe of financial stock related data
    """

    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    iterations: int
    error_messages: Optional[List]
    data: object
    json: object
    metric_filter: Optional[List[str]]
    start_date: Optional[str]
    end_date: Optional[str]
    code: str
    df: pd.DataFrame
    json_df: object
    question_set: object
    current_question: int
    plot: Optional[str]


def retrieve_stock_data(state: GraphState) -> GraphState:
    """
    Retrieve technical and fundumantal data of stocks.
    """
    print("---RETRIEVING DATA---")

    class QuestionTranslation(BaseModel):
        question_1: str
        question_2: Optional[str]
        question_3: Optional[str]

    messages = [
        SystemMessage(
            content="Do a simple translation of the user's query into 3 parts."
            "\nQuestion 1: Question to retrieve data to perform data analysis or answer the user's question"
            "\nQuestion 2: Question to analyse data"
            "\nQuestion 3: Question to plot the data"
            "\n\nNote, Question 2 and Question 3 are optional and may not be requested, in which case just return and empty string",
        ),
    ]
    messages.extend(state["messages"][-5:])

    question_set = llm.with_structured_output(QuestionTranslation).invoke(messages)

    return {
        "error_messages": [],
        "question_set": question_set,
    }


def identify_filters(state: GraphState) -> GraphState:
    """
    Identify filters required from the user's question
    """
    print("---IDENTIFYING FILTERS---")

    class Filters(BaseModel):
        data_filter: str
        metric_filter: Optional[List[str]]
        start_date: Optional[str]
        end_date: Optional[str]

    json_schema = json.loads(json_schema_raw)

    messages = [
        SystemMessage(
            content="Here is the output JSON schema for an API reponse:"
            f"\n{json_schema}"
            "\n\nBased on the schema, separate out the request into the following filters"
            "\nData_Filter: filter string for the requested data that separates the separate json layers with '::' such as 'Financials::Income_Statement::yearly'."
            "\nMetric_Filter: Optional list of strings that contains specific metrics the user may have requested. Return an empty list if the user has not specified metrics."
            "\nStart_Date: Optional start date if the user has requests a date range."
            "\nEnd_Date: Optional end date if the user has requests a date range."
            f"\n\nFor reference, the date today is {dt.today()}"
        )
    ]
    messages.append(HumanMessage(content=(state["question_set"].question_1)))

    response = llm.with_structured_output(Filters).invoke(messages)

    filtered_json_schema = filter_json_schema(json_schema, response.data_filter)
    filtered_data = retrieve_data("CBA.json", response.data_filter)

    return {
        "data": filtered_data,
        "json": filtered_json_schema,
        "metric_filter": response.metric_filter,
        "start_date": response.start_date,
        "end_date": response.end_date,
        "current_question": 1,
    }


def filter_data(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Generates the code to filter the data based on the identifiedfilters
    """
    print("---FILTERING DATA---")

    class FilterCode(BaseModel):
        code: str

    json = state["json"]

    messages = [
        SystemMessage(
            content="You have access to some json called 'data' that follows the following json schema:"
            f"\n{json}"
            "\n\nReturn the code to convert the json into a dataframe and filter it based on the provided filters."
            "\nAssume you have access to the 'data' variable. Do not create a sample dataset"
            "\nThe final dataframe you output should be called 'df'"
        )
    ]
    messages.append(
        HumanMessage(
            content=f"Here are the metrics to filter: {state['metric_filter']}, and here is the date range: {state['start_date']} {state['end_date']}"
        )
    )
    if state["error_messages"]:
        messages.extend(state["error_messages"])

    response = llm.with_structured_output(FilterCode).invoke(messages, config)

    return {
        "code": response.code,
    }


def analyse_data(state: GraphState, config: RunnableConfig):
    """
    Analyse data
    """
    print("---ANALYSING DATA---")
    df_info = get_info(state["df"])

    messages = [
        SystemMessage(
            content="You are a helpful assistant that generates python analyse dataframe (without any plotting)."
            " Keep your analysis as simple as possible while still answering the user's question."
            "\nStick to the basic python librares - pandas, numpy, scikit-learn"
            "\nHere is the df.info() of the dataframe you are analysing:"
            f"\n{df_info}"
            "\n\nHere are some rules that are importants:"
            "\n- Make sure sure to include all the necessary imports at the top of the code"
            "\n- You already have the dataframe called 'df'. Do not make a sample dataframe"
            "\n- The final dataframe you output should also be called 'df', overriding the original dataframe",
        )
    ]
    messages.append(HumanMessage(content=state["question_set"].question_2)),
    if state["error_messages"]:
        messages.extend(state["error_messages"])

    response = llm.invoke(messages, config)

    code = extract_python_code(response.content)

    return {
        "code": code,
        "current_question": 2,
    }


def plot_data(state: GraphState):
    """
    Plot dat
    """
    print("---PLOTTING DATA---")

    df_info = get_info(state["df"])

    messages = [
        SystemMessage(
            content="You are a helpful assistant that generates python code to plot a dataframe called 'df'."
            "\nStick to matplotlib for plotting"
            "\nHere is the df.info():"
            f"\n{df_info}"
            "\n\n"
            "\n\nHere are some rules that are importants to follow:"
            "\n- Include all the necessary imports at the top of the code"
            "\n -Include triple backticks when generating code."
            "\n- Never use plt.show(), instead just save the plot to a file called 'plot.png'"
            "\n- You already have the dataframe called 'df'. Do not make a sample dataframe"
        )
    ]
    messages.append(HumanMessage(content=state["question_set"].question_3)),
    if state["error_messages"]:
        messages.extend(state["error_messages"])

    response = llm.invoke(messages)

    code = extract_python_code(response.content)

    return {
        "code": code,
        "current_question": 3,
    }


def code_check(state: GraphState):
    """
    Runs code and returns error messages to regenerate code with error context
    """

    print("---CHECKING CODE---")

    try:
        data = state["data"]
        df = state.get("df", None)
        plt = state.get("plt", None)
        local_scope = {"data": data, "df": df, "plt": None}

        if not state.get("code"):
            raise NameError("Code not found")
        exec(state["code"], local_scope, local_scope)

        df = local_scope.get("df")
        json_df = dataframe_to_json_serializable(df)
        plt = local_scope.get("plt")
        base64_plot = png_to_base64(plt)
    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()  # Get exception details
        frame = tb.tb_frame  # Get the current frame
        lineno = tb.tb_lineno  # Extract the line number where the error occurred
        line = traceback.extract_tb(tb)[-1].line  # Extract the actual line of code

        print("---CODE BLOCK CHECK: FAILED---")
        print(f"Error Type: {type(e).__name__}")  # Print the error type
        print(f"Error Message: {e}")  # Print the error message
        print(f"Error Line: {lineno}")  # Print the line number
        print(f"Code causing the error: {line}")  # Print the line of code
        code_message = AIMessage(content=(state["code"]))
        error_message = HumanMessage(content=(f"Your solution failed to execute: {e}"))
        error_messages = [code_message, error_message]
        return {
            "error_messages": error_messages,
            "iterations": state["iterations"] + 1,
        }

    return {
        "df": df,
        "json_df": json_df,
        "iterations": 0,
        "error_messages": [],
        "plot": base64_plot,
    }


def decide_to_finish(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    if state["error_messages"]:
        if state["iterations"] <= 2:
            return {1: "filter_data", 2: "analyse_data", 3: "plot_data"}.get(
                state["current_question"], "end"
            )
        return "end"

    if state["current_question"] == 1:
        return (
            "analyse_data"
            if state["question_set"].question_2
            else "plot_data" if state["question_set"].question_3 else "end"
        )

    if state["current_question"] == 2 and state["question_set"].question_3:
        return "plot_data"

    print("END")
    return "end"


# def route_to_pricing_or_fundamental(state: GraphState):
#     """
#     Route to the pricing or fundamental data analysis based on the user's question
#     """
#     print("---ROUTING TO PRICING OR FUNDAMENTAL DATA---")

#     class Route(BaseModel):
#         route: Literal["yes", "no"]

#     messages = [
#         SystemMessage(
#             content="If the user has requested historical price data, return yes, otherwise return no"
#         )
#     ]
#     messages.append(HumanMessage(content=(state["question_set"].question_1)))

#     response = llm.with_structured_output(Route).invoke(messages)

#     route = "get_historical_price" if response.route == "yes" else "identify_filters"

#     return route


# def get_historical_price(state: GraphState) -> GraphState:
#     """
#     Identify filters required from the user's question
#     """
#     print("---IDENTIFYING FILTERS---")

#     class FilterCode(BaseModel):
#         code: str

#     json_schema = json.loads(json_schema_historical_price)

#     messages = [
#         SystemMessage(
#             content="Here is the output JSON schema for an API reponse:"
#             f"\n{json_schema}"
#             "\n\nGenerate the python code to filter the data based on the user's request"
#             "\nAssume you have access to the 'data' variable. Do not create a sample dataset"
#             "\nThe final dataframe you output should be called 'df'"
#         )
#     ]
#     messages.append(HumanMessage(content=(state["question_set"].question_1)))

#     response = llm.with_structured_output(FilterCode).invoke(messages)

#     filtered_data = retrieve_data("CBA_historical_price.json", response.data_filter)

#     return {
#         "data": filtered_data,
#         "current_question": 1,
#     }
