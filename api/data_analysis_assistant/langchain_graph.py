import json
from dotenv import load_dotenv
from utils.utils import sanitize_text
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from api.data_analysis_assistant.tools import (
    retrieve_stock_data,
    code_check,
    filter_data,
    analyse_data,
    plot_data,
    decide_to_finish,
    identify_filters,
    GraphState,
)
from api.data_analysis_assistant.helpers import serialize_messages
from langchain_openai import ChatOpenAI
import uuid

load_dotenv(".env.local")

llm = ChatOpenAI(temperature=0, model="gpt-4o")


def assistant(state: GraphState):
    return llm.invoke(state["messages"])


workflow = StateGraph(GraphState)


# Define the nodes
workflow.add_node("retrieve_stock_data", retrieve_stock_data)  # generation solution
workflow.add_node("check_code", code_check)  # check code
workflow.add_node("filter_data", filter_data)  # check code
workflow.add_node("analyse_data", analyse_data)  # check code
workflow.add_node("plot_data", plot_data)  # check code
workflow.add_node("identify_filters", identify_filters)  # check code

# Build graph
workflow.add_edge(START, "retrieve_stock_data")
workflow.add_edge("retrieve_stock_data", "identify_filters")
workflow.add_edge("identify_filters", "filter_data")
workflow.add_edge("filter_data", "check_code")
workflow.add_edge("analyse_data", "check_code")
workflow.add_edge("plot_data", "check_code")
workflow.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "filter_data": "filter_data",
        "analyse_data": "analyse_data",
        "plot_data": "plot_data",
    },
)
# workflow.add_edge("check_code", END)

# workflow.add_node("assistant", assistant)  # generation solution
# workflow.add_edge(START, "assistant")
# workflow.add_edge("assistant", END)


memory = MemorySaver()
graph = workflow.compile()

# question = "Can you show me a projection of the income statement for AAPL over the next 5 years based on the last 5 years"
# messages_test = [AIMessage(content=question)]

# super messy. need to find a good way to clean this up
async def stream_graph(messages, protocol):
    async for event in graph.astream_events(
        {"messages": messages, "iterations": 0, "error": ""},
        version="v2",
    ):
        print(event)
        if event["event"] == "on_chat_model_start":
            name = event["metadata"].get("langgraph_node", "")
            id = event["run_id"]
            payload = 'b:{{"toolCallId":"{id}","toolName":"{name}"}}\n'.format(
                id=id, name=name
            )
            yield payload
        if event["event"] == "on_chat_model_stream":
            additional_kwargs = event["data"]["chunk"].additional_kwargs
            tool_call_chunk = additional_kwargs.get("tool_calls", [])
            stream_chunk = event["data"]["chunk"].content
            if stream_chunk:
                chunk = stream_chunk
            elif len(tool_call_chunk) > 0 and "function" in tool_call_chunk[0]:
                chunk = tool_call_chunk[0]["function"]["arguments"]
            else:
                chunk = ""
            payload = (
                'c:{{"toolCallId":"{id}","argsTextDelta":"{argsTextDelta}"}}\n'.format(
                    id=event["run_id"],
                    argsTextDelta=sanitize_text(chunk),
                )
            )
            yield payload
        if event["event"] == "on_chat_model_end":
            additional_kwargs = event["data"]["output"].additional_kwargs
            tool_call_chunk = additional_kwargs.get("tool_calls", [])
            stream_chunk = event["data"]["output"].content
            if stream_chunk:
                result = stream_chunk
            elif len(tool_call_chunk) > 0 and "function" in tool_call_chunk[0]:
                result = tool_call_chunk[0]["function"]["arguments"]
            payload = (
                '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
                    id=event["run_id"],
                    name=name,
                    args=json.dumps(result),
                )
            )
            yield payload
            payload = 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
                id=event["run_id"],
                name=name,
                args=json.dumps(result),
                result=json.dumps(result),
            )
            yield payload
        if event["event"] == "on_chain_end" and event["name"] == "check_code":
            if event["data"]["output"]["error_messages"]:
                output = serialize_messages(event["data"]["output"]["error_messages"])
                name = "check_code"
                payload = '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
                    id=event["run_id"], name=name, args=output
                )
                yield payload
                payload = 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
                    id=event["run_id"], name=name, args=output, result=output
                )
                yield payload
            else:
                output = json.dumps(event["data"]["output"]["json_df"])
                name = "dataframe"
                payload = '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
                    id=event["run_id"], name=name, args=output
                )
                yield payload
                payload = 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
                    id=event["run_id"], name=name, args=output, result=output
                )
                yield payload
                plt = event["data"]["output"].get("plot")
                if plt:
                    id = str(
                        uuid.uuid4()
                    )  # hack becuase the id is the same as the dataframe. Dataframe ideally should not be sent
                    output = json.dumps(plt)
                    name = "plot"
                    payload = '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
                        id=id, name=name, args=output
                    )
                    yield payload
                    payload = 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
                        id=id, name=name, args=output, result=output
                    )
                    yield payload

    yield 'd:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}}}}\n'.format(
        reason="stop",
        prompt=0,
        completion=0,
    )
