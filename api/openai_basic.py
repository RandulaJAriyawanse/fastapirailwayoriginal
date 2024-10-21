import os
import json
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from utils.prompt import ClientMessage
from utils.tools import get_current_weather
from utils.utils import sanitize_text
import time


load_dotenv(".env.local")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


available_tools = {
    "get_current_weather": get_current_weather,
}


def stream_text(messages: List[ClientMessage], protocol: str = "data"):
    stream = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        stream=True,
        stream_options={"include_usage": True},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location", "unit"],
                    },
                },
            }
        ],
    )

    # When protocol is set to "text", you will send a stream of plain text chunks
    # https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol#text-stream-protocol

    if protocol == "text":
        for chunk in stream:
            for choice in chunk.choices:
                if choice.finish_reason == "stop":
                    break
                else:
                    yield "{text}".format(text=choice.delta.content)

    # When protocol is set to "data", you will send a stream data part chunks
    # https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol#data-stream-protocol

    elif protocol == "data":
        draft_tool_calls = []
        draft_tool_calls_index = -1

        for chunk in stream:
            print("_______________________")
            print(chunk)
            for choice in chunk.choices:
                if choice.finish_reason == "stop":
                    continue

                elif choice.finish_reason == "tool_calls":
                    for tool_call in draft_tool_calls:
                        print("YIELD 9")
                        yield '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
                            id=tool_call["id"],
                            name=tool_call["name"],
                            args=json.dumps("testtest"),
                        )
                    for tool_call in draft_tool_calls:
                        tool_result = available_tools[tool_call["name"]](
                            **json.loads(tool_call["arguments"])
                        )
                        print("YIELD a")
                        yield 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
                            id=tool_call["id"],
                            name=tool_call["name"],
                            args=json.dumps("testtest"),
                            result=json.dumps("testtest"),
                        )

                elif choice.delta.tool_calls:
                    for tool_call in choice.delta.tool_calls:
                        id = tool_call.id
                        name = tool_call.function.name
                        arguments = tool_call.function.arguments

                        if id is not None:
                            draft_tool_calls_index += 1
                            draft_tool_calls.append(
                                {"id": id, "name": name, "arguments": ""}
                            )
                            print(
                                "draft_tool_calls append: ",
                                {"id": id, "name": name, "arguments": ""},
                            )
                            print("YIELD b")
                            yield 'b:{{"toolCallId":"{id}","toolName":"{name}"}}\n'.format(
                                id=id, name=name
                            )
                        else:
                            draft_tool_calls[draft_tool_calls_index][
                                "arguments"
                            ] += arguments
                            time.sleep(0.5)
                            print("draft_tool_calls append args: ", arguments)
                            print("type ", type(arguments))
                            print("YIELD c")
                            yield 'c:{{"toolCallId":"{id}","argsTextDelta":"{argsTextDelta}"}}\n'.format(
                                id=draft_tool_calls[draft_tool_calls_index]["id"],
                                argsTextDelta=sanitize_text("test"),
                            )

                else:
                    print("YIELD 0")
                    yield '0:"{text}"\n'.format(
                        text=sanitize_text(choice.delta.content)
                    )

            if chunk.choices == []:
                usage = chunk.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                print("#########################################")

                yield 'd:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}}}}\n'.format(
                    reason="tool-calls" if len(draft_tool_calls) > 0 else "stop",
                    prompt=prompt_tokens,
                    completion=completion_tokens,
                )
        # print("USAGE: ", stream.usage)
