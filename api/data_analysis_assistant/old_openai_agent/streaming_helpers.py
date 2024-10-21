from openai import OpenAI
from utils.utils import sanitize_text
import uuid
import json

client = OpenAI(
    api_key="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"
)


def call_api(name, messages, response_format, temperature=0):
    if response_format:
        return call_chat_parsed_api(name, messages, response_format, temperature)
    else:
        return call_chat_api(name, messages, temperature)


def call_chat_parsed_api(name, messages, response_format, temperature=0):
    print("call_chat_parsed_api", name)
    tool_uuid = str(uuid.uuid4())

    yield 'b:{{"toolCallId":"{id}","toolName":"{name}"}}\n'.format(
        id=tool_uuid, name=name
    )

    yield '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
        id=tool_uuid,
        name=name,
        args=json.dumps({"name": name}),
    )
    print("call_chat_parsed_api", name, "started")

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        response_format=response_format,
        temperature=temperature,
    )

    yield 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
        id=tool_uuid,
        name=name,
        args=json.dumps({"name": name}),
        result=json.dumps(
            {
                "output": completion.choices[0].message.parsed.dict(),
            }
        ),
    )

    print("call_chat_parsed_api", name, "done")
    # yield 'd:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}}}}\n'.format(
    #     reason="stop",
    #     prompt=completion.usage.prompt_tokens,
    #     completion=completion.usage.completion_tokens,
    # )


def call_chat_api(name, messages, temperature=0):
    tool_uuid = str(uuid.uuid4())

    completion = client.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=temperature, stream=True
    )

    # for chunk in completion:
    #     for choice in chunk.choices:
    #         if choice.finish_reason == "stop":
    #             continue
    #         else:
    #             print("yield 0", choice.delta.content)
    #             yield '0:"{text}"\n'.format(text=sanitize_text(choice.delta.content))

    #     if chunk.choices == []:
    #         yield 'd:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}}}}\n'.format(
    #             reason="stop",
    #             prompt=chunk.usage.prompt_tokens,
    #             completion=chunk.usage.completion_tokens,
    #         )

    output = ""

    yield 'b:{{"toolCallId":"{id}","toolName":"{name}"}}\n'.format(
        id=tool_uuid, name=name
    )

    count = 0

    for chunk in completion:
        for choice in chunk.choices:
            if choice.finish_reason == "stop":
                continue
            else:
                arguments = choice.delta.content
                if arguments is None:
                    arguments = ""

                yield 'c:{{"toolCallId":"{id}","argsTextDelta":"{argsTextDelta}"}}\n'.format(
                    id=tool_uuid,
                    argsTextDelta=sanitize_text(arguments),
                )

                output += arguments

        # if chunk.choices == []:
    yield '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
        id=tool_uuid,
        name=name,
        args=json.dumps(sanitize_text(output)),
    )
    yield 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
        id=tool_uuid,
        name=name,
        args=json.dumps(sanitize_text(output)),
        result=json.dumps(output),
    )
