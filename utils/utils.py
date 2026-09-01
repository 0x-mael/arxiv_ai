import json
from typing import AsyncGenerator, List, Optional
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, AgentStream
from llama_index.core.workflow import Context
from gradio import ChatMessage


async def stream_from_agent(
    agent: FunctionAgent, prompt: str, ctx: Optional[Context] = None
) -> AsyncGenerator[List[ChatMessage], None]:
    """Runs a LlamaIndex FunctionAgent with the given prompt and streams

    messages as a list of Gradio ChatMessage objects.

    Tool calls remain persistent in the chat as collapsible dropdown accordions
    even when the final text response is being streamed.
    """
    handler = agent.run(user_msg=prompt, ctx=ctx)
    messages: List[ChatMessage] = []
    text_message: Optional[ChatMessage] = None

    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            # Formate les arguments et la sortie de l'outil
            args_str = json.dumps(event.tool_kwargs, indent=2, ensure_ascii=False)
            if isinstance(event.tool_output, (dict, list)):
                out_str = json.dumps(event.tool_output, indent=2, ensure_ascii=False)
            else:
                out_str = str(event.tool_output)

            tool_content = (
                f"**Arguments:**\n```json\n{args_str}\n```\n\n"
                f"**Results:**\n```json\n{out_str}\n```"
            )

            tool_message = ChatMessage(
                role="assistant",
                content=tool_content,
                metadata={"title": f"🛠️ Tool called : {event.tool_name}", "status": "done"},
            )
            messages.append(tool_message)
            yield list(messages)

        elif isinstance(event, AgentStream):
            if text_message is None:
                text_message = ChatMessage(role="assistant", content=event.delta)
                messages.append(text_message)
            else:
                text_message.content += event.delta
            yield list(messages)

    # Récupération et résolution complète du résultat final
    final_response = await handler
    if text_message is None:
        messages.append(ChatMessage(role="assistant", content=str(final_response)))
        yield list(messages)

