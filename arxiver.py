import os
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import AgentStream, ToolCallResult
from llama_index.core.workflow import Context
import gradio as gr
from gradio import ChatMessage

from tools.tools_spec import fetch_paper, download_arxiv
from utils.utils import stream_from_agent

load_dotenv()


class arxivAgent:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_API_URL")

        self.llm = Ollama(
            model='mistral-small3.1:latest',
            base_url=self.ollama_url,
            request_timeout=120,
        )
        self.agent = self.define_agent()
        self.ctx = Context(self.agent)

    def define_agent(self):
        agent = FunctionAgent(
            tools=[fetch_paper, download_arxiv],
            llm=self.llm,
            system_prompt=
            "You are a research scientist agent that helps engineers find ARXIV papers based on their filters. Use the given tools to give correct and grounded answers. You should ALWAYS call the appropriate tool with corresponding parameters like query and criteria based on the user prompt value. ",
            verbose=False,
        )
        return agent

    async def query_eng(self, query: str) -> str:
        response = self.agent.run(user_msg=query, ctx=self.ctx)
        async for event in response.stream_events():

            if isinstance(event, ToolCallResult):
                print(f"\nCall {event.tool_name} with {event.tool_kwargs}\n Returned : {event.tool_output}")

            if isinstance(event,AgentStream):
                print(f"{event.delta}", end="", flush=True)
        final_response = await response
        return str(final_response)


async def process_query(message: str, history) -> str:
    arxivagent = arxivAgent()
    async for msg in stream_from_agent(arxivagent.agent, message, arxivagent.ctx):
            yield msg



async def main():
    app = gr.ChatInterface(
        process_query,
        chatbot=gr.Chatbot(
            label="Agent",
            avatar_images=(
                None,
                "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
            ),
        ),
        examples=[
            ["What is the last paper about RAG?"],
            ["What is the most relevant paper about knowledge distillation?"],
            ["What is the most relevant paper about agentic IA?"],
        ],
        title="ARXIV question based bot",
    )
    app.launch(debug=True)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
