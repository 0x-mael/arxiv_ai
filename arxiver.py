import os
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import AgentStream, ToolCallResult
from llama_index.core.workflow import Context
import gradio as gr
from gradio import ChatMessage

import mlflow

from tools.tools_spec import download_arxiv
from utils.subagents import call_arg_agent, call_evaluator_agent, call_retriever_agent
from utils.utils import stream_from_agent

load_dotenv()
mlflow.set_experiment("Arxiv-Multi-Agent-Research")
mlflow.llama_index.autolog()

class arxivAgent:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_API_URL")
        self.model_name = os.getenv("MODEL_NAME", os.getenv("MODEL_SMALL","mistral:7b"))

        self.llm = Ollama(
            model=self.model_name,
            base_url=self.ollama_url,
            temperature=0.2,
            request_timeout=300.0,
            additional_kwargs={"keep_alive": "10m"},
        )
        self.orchestrator = self.define_agent()
        self.ctx = Context(self.orchestrator)

    def define_agent(self):
        orchestrator = FunctionAgent(
            tools=[call_arg_agent, call_retriever_agent, call_evaluator_agent, download_arxiv],
            llm=self.llm,
            system_prompt=(
                "You are an expert research scientist assistant that helps engineers search and download scientific papers from arXiv.\n\n"
                "ROUTING INSTRUCTIONS:\n"
                "A. If the user asks to SEARCH for a paper (or SEARCH AND DOWNLOAD):\n"
                "   1. Call `call_arg_agent` with the user prompt to format query parameters.\n"
                "   2. Call `call_retriever_agent` to search the paper with the exact parameters values you received from the previous agent.\n"
                "   3. Call `call_evaluator_agent` to review relevance.\n"
                "   4. If score < 8, recall `call_retriever_agent` with feedback (max 2 retries).\n"
                "   5. If score >= 8 (or after 2 retries), and if the user requested a download, call `download_arxiv(pdf_url=...)` with the paper's PDF URL.\n"
                "   6. Formulate the final complete answer with the paper title, authors, date, summary, and download status.\n\n"
                "B. If the user asks to DOWNLOAD a paper that was ALREADY discussed or provided in context:\n"
                "   - Do NOT run the search pipeline. Directly call `download_arxiv(pdf_url=...)` using the PDF URL from the conversation history, then confirm to the user.\n\n"
                "CRITICAL: Never output a direct final text answer to the user unless the 3 required steps are completed at least."
            ),
            verbose=True,
            initial_state={
                "query_args": None,
                "retrieved_paper": None,
                "evaluation_content": None,
            },
        )
        return orchestrator


    # async def query_eng(self, query: str) -> str:
    #     response = self.agent.run(user_msg=query, ctx=self.ctx)
    #     async for event in response.stream_events():

    #         if isinstance(event, ToolCallResult):
    #             print(f"\nCall {event.tool_name} with {event.tool_kwargs}\n Returned : {event.tool_output}")

    #         if isinstance(event,AgentStream):
    #             print(f"{event.delta}", end="", flush=True)
    #     final_response = await response
    #     return str(final_response)


arxivagent = arxivAgent()
async def process_query(message: str, history) -> str:
    async for msg in stream_from_agent(arxivagent.orchestrator, message,ctx=arxivagent.ctx):
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
