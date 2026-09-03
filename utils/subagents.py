from datetime import datetime
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
import os
import json
from tools.tools_spec import fetch_paper
from tools.schemas_spec import QueryFormat, ResponseFormat, EvaluationFormat
from typing import Dict

load_dotenv()

class subAgents :
    def __init__(self):

        self.ollama_url = os.getenv("OLLAMA_API_URL")
        self.model_name = os.getenv("MODEL_NAME")
        self.model_small = os.getenv("MODEL_SMALL")

        self.llm = Ollama(
            model=self.model_name,
            base_url=self.ollama_url,
            temperature=0.2,
            request_timeout=300.0,
            additional_kwargs={"keep_alive": "10m"},
        )
        self.slm = Ollama(
            model=self.model_small,
            base_url=self.ollama_url,
            temperature=0.1,
            request_timeout=300.0,
            additional_kwargs={"keep_alive": "10m"},
        )
        self.queryformat = json.dumps(QueryFormat.model_json_schema()["properties"], indent=2)

        self.argformat_agent = FunctionAgent(
            name="ArgFormatterAgent",
            description="Given an arxiv research request, return structured parameters for search",
            system_prompt=(
                "You are an expert arXiv query formatter. Convert the user prompt into a valid arXiv boolean query string.\n\n"
                f"Allowed Schema:\n{self.queryformat}\n\n"
                "arXiv Query Syntax Rules:\n"
                "- Field Prefixes:\n"
                "  * `ti:` for Title: For multi-word titles, connect words with `+` (e.g. `ti:knowledge+distillation` or `ti:attention+is+all+you+need`)\n"
                "  * `au:` for Author: Write the author name directly (e.g. `au:Geoffrey Hinton` or `au:Hinton`)\n"
                "  * `abs:` for Abstract text (e.g. `abs:graph+neural+network`)\n"
                "  * `cat:` for Subject Category (e.g. `cat:cs.AI`, `cat:cs.CL`, `cat:cs.CV`)\n"
                "  * `all:` for all fields (default if no specific field is specified)\n"
                "- Boolean Operators: MUST be UPPERCASE `AND`, `OR`, `ANDNOT`.\n"
                "- CRITICAL: Do NOT use quotes (`\"` or `'`). Connect title words with `+` directly.\n\n"
                "Examples:\n"
                "1. 'What is the last paper about RAG?' -> {\"query\": \"ti:RAG\", \"criteria\": \"submitteddate\"}\n"
                "2. 'Papers on knowledge distillation by Geoffrey Hinton' -> {\"query\": \"ti:knowledge+distillation AND au:Geoffrey Hinton\", \"criteria\": \"relevance\"}\n"
                "3. 'What is the most relevant paper about attention is all you need?' -> {\"query\": \"ti:attention+is+all+you+need\", \"criteria\": \"relevance\"}\n"
                "4. 'Recent papers on reinforcement learning in computer vision' -> {\"query\": \"ti:reinforcement+learning AND cat:cs.CV\", \"criteria\": \"submitteddate\"}\n\n"
                "Output strictly the JSON result adhering to QueryFormat schema without extra text."
            ),
            llm=self.llm,
            output_cls=QueryFormat,
        )

        self.retriever_agent = FunctionAgent(
            name="RetrieverAgent",
            description="Search a research paper according to the query and params",
            system_prompt=(
                "You are a silent tool-calling agent. Your ONLY task is to immediately execute the `fetch_paper` tool with the given query and criteria.\n\n"
                "CRITICAL RULES:\n"
                "1. NEVER output conversational text, thoughts, reasoning, or explanations.\n"
                "2. Call `fetch_paper(query=..., criteria=...)` directly with the EXACT parameters you received.\n"
                "3. Return strictly the ResponseFormat JSON object."
            ),
            llm=self.slm,
            tools=[fetch_paper],
            output_cls=ResponseFormat,
        )

        current_date_str = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year
        self.evalformat = json.dumps(EvaluationFormat.model_json_schema()["properties"], indent=2)

        self.evaluator_agent = FunctionAgent(
            name="EvaluatorAgent",
            description="Evaluate and give a bias-free feedback on an agent response based on the relevance of the response",
            system_prompt=(
                "You are an expert evaluator judge agent. You will evaluate the retrieved paper's relevance against the user research prompt.\n\n"
                f"Temporal Reference: The current real-world date is {current_date_str} (year {current_year}). "
                f"Any papers published up to {current_year} are valid, NOT in the future.\n\n"
                f"Target JSON Schema:\n{self.evalformat}\n\n"
                "Output Format Example:\n"
                '{"score": 9, "feedback": "The paper is directly relevant to knowledge distillation by Geoffrey Hinton."}\n\n'
                "CRITICAL RULES:\n"
                "1. Output ONLY a valid JSON object matching the schema above.\n"
                "2. Score must be an integer from 0 to 10.\n"
                "3. No explanations before or after the JSON."
            ),
            llm=self.llm,
            output_cls=EvaluationFormat,
        )


subagents = subAgents()


async def call_arg_agent(ctx: Context, prompt: str = "", **kwargs) -> str:
    """Step 1: Given a user prompt, create a json schema parameters query for the research agent."""
    user_prompt = prompt or kwargs.get("prompt") or kwargs.get("user_prompt", "")
    result = await subagents.argformat_agent.run(user_msg=f"Prepare the research query based on this prompt {user_prompt}")
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["query_args"] = str(result)
        ctx_state["state"]["user_prompt"] = str(user_prompt)
        # Clear state from previous queries to avoid feedback bleeding
        ctx_state["state"]["evaluation_content"] = None
        ctx_state["state"]["retrieved_paper"] = None

    return f"Step 1 Complete. Formatted query parameters: {result}\n\nAction required: Call `call_retriever_agent()` now to search for the paper."


async def call_retriever_agent(ctx: Context, **kwargs) -> str:
    """Step 2: Search and retrieve research papers using the parameters prepared in Step 1."""
    async with ctx.store.edit_state() as ctx_state:
        stored_args = ctx_state["state"].get("query_args", None)
        args = kwargs.get("query_args") or kwargs or stored_args

        if not args:
            return "No args to do the query. Please call call_arg_agent first."

        user_msg = f"Use the `fetch_paper` tool to search for the requested paper using these parameters: {args}\n\n"

        feedback = ctx_state["state"].get("evaluation_content", None)
        if feedback:
            user_msg += f"Feedback from the evaluator agent to improve the search: {feedback}\n\n"

        result = await subagents.retriever_agent.run(user_msg=user_msg)

        ctx_state["state"]["retrieved_paper"] = str(result)
        return f"Step 2 Complete. Paper retrieved: {result}\n\nAction required: Call `call_evaluator_agent()` now to evaluate the paper."


async def call_evaluator_agent(ctx: Context, paper_info: str = "", query: str = "", **kwargs) -> str:
    """Step 3: Evaluate the retrieved paper's relevancy against the user's initial research query."""
    async with ctx.store.edit_state() as ctx_state:
        retrieved_paper = paper_info or kwargs.get("paper_info") or ctx_state["state"].get("retrieved_paper", None)
        if not retrieved_paper:
            return "No paper retrieved to evaluate. Please call call_retriever_agent first."

        user_prompt = query or kwargs.get("query") or kwargs.get("user_prompt") or ctx_state["state"].get("user_prompt", None)
        if not user_prompt:
            return "Empty user prompt. Unable to give feedback."

        user_msg = f"Given the following retrieved paper informations, {retrieved_paper}, give a feedback of relevancy of the paper according to this: {user_prompt}"
        result = await subagents.evaluator_agent.run(user_msg=user_msg)

        ctx_state["state"]["evaluation_content"] = str(result)

        return f"Step 3 Complete. Evaluation review: {result}\n\nAction required: If score is 8 or higher (or max retries reached), write your final complete answer to the user. Otherwise recall `call_retriever_agent()`."



