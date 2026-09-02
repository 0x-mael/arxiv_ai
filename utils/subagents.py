from datetime import datetime
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
import os
from tools.tools_spec import fetch_paper
from tools.schemas_spec import QueryFormat, ResponseFormat, EvaluationFormat
from typing import Dict

load_dotenv()

class subAgents :
    def __init__(self):

        self.ollama_url = os.getenv("OLLAMA_API_URL")
        self.model_name = os.getenv("MODEL_NAME")

        self.llm = Ollama(
            model=self.model_name,
            base_url=self.ollama_url,
            temperature=0.2,
            request_timeout=120,
        )

        self.argformat_agent = FunctionAgent(
            name="ArgFormatterAgent",
            description="Given an arxiv query, return params for the api call",
            system_prompt=(
                "You are a formatter agent. You should return the structured query for advanced search if needed according to user prompt and the query format. "
                "Instructions:\n"
                "1. Query format should STRICTLY look like ex : {query:'RAG',query_prefix:'ti:', criteria : 'relevance'}\n"
                "2. The query_prefix arg should be one choice between pydantic queryformat hinted ones ONLY according to the search query.\n"
                "3. You should only return the result. No text BEFORE or AFTER the json result."
            ),
            llm=self.llm,
            output_cls=QueryFormat,
        )

        self.retriever_agent = FunctionAgent(
            name="RetrieverAgent",
            description="Search a research paper according to the query and params",
            system_prompt=(
                "You are a retriever agent. Given a json schema with query params, you will call the `fetch_paper` tool and format the response using the response format schema.\n"
                "Instructions:\n"
                "1. Always use `fetch_paper` to search for papers with the user's query prefixed or not with the given wildcard and appropriate sorting criteria ('relevance', 'submitteddate', or 'lastUpdateddate').\n"
                "2. If advanced search is needed, you should use the given prefix to enrich the query and get a better response.\n"
                "3. You should give a final response using strictly the responseFormat schema.\n"
                "4. If the evaluator agent gives you feedback, you should retry by taking the feedback as advanced context to get a better answer."
            ),
            llm=self.llm,
            tools=[fetch_paper],
            output_cls=ResponseFormat,
        )

        current_date_str = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year

        self.evaluator_agent = FunctionAgent(
            name="EvaluatorAgent",
            description="Evaluate and give a bias-free feedback on an agent response based on the relevance of the response",
            system_prompt=(
                "You are a judge agent. You will give feedback on another agent's work.\n"
                f"Temporal Reference: The current real-world date is {current_date_str} (year {current_year}). "
                f"Any scientific papers published up to this date (including {current_year}) are valid and published research, NOT future papers.\n"
                "Instructions:\n"
                "1. Format your review using the evaluationFormat pydantic schema.\n"
                "2. Give a score from 0 to 10 based on the topical relevance to the user's research query.\n"
                "3. Give concise feedback on what to do to get a better answer if score is under 8."
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
    return f"Query format parameters generated: {result}\n\n[SYSTEM NOTICE: Step 1 complete. You MUST now proceed to Step 2 and call call_retriever_agent(). Do NOT output a final response to the user yet.]"


async def call_retriever_agent(ctx: Context, **kwargs) -> str:
    """Step 2: Search and retrieve research papers using the parameters prepared in Step 1."""
    async with ctx.store.edit_state() as ctx_state:
        args = kwargs or query_args or ctx_state["state"].get("query_args", None)

        if not args:
            return "No args to do the query. Please call call_arg_agent first."

        user_msg = f"Use the `fetch_paper` tool to search for the requested paper using these parameters: {args}\n\n"

        feedback = ctx_state["state"].get("evaluation_content", None)
        if feedback:
            user_msg += f"Feedback of the evaluator agent : {feedback}\n\n"

        result = await subagents.retriever_agent.run(user_msg=user_msg)

        ctx_state["state"]["retrieved_paper"] = str(result)
        return f"Paper retrieved: {result}\n\n[SYSTEM NOTICE: Step 2 complete. You MUST now proceed to Step 3 by calling call_evaluator_agent(). Do NOT output a final response to the user yet.]"


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

        return f"Evaluation review: {result}\n\n[SYSTEM NOTICE: Step 3 complete. If score >= 8 or 2 retries reached, present your final complete response to the user. Otherwise call call_retriever_agent again.]"

