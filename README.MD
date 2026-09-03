<!-- author : ABISSI Kotchikpa Ismael -->
<!-- date : September 2026 -->

# Arxiver - Multi-Agent AI Research Assistant for arXiv

**Arxiver** is a multi-agent AI research assistant designed to discover, evaluate, and download scientific papers from [arXiv](https://arxiv.org/). Built on **LlamaIndex FunctionAgent**, local LLMs via **Ollama**, **MLflow** tracing, and an interactive **Gradio** web interface, it coordinates specialized sub-agents with state management and automated relevance review.

---

## Features

- **Multi-Agent Orchestration**: A supervisor orchestrator coordinates specialized sub-agents across a structured research pipeline:
  - **ArgFormatterAgent**: Translates user prompts into structured search parameters (`QueryFormat`).
  - **RetrieverAgent**: Executes search queries on arXiv and extracts structured metadata (`ResponseFormat`).
  - **EvaluatorAgent**: Scores paper relevance (0 to 10) and provides refinement feedback (`EvaluationFormat`).
- **Self-Correction & Retry Mechanism**: Automatically refines and retries paper retrieval if the relevance score is below 8 (up to 2 retries).
- **Direct Paper Download**: Downloads PDF papers locally via `download_arxiv` on request.
- **MLflow Tracing & Observability**: Automated tracing of agent interactions, tool calls, and execution latencies using `mlflow.llama_index.autolog()`.
- **Real-time Streaming & Structured UI**: Live response streaming with collapsible accordions in Gradio displaying tool calls, arguments, and execution outputs.
- **Local & Privacy-Friendly**: Runs on local open-weights models served through Ollama.

---

## Project Architecture

```text
arxiver/
├── tools/
│   ├── schemas_spec.py    # Pydantic schemas (QueryFormat, ResponseFormat, EvaluationFormat)
│   └── tools_spec.py      # arXiv API functions (fetch_paper, download_arxiv)
├── utils/
│   ├── subagents.py       # Specialized sub-agents and orchestrator tool wrappers
│   └── utils.py           # Event stream handler for Gradio ChatMessage UI
├── arxiver.py             # Orchestrator agent, MLflow setup, and Gradio interface
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables
└── README.MD              # Project documentation
```

---

## Tech Stack

- **Agent Framework**: [LlamaIndex](https://www.llamaindex.ai/) (`FunctionAgent` / Workflows)
- **LLM Provider**: [Ollama](https://ollama.com/)
- **Observability**: [MLflow](https://mlflow.org/) (Tracing & LLM Autologging)
- **User Interface**: [Gradio](https://www.gradio.app/) (`ChatMessage` interface)
- **arXiv Integration**: `arxiv` Python API
- **Data Validation**: Pydantic

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with your model:
  ```bash
  ollama run llama3.1:8b
  ```

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone <repository-url>
cd arxiver

# Create and activate virtual environment
python -m venv myenv
# On Windows:
.\myenv\Scripts\activate
# On Linux/macOS:
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Configure your `.env` file in the root directory:

```env
OLLAMA_API_URL=http://localhost:11434
MODEL_NAME=gemma4:latest
MODEL_SMALL = llama3.1:8b
```

### 4. Running the Application

Launch the Gradio web interface:

```bash
python arxiver.py
```

To monitor traces and execution graphs in MLflow:

```bash
mlflow ui
```

The Gradio interface will be accessible at `http://127.0.0.1:7860` and the MLflow dashboard at `http://127.0.0.1:5000`.

---

## Example Queries

- *"What is the last paper about RAG?"*
- *"What is the most relevant paper about knowledge distillation ?"*
- *"What is the most relevant paper about knowledge distillation with Hintor as author ?"*
- *"Find the most recent paper about agentic AI and download it."*

---

## Author

**ABISSI Kotchikpa Ismael**  
*September 2026*
