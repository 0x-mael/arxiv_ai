<!-- author : ABISSI Kotchikpa Ismael -->
<!-- date : September 2026 -->

# Arxiver - AI Research Assistant for arXiv

**Arxiver** is an agentic AI assistant designed to help researchers and engineers discover, explore, and download scientific papers from [arXiv](https://arxiv.org/). Powered by **LlamaIndex FunctionAgent**, local LLMs via **Ollama**, and an interactive **Gradio** web interface, it provides grounded answers with live, transparent tool execution tracking.

---

## Features

- **Intelligent arXiv Search**: Query scientific papers in natural language with automatic filtering by relevance, submission date, or last update date.
- **Agentic Tool Calling**: Uses LlamaIndex `FunctionAgent` workflows to dynamically decide when and how to call specialized tools.
- **Paper Download**: Automated retrieval and downloading of arXiv research papers.
- **Real-time Streaming & Collapsible Tool Logs**: Streamed responses in Gradio with persistent, collapsible accordions displaying the exact tool arguments and execution outputs.
- **Local & Privacy-Friendly**: Runs with local or remote open-weights models served through Ollama (e.g., `mistral-small3.1`).

---

## Project Architecture

```text
arxiver/
├── tools/
│   └── tools_spec.py      # arXiv API tool definitions (search, download)
├── utils/
│   └── utils.py           # Streaming handler converting agent events to Gradio ChatMessages
├── arxiver.py             # Main entry point & Gradio ChatInterface
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables
└── README.MD              # Project documentation
```

---

## Tech Stack

- **Agent Framework**: [LlamaIndex](https://www.llamaindex.ai/) (`FunctionAgent` / Workflows)
- **LLM Provider**: [Ollama](https://ollama.com/) (`mistral-small3.1:latest`)
- **User Interface**: [Gradio](https://www.gradio.app/) (with `gr.ChatMessage` structured UI)
- **arXiv Integration**: `arxiv` Python API

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running with your chosen model:
  ```bash
  ollama run mistral-small3.1:latest
  ```

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
# Clone repository
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

Create a `.env` file in the root directory (or update existing):

```env
OLLAMA_API_URL=http://localhost:11434
```

### 4. Running the Application

Launch the Gradio web interface:

```bash
python arxiver.py
```

The application will start locally at `http://127.0.0.1:7860`.

---

## Example Queries

- *"What is the last paper about RAG?"*
- *"What is the most relevant paper about knowledge distillation?"*
- *"What is the most recent research on Agentic AI?"*

---

## Author

**ABISSI Kotchikpa Ismael**  
*September 2026*
