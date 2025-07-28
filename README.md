# LangGraph AI Repository

A comprehensive collection of LangGraph implementations, tutorials, and advanced AI workflows covering Agentic RAG systems, MCP (Model Context Protocol) development, and practical AI application patterns.

## Overview

This repository serves as a implementation guide for building sophisticated AI applications using LangGraph. It contains practical examples, tutorials, and production-ready implementations across multiple domains:

- **Agentic RAG Systems**: Advanced retrieval-augmented generation with adaptive routing and self-correction mechanisms
- **MCP Development**: Complete Model Context Protocol server and client implementations
- **Workflow Patterns**: Orchestration patterns for complex AI workflows
- **Human-in-the-Loop Systems**: Interactive AI systems with human oversight
- **Advanced RAG Agents**: Sophisticated retrieval and generation systems

## Repository Structure

```
langgraph-ai/
├── agentic-rag/
│   ├── agentic-rag-systems/
│   │   └── building-adaptive-rag/
│   └── agentic-workflow-pattern/
│       ├── 1-prompting_chaining.ipynb
│       ├── 2-routing.ipynb
│       ├── 3-parallelization.ipynb
│       ├── 4-orchestrator-worker.ipynb
│       └── 5-Evaluator-optimizer.ipynb
├── mcp/
│   ├── 01-build-your-own-server-client/
│   ├── 02-build-mcp-client-with-multiple-server-support/
│   └── 03-build-mcp-server-client-using-sse/
├── langgraph-cookbook/
│   ├── human-in-the-loop/
│   │   ├── 01-human-in-the-loop.ipynb
│   │   ├── 02-human-in-the-loop.ipynb
│   │   └── 03-human-in-the-loop.ipynb
│   └── tool-calling -vs-react.ipynb
├── rag/
│   ├── Building an Advanced RAG Agent.ipynb
│   └── rag-as-tool-in-langgraph-agents.ipynb
├── .gitignore
├── .gitmodules
├── README.md
└── requirements.txt
```

## Prerequisites

Before setting up this repository, ensure you have the following installed:

- Python 3.10 or higher
- UV package manager (recommended) or pip
- Git

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/piyushagni5/langgraph-ai.git
cd langgraph-ai
```

### Step 2: Install UV Package Manager

If you haven't installed UV yet, install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 3: Create Virtual Environment

Navigate to the specific project directory you want to work with. For example, to work with the Adaptive RAG system:

```bash
cd agentic-rag/agentic-rag-systems/building-adaptive-rag
```

Create a virtual environment using UV:

```bash
uv venv --python 3.10
```

### Step 4: Activate Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### Step 5: Install Dependencies

**Using UV (Recommended):**
```bash
uv pip install -r requirements.txt
```

**Using pip (Alternative):**
```bash
pip install -r requirements.txt
```

### Step 6: Environment Configuration

Create a `.env` file in your project directory with the necessary API keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=your_project_name
```

**Note**: The `LANGCHAIN_API_KEY` is required if you enable tracing with `LANGCHAIN_TRACING_V2=true`.

## VS Code Integration

### Connecting Virtual Environment to VS Code

1. **Open VS Code** in your project directory:
   ```bash
   code .
   ```

2. **Open Command Palette** (Cmd+Shift+P on macOS, Ctrl+Shift+P on Windows/Linux)

3. **Select Python Interpreter**:
   - Type "Python: Select Interpreter"
   - Choose the interpreter from your virtual environment
   - Path will typically be: `./.venv/bin/python` (macOS/Linux) or `./.venv/Scripts/python.exe` (Windows)

4. **Verify Installation**:
   - Open a Python file or Jupyter notebook
   - Check the bottom-left corner of VS Code to confirm the correct interpreter is selected

### Alternative Method (Manual Path)

If the automatic detection doesn't work, you can manually specify the interpreter path:

1. Open Command Palette
2. Type "Python: Select Interpreter"
3. Choose "Enter interpreter path..."
4. Enter the full path to your virtual environment's Python executable

## Running Projects

### Adaptive RAG System

```bash
cd agentic-rag/agentic-rag-systems/building-adaptive-rag
uv run main.py
```

### Running Tests

```bash
uv run pytest . -s -v
```

### Jupyter Notebooks

For the tutorial notebooks, start Jupyter:

```bash
uv run jupyter lab
```

Or for classic Jupyter:

```bash
uv run jupyter notebook
```

## Key Features

### Adaptive RAG System
- Dynamic query analysis and routing
- Self-reflection mechanisms for improved answer quality
- Document relevance grading
- Hallucination detection and prevention
- Web search integration for knowledge gaps

### MCP Development
- Complete server and client implementations
- Multiple server support with configuration management
- Real-time communication via Server-Sent Events
- Integration with LangChain ecosystem

### Workflow Patterns
- Orchestrator-worker architectures
- Parallel processing implementations
- Human-in-the-loop workflows
- Evaluation and optimization patterns

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes and improvements
- New tutorial implementations
- Documentation enhancements
- Performance optimizations

## Dependencies

The main dependencies across projects include:

- **LangGraph**: Core workflow orchestration
- **LangChain**: AI application framework
- **LangChain Community**: Community integrations
- **Tavily**: Web search capabilities
- **Chroma**: Vector database
- **Pytest**: Testing framework
- **MCP**: Model Context Protocol

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Original LangChain Cookbook by Mistral AI and LangChain
- Built with LangGraph and the LangChain ecosystem
- Community contributions and feedback

## Support

For questions, issues, or contributions, please:

1. Check existing issues in the repository
2. Create a new issue with detailed information
3. Follow the contribution guidelines

---

**Note**: This repository contains multiple independent projects. Each project has its own requirements and setup instructions. Please refer to individual project README files for specific details.