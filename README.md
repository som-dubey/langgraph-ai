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

## Jupyter Notebook Kernel Setup

### Adding Virtual Environment to Jupyter Kernel

To use your UV virtual environment with Jupyter notebooks, you need to install ipykernel and register the environment as a kernel:

1. **Activate your virtual environment**:
   ```bash
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate     # Windows
   ```

2. **Install ipykernel in the virtual environment**:
   ```bash
   uv pip install ipykernel
   ```

3. **Register the virtual environment as a Jupyter kernel**:
   ```bash
   python -m ipykernel install --user --name=langgraph-ai --display-name="LangGraph AI"
   ```

4. **Verify kernel installation**:
   ```bash
   jupyter kernelspec list
   ```

5. **Start Jupyter and select the kernel**:
   ```bash
   uv run jupyter lab
   # or
   uv run jupyter notebook
   ```

   When you open a notebook, you can select the "LangGraph AI" kernel from the kernel menu.

### Managing Kernels

**List all available kernels**:
```bash
jupyter kernelspec list
```

**Remove a kernel** (if needed):
```bash
jupyter kernelspec remove langgraph-ai
```

**Alternative Method Using UV Run**:
You can also run Jupyter directly with UV without installing the kernel:
```bash
uv run jupyter lab
```
This will automatically use the Python interpreter from your virtual environment.

## VS Code Jupyter Kernel Selection

### Selecting Kernel in VS Code

When working with Jupyter notebooks in VS Code, you need to select the correct kernel to use your UV virtual environment:

1. **Open a Jupyter notebook** in VS Code (`.ipynb` file)

2. **Select the kernel**:
   - Look at the top-right corner of the notebook
   - Click on the kernel selector (it might show "Select Kernel" or the current kernel name)
   - Choose "LangGraph AI" from the dropdown list

3. **Alternative method**:
   - Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux) to open Command Palette
   - Type "Python: Select Notebook Kernel"
   - Choose "LangGraph AI" from the available kernels

4. **Verify kernel selection**:
   - The kernel name should appear in the top-right corner of the notebook
   - You should see "LangGraph AI" displayed

### If Kernel is Not Visible

If the "LangGraph AI" kernel doesn't appear in the list:

1. **Refresh kernel list**:
   - Open Command Palette (`Cmd+Shift+P` or `Ctrl+Shift+P`)
   - Type "Python: Refresh Kernel List"
   - Select this command

2. **Restart VS Code**:
   - Close VS Code completely
   - Reopen VS Code and the notebook
   - Try selecting the kernel again

3. **Verify kernel installation**:
   ```bash
   jupyter kernelspec list
   ```
   Make sure "LangGraph AI" appears in the list. If not, refer to the "Managing Kernels" section above.

### Using Python Interpreter Instead

If you prefer to use the Python interpreter directly without creating a kernel, follow the steps in the "VS Code Integration" section above to select your virtual environment's Python interpreter. VS Code will automatically use the selected interpreter for notebooks.

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

For the tutorial notebooks, you can start Jupyter using the methods described in the "Jupyter Notebook Kernel Setup" section above. The recommended approach is to use the registered kernel:

```bash
uv run jupyter lab
```

Or for classic Jupyter:

```bash
uv run jupyter notebook
```

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