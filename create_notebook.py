import json

# The content of the python files will be stored in this dictionary
# I'll populate this with the content I've already read.
files = {
    "model.py": """
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

llm_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)

embed_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
""",
    "ingestion.py": """
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from model import embed_model

def create_vectorstore():
    \"\"\"Create vector store only if it doesn't exist\"\"\"

    chroma_path = "./.chroma"
    if os.path.exists(chroma_path) and os.listdir(chroma_path):
        print("📚 Loading existing vector store...")
        vectorstore = Chroma(
            collection_name="rag-chroma",
            embedding_function=embed_model,
            persist_directory=chroma_path,
        )
        return vectorstore

    print("🔄 Creating new vector store...")

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embed_model,
        persist_directory=chroma_path,
    )

    print("✅ Vector store created and persisted!")
    return vectorstore
""",
    "graph/consts.py": """
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"
""",
    "graph/state.py": """
from typing import List, TypedDict


class GraphState(TypedDict):
    \"\"\"
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    \"\"\"

    question: str
    generation: str
    web_search: bool
    documents: List[str]
""",
    "graph/nodes/retrieve.py": """
from typing import Any, Dict
from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
""",
    "graph/nodes/grade_documents.py": """
from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    \"\"\"
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    \"\"\"

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
""",
    "graph/nodes/generate.py": """
from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
""",
    "graph/nodes/web_search.py": """
from typing import Any, Dict
from langchain.schema import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]

    documents = state.get("documents", [])

    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)

    if documents:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}
""",
    "graph/chains/router.py": """
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from model import llm_model

class RouteQuery(BaseModel):
    \"\"\"Route a user query to the most relevant datasource.\"\"\"

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

llm = llm_model

structured_llm_router = llm.with_structured_output(RouteQuery)

system = \"\"\"You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search.\"\"\"
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
""",
    "graph/chains/retrieval_grader.py": """
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from model import llm_model

llm = llm_model

class GradeDocuments(BaseModel):
    \"\"\"Binary score for relevance check on retrieved documents.\"\"\"

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = \"\"\"You are a grader assessing relevance of a retrieved document to a user question. \\n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \\n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\"\"\"
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \\n\\n {document} \\n\\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
""",
    "graph/chains/generation.py": """
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from model import llm_model

llm = llm_model

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
""",
    "graph/chains/hallucination_grader.py": """
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from model import llm_model

llm =  llm_model

class GradeHallucinations(BaseModel):
    \"\"\"Binary score for hallucination present in generation answer.\"\"\"

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = \"\"\"You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \\n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.\"\"\"
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \\n\\n {documents} \\n\\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
""",
    "graph/chains/answer_grader.py": """
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from model import llm_model


class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

llm =  llm_model

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = \"\"\"You are a grader assessing whether an answer addresses / resolves a question \\n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.\"\"\"
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \\n\\n {question} \\n\\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
""",
    "graph/graph.py": """
from langgraph.graph import END, StateGraph
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def route_question(state: GraphState) -> str:
    \"\"\"
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    \"\"\"

    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})

    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
"""
}

# The structure of the notebook
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source]
    }

# Build the notebook
notebook["cells"].append(create_markdown_cell("# Adaptive RAG on Google Colab"))
notebook["cells"].append(create_markdown_cell("This notebook runs an Adaptive RAG system on Google Colab. It's a conversion of the code from the [LangGraph AI repository](https://github.com/piyushagni5/langgraph-ai/tree/main/agentic-rag/agentic-rag-systems/building-adaptive-rag)."))
notebook["cells"].append(create_markdown_cell("## 1. Install Dependencies"))
notebook["cells"].append(create_code_cell("!pip install -q beautifulsoup4 langchain-community tiktoken langchainhub langchain langgraph tavily-python langchain-openai python-dotenv black isort pytest langchain-chroma langchain-tavily==0.1.5 langchain_aws langchain_google_genai"))
notebook["cells"].append(create_markdown_cell("## 2. Set Up Environment Variables\n\nYou need to provide API keys for Google and Tavily.\n\n- **`GOOGLE_API_KEY`**: Your Google API key for Gemini models. You can get one [here](https://aistudio.google.com/app/apikey).\n- **`TAVILY_API_KEY`**: Your Tavily API key for web search. You can get one [here](https://app.tavily.com/).\n- **`LANGCHAIN_API_KEY`** (Optional): Your LangSmith API key for tracing. You can get one [here](https://smith.langchain.com/)."))
notebook["cells"].append(create_code_cell(
"""import os
import getpass

os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")
os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API Key: ")

# Optional: LangSmith for tracing
use_langsmith = input("Do you want to use LangSmith for tracing? (yes/no): ").lower()
if use_langsmith == 'yes':
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangChain API Key: ")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "agentic-rag"
else:
    if "LANGCHAIN_API_KEY" in os.environ:
        del os.environ["LANGCHAIN_API_KEY"]
    if "LANGCHAIN_TRACING_V2" in os.environ:
        del os.environ["LANGCHAIN_TRACING_V2"]
    if "LANGCHAIN_ENDPOINT" in os.environ:
        del os.environ["LANGCHAIN_ENDPOINT"]
    if "LANGCHAIN_PROJECT" in os.environ:
        del os.environ["LANGCHAIN_PROJECT"]"""))
notebook["cells"].append(create_markdown_cell("## 3. Recreate the Project's File Structure\n\nThis will write the Python files from the original project to the Colab filesystem. This preserves the original, modular structure of the code."))
notebook["cells"].append(create_code_cell(
"""import os

os.makedirs("graph/chains", exist_ok=True)
os.makedirs("graph/nodes", exist_ok=True)"""))

for filepath, content in files.items():
    notebook["cells"].append(create_code_cell(f"%%writefile {filepath}\n{content}"))

# Add the final cells for ingestion and running the app
notebook["cells"].append(create_markdown_cell("## 4. Ingest Data"))
notebook["cells"].append(create_code_cell(
"""from ingestion import create_vectorstore
# Create the vector store
vectorstore = create_vectorstore()
# Create the retriever
retriever = vectorstore.as_retriever()
"""))

notebook["cells"].append(create_markdown_cell("## 5. Run the Adaptive RAG"))
notebook["cells"].append(create_code_cell(
"""from graph.graph import app
import pprint

def run_rag(question):
    \"\"\"Helper function to run the RAG graph and print the output.\"\"\"
    inputs = {"question": question}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Finished running: {key}:")
    pprint.pprint(value["generation"])

# Example usage:
run_rag("What is agent memory?")
"""))

# Write the notebook to a file
with open("Adaptive_RAG_on_Colab.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook 'Adaptive_RAG_on_Colab.ipynb' created successfully.")
