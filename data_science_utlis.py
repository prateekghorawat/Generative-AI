"""Agent functionality."""
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor
import pandas as pd
import os


os.environ["OPENAI_API_KEY"] = "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"


PROMPT = (
    "If you do not know the answer, say you don't know.\n"
    "Think step by step.\n"
    "\n"
    "Below is the query.\n"
    "Query: {query}\n"
)

def create_agent(csv_file: str) -> AgentExecutor:
    """
    Create data agent.

    Args:
        csv_file: The path to the CSV file.

    Returns:
        An agent executor.
    """
    llm = OpenAI()
    df = pd.read_csv(csv_file)
    return create_pandas_dataframe_agent(llm, df, verbose=True ,allow_dangerous_code=True)


def query_agent(agent: AgentExecutor, query: str) -> str:
    """Query an agent and return the response."""
    prompt = PromptTemplate(template=PROMPT, input_variables=["query"])
    return agent.run(prompt.format(query=query))