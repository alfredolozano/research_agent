# chain_setup.py

from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.utilities import PubMedAPIWrapper
from langchain import ArxivAPIWrapper, LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain.tools import StructuredTool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

from typing import Tuple, Dict  # Import Tuple and Dict from typing module


from langchain.prompts.chat import MessagesPlaceholder
import tools_wrappers

class Config():
    def __init__(self, openai_api_key):
        self.model = 'gpt-4'
        self.llm = ChatOpenAI(temperature=1, model=self.model, max_tokens=4000, openai_api_key=openai_api_key)

def setup_memory() -> Tuple[Dict, ConversationBufferMemory]:
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    return agent_kwargs, memory

def setup_agent(openai_api_key) -> AgentExecutor:
    cfg = Config(openai_api_key)
    duckduck_search = DuckDuckGoSearchAPIWrapper()
    wikipedia = WikipediaAPIWrapper()
    pubmed = PubMedAPIWrapper()
    events = tools_wrappers.EventsAPIWrapper()
    events.doc_content_chars_max = 5000
    llm_math_chain = LLMMathChain.from_llm(llm=cfg.llm, verbose=False)
    arxiv = ArxivAPIWrapper()
    
    tools = [
        Tool(
            name = "SearchDuckDuckGo",
            func=duckduck_search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        ),
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="useful when you need an answer about encyclopedic general knowledge"
        ),
        Tool(
            name="Arxiv",
            func=arxiv.run,
            description="useful when you need an answer about encyclopedic general knowledge"
        ),
        StructuredTool.from_function(
            func=events.run,
            name="Events",
            description="useful when you need an answer about meditation related events in the united kingdom"
        ),
        StructuredTool.from_function(
            func=pubmed.run, 
            name='PubMed',
            description='Useful tool for querying medical publications'
        )
    ]
    
    agent_kwargs, memory = setup_memory()
    return initialize_agent(
        tools, 
        cfg.llm, 
        agent=AgentType.OPENAI_FUNCTIONS, 
        verbose=False, 
        agent_kwargs=agent_kwargs,
        memory=memory
    )
