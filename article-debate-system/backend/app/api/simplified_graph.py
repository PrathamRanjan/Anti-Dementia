from langgraph.graph import StateGraph, END
from app.utils.models import DebateState, Argument
from app.agents.supervisor import SupervisorAgent
from app.agents.reader import ReaderAgent
from app.agents.writer import WriterAgent
from app.agents.fact_checker import FactCheckerAgent
import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

# Get API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

# Initialize agents
supervisor_agent = SupervisorAgent(GROQ_API_KEY)
reader_agent = ReaderAgent(GROQ_API_KEY)
pro_writer_agent = WriterAgent(GROQ_API_KEY, "pro")
con_writer_agent = WriterAgent(GROQ_API_KEY, "con")
fact_checker_agent = FactCheckerAgent(GROQ_API_KEY, GOOGLE_FACT_CHECK_API_KEY)

def create_debate_graph():
    """Create a very simple debate graph that just analyzes the article and ends"""
    
    # Define the state graph
    workflow = StateGraph(DebateState)
    
    # Define a simple node that just analyzes the article
    def analyze_article(state: DebateState) -> DebateState:
        print("Starting article analysis...")
        summary = reader_agent.analyze_article(state.article)
        state.summary = summary
        state.iteration_count = 1
        print("Analysis complete")
        return state
    
    # Add the node to the graph
    workflow.add_node("analyze_article", analyze_article)
    
    # Set the entry point
    workflow.set_entry_point("analyze_article")
    
    # Just go straight to the end
    workflow.add_edge("analyze_article", END)
    
    # Compile and return
    print("Compiling graph...")
    return workflow.compile()