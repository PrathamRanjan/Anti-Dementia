from langgraph.graph import StateGraph, END
from app.utils.models import DebateState, Argument
from app.agents.supervisor import SupervisorAgent
from app.agents.reader import ReaderAgent
from app.agents.writer import WriterAgent
from app.agents.fact_checker import FactCheckerAgent
import os
from dotenv import load_dotenv
from typing import Dict, Any, Union, TypedDict, cast, List, Tuple

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
    """Create the debate graph with all agents"""
    
    # Define the state graph with config
    debate_graph = StateGraph(DebateState, {"recursion_limit": 10})
    
    # Define nodes
    
    # Read and analyze the article
    def analyze_article(state: Any) -> DebateState:
        # Handle input flexibility
        if not isinstance(state, DebateState):
            # If somehow we get a dict or other type
            raise TypeError(f"Expected DebateState, got {type(state)}")
            
        summary = reader_agent.analyze_article(state.article)
        state.summary = summary
        
        # Initialize iteration counter
        state.iteration_count = 0
            
        return state
    
    # Generate pro argument
    def generate_pro_argument(state: Any) -> Dict:
        # Handle input flexibility
        if not isinstance(state, DebateState):
            # If somehow we get a dict or other type
            raise TypeError(f"Expected DebateState, got {type(state)}")
            
        # Increment iteration counter
        state.iteration_count += 1
        
        # Force end debate if iteration count exceeds limit (extra safety)
        if state.iteration_count >= 3:
            print(f"Ending debate in pro_argument due to iteration limit ({state.iteration_count})")
            state.is_active = False
            return {"state": state, "argument": None}
            
        # Get the most recent user input if available
        user_input = state.user_inputs[-1] if state.user_inputs else ""
        
        argument = pro_writer_agent.create_argument(
            article_summary=state.summary,
            previous_arguments=state.arguments,
            user_input=user_input,
            argument_number=state.pro_count + 1
        )
        
        return {"state": state, "argument": argument}
    
    # Generate con argument
    def generate_con_argument(state: Any) -> Dict:
        # Handle input flexibility
        if not isinstance(state, DebateState):
            # If somehow we get a dict or other type
            raise TypeError(f"Expected DebateState, got {type(state)}")
            
        # Increment iteration counter
        state.iteration_count += 1
        
        # Force end debate if iteration count exceeds limit (extra safety)
        if state.iteration_count >= 3:
            print(f"Ending debate in con_argument due to iteration limit ({state.iteration_count})")
            state.is_active = False
            return {"state": state, "argument": None}
            
        # Get the most recent user input if available
        user_input = state.user_inputs[-1] if state.user_inputs else ""
        
        argument = con_writer_agent.create_argument(
            article_summary=state.summary,
            previous_arguments=state.arguments,
            user_input=user_input,
            argument_number=state.con_count + 1
        )
        
        return {"state": state, "argument": argument}
    
    # Fact check argument
    def fact_check_argument(inputs: Any) -> Dict:
        # Handle different input types
        if isinstance(inputs, dict) and "state" in inputs and "argument" in inputs:
            # Normal case: dictionary with state and argument
            state = inputs["state"]
            argument = inputs["argument"]
            
            # Check if we have a null argument (from safety termination)
            if argument is None:
                return {
                    "state": state, 
                    "argument": Argument(
                        content="Debate ended due to iteration limit",
                        position="pro", 
                        number=0,
                        verified=True
                    ),
                    "is_verified": True, 
                    "feedback": "Debate terminated due to safety limits"
                }
        elif isinstance(inputs, DebateState):
            # Edge case: When passed directly from certain nodes
            # Create a mock argument for testing
            state = inputs
            argument = Argument(
                content="Mock argument for testing",
                position="pro", 
                number=1,
                verified=False
            )
        else:
            # Unexpected input format
            raise TypeError(f"Unexpected input type for fact_check_argument: {type(inputs)}")
        
        is_verified, feedback, updated_argument = fact_checker_agent.verify_argument(argument)
        
        return {
            "state": state, 
            "argument": updated_argument, 
            "is_verified": is_verified, 
            "feedback": feedback
        }
    
    # Process verified argument
    def process_verified_argument(inputs: Any) -> DebateState:
        # Handle different input types
        if isinstance(inputs, dict) and "state" in inputs and "argument" in inputs:
            # Normal case: dictionary with state and argument
            state = inputs["state"]
            argument = inputs["argument"]
        elif isinstance(inputs, DebateState):
            # Edge case: this shouldn't normally happen but just in case
            # We can't proceed meaningfully without an argument
            return inputs
        else:
            # Unexpected input format
            raise TypeError(f"Unexpected input type for process_verified_argument: {type(inputs)}")
        
        # Add the argument to the state
        state.arguments.append(argument)
        
        # Update the appropriate counter
        if argument.position == "pro":
            state.pro_count += 1
        else:
            state.con_count += 1
        
        # Switch turns
        state.current_turn = "con" if state.current_turn == "pro" else "pro"
        
        return state
    
    # Revise failed argument
    def revise_argument(inputs: Any) -> Dict:
        # Handle different input types
        if isinstance(inputs, dict) and "state" in inputs and "argument" in inputs and "feedback" in inputs:
            # Normal case with all expected keys
            state = inputs["state"]
            argument = inputs["argument"]
            feedback = inputs["feedback"]
        elif isinstance(inputs, dict) and "state" in inputs and "argument" in inputs:
            # Partial case - missing feedback
            state = inputs["state"]
            argument = inputs["argument"]
            feedback = "Please revise this argument for factual accuracy."
        elif isinstance(inputs, DebateState):
            # Edge case: this shouldn't normally happen
            # We can't proceed meaningfully without an argument and feedback
            # Return a stub response
            return {
                "state": inputs,
                "argument": Argument(
                    content="Mock revised argument for testing",
                    position="pro", 
                    number=1,
                    verified=False
                )
            }
        else:
            # Unexpected input format
            raise TypeError(f"Unexpected input type for revise_argument: {type(inputs)}")
        
        if argument.position == "pro":
            revised_argument = pro_writer_agent.revise_argument(argument, feedback)
        else:
            revised_argument = con_writer_agent.revise_argument(argument, feedback)
        
        return {"state": state, "argument": revised_argument}
    
    # Wait for user input
    def wait_for_user_input(state: Any) -> DebateState:
        # Handle input flexibility
        if not isinstance(state, DebateState):
            # If somehow we get a dict or other type
            raise TypeError(f"Expected DebateState, got {type(state)}")
            
        # This is a placeholder - in the actual API, we'll pause here for user input
        return state
    
    # Check debate status
    def check_debate_status(state: Any) -> DebateState:
        # Handle input flexibility
        if not isinstance(state, DebateState):
            # If somehow we get a dict or other type
            raise TypeError(f"Expected DebateState, got {type(state)}")
        
        # Force end debate if iteration count exceeds limit (safety mechanism)
        if state.iteration_count >= 3:
            print(f"Ending debate due to iteration limit ({state.iteration_count})")
            state.is_active = False
        
        # Just return the state - routing will be handled in conditional edges
        return state
    
    # Add nodes to the graph
    debate_graph.add_node("analyze_article", analyze_article)
    debate_graph.add_node("generate_pro_argument", generate_pro_argument)
    debate_graph.add_node("generate_con_argument", generate_con_argument)
    debate_graph.add_node("fact_check_argument", fact_check_argument)
    debate_graph.add_node("process_verified_argument", process_verified_argument)
    debate_graph.add_node("revise_argument", revise_argument)
    debate_graph.add_node("wait_for_user_input", wait_for_user_input)
    debate_graph.add_node("check_debate_status", check_debate_status)
    
    # Define edges
    
    # Start with article analysis
    debate_graph.set_entry_point("analyze_article")
    
    # After analysis, generate first pro argument
    debate_graph.add_edge("analyze_article", "generate_pro_argument")
    
    # After generating an argument, fact check it
    debate_graph.add_edge("generate_pro_argument", "fact_check_argument")
    debate_graph.add_edge("generate_con_argument", "fact_check_argument")
    debate_graph.add_edge("revise_argument", "fact_check_argument")
    
    # Define conditional edges from fact checking
    def route_after_fact_check(inputs: Any) -> str:
        # Handle different input types
        if isinstance(inputs, dict) and "is_verified" in inputs:
            # Normal case
            return "process_verified_argument" if inputs["is_verified"] else "revise_argument"
        elif isinstance(inputs, DebateState):
            # Edge case: Default to verified for testing
            return "process_verified_argument"
        else:
            # Fallback to revise if uncertain
            return "revise_argument"
    
    debate_graph.add_conditional_edges(
        "fact_check_argument",
        route_after_fact_check,
        {
            "process_verified_argument": "process_verified_argument",
            "revise_argument": "revise_argument"
        }
    )
    
    # After processing a verified argument, wait for user input
    debate_graph.add_edge("process_verified_argument", "wait_for_user_input")
    
    # After user input, check debate status
    debate_graph.add_edge("wait_for_user_input", "check_debate_status")
    
    # Add the conditional edges for routing after status check
    def route_after_status_check(state: Any) -> str:
        # Handle input flexibility
        if not isinstance(state, DebateState):
            # If somehow we get something other than a DebateState
            return "end_debate"  # Default to end in case of errors
        
        # Check if debate should end
        if not state.is_active:
            return "end_debate"
        
        # Route based on whose turn it is
        if state.current_turn == "pro":
            return "pro_turn"
        else:
            return "con_turn"
    
    debate_graph.add_conditional_edges(
        "check_debate_status",
        route_after_status_check,
        {
            "end_debate": END,
            "pro_turn": "generate_pro_argument",
            "con_turn": "generate_con_argument"
        }
    )
    
    # Compile with configuration
    return debate_graph.compile()