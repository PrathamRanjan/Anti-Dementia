from fastapi import FastAPI, HTTPException, Depends, Body
from pydantic import BaseModel
from typing import List, Optional
from app.utils.models import Article, DebateState, Argument
from app.api.graph import create_debate_graph, supervisor_agent
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI(title="Article Debate System")

# Create the debate graph
debate_graph = create_debate_graph()

# Configuration for graph execution
graph_config = {
    "recursion_limit": 50  # Higher limit for complex debates
}

class DebateRequest(BaseModel):
    article_title: str
    article_content: str
    article_source: Optional[str] = None

class UserInputRequest(BaseModel):
    debate_id: str
    user_input: str

class ArgumentResponse(BaseModel):
    content: str
    position: str
    number: int
    
class DebateResponse(BaseModel):
    debate_id: str
    article_title: str
    summary: Optional[str] = None
    arguments: List[ArgumentResponse] = []
    current_turn: str
    waiting_for_user: bool = True
    is_active: bool = True
    iteration_count: int = 0

# In-memory store for debate states (in a production app, use a database)
debate_sessions = {}

@app.post("/debates", response_model=DebateResponse)
async def create_debate(request: DebateRequest):
    """Start a new debate based on an article"""
    
    # Create article object
    article = Article(
        title=request.article_title,
        content=request.article_content,
        source=request.article_source
    )
    
    # Initialize debate state
    initial_state = supervisor_agent.initialize_debate(article)
    
    try:
        # Generate a simple ID (use UUID in production)
        debate_id = f"debate_{len(debate_sessions) + 1}"
        
        # Store the initial state and graph
        debate_sessions[debate_id] = {
            "state": initial_state,
            "graph": debate_graph
        }
        
        # Start the graph execution with increased recursion limit
        next_state = debate_graph.invoke(initial_state, config=graph_config)
        
        # Update stored state
        debate_sessions[debate_id]["state"] = next_state
        
        # Convert to response format
        response = DebateResponse(
            debate_id=debate_id,
            article_title=article.title,
            summary=next_state.summary,
            arguments=[],
            current_turn=next_state.current_turn,
            waiting_for_user=True,
            is_active=next_state.is_active,
            iteration_count=next_state.iteration_count
        )
        
        return response
    except Exception as e:
        print(f"Error creating debate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating debate: {str(e)}")

@app.post("/debates/{debate_id}/input", response_model=DebateResponse)
async def add_user_input(debate_id: str, input_request: UserInputRequest):
    """Add user input to an ongoing debate"""
    
    if debate_id not in debate_sessions:
        raise HTTPException(status_code=404, detail="Debate session not found")
    
    try:
        # Get current state and graph
        current_state = debate_sessions[debate_id]["state"]
        graph = debate_sessions[debate_id]["graph"]
        
        # Process user input
        updated_state = supervisor_agent.process_user_input(
            current_state, 
            input_request.user_input
        )
        
        # Continue the graph execution if debate is still active
        if updated_state.is_active:
            # Use increased recursion limit when continuing
            next_state = graph.continue_from(updated_state, config=graph_config)
            debate_sessions[debate_id]["state"] = next_state
        else:
            debate_sessions[debate_id]["state"] = updated_state
            next_state = updated_state
        
        # Format arguments for response
        formatted_arguments = [
            ArgumentResponse(
                content=arg.content,
                position=arg.position,
                number=arg.number
            )
            for arg in next_state.arguments
        ]
        
        # Create response
        response = DebateResponse(
            debate_id=debate_id,
            article_title=next_state.article.title,
            summary=next_state.summary,
            arguments=formatted_arguments,
            current_turn=next_state.current_turn,
            waiting_for_user=True,  # Always true when returning to frontend
            is_active=next_state.is_active,
            iteration_count=next_state.iteration_count
        )
        
        return response
    except Exception as e:
        print(f"Error processing input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")

@app.get("/debates/{debate_id}", response_model=DebateResponse)
async def get_debate_status(debate_id: str):
    """Get the current status of a debate"""
    
    if debate_id not in debate_sessions:
        raise HTTPException(status_code=404, detail="Debate session not found")
    
    try:
        current_state = debate_sessions[debate_id]["state"]
        
        # Format arguments for response
        formatted_arguments = [
            ArgumentResponse(
                content=arg.content,
                position=arg.position,
                number=arg.number
            )
            for arg in current_state.arguments if arg is not None
        ]
        
        # Create response
        response = DebateResponse(
            debate_id=debate_id,
            article_title=current_state.article.title,
            summary=current_state.summary,
            arguments=formatted_arguments,
            current_turn=current_state.current_turn,
            waiting_for_user=True,
            is_active=current_state.is_active,
            iteration_count=current_state.iteration_count
        )
        
        return response
    except Exception as e:
        print(f"Error getting debate status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting debate status: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)