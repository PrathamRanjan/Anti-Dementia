from app.api.graph import create_debate_graph
from app.utils.models import DebateState, Article

def test_graph():
    """Test the debate graph with a simple article."""
    
    # Create a test article
    article = Article(
        title="Test Article",
        content="This is a test article to verify if the graph works correctly."
    )
    
    # Create initial state
    initial_state = DebateState(article=article)
    
    # Create debate graph
    print("Creating debate graph...")
    debate_graph = create_debate_graph()
    
    print("Starting graph execution...")
    
    # Execute graph
    try:
        result = debate_graph.invoke(initial_state)
        print("Graph executed successfully!")
        print(f"Summary: {result.summary}")
        return True
    except Exception as e:
        print(f"Error executing graph: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_graph()
    print(f"Test {'passed' if success else 'failed'}")