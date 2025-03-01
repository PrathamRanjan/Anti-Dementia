from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_groq import ChatGroq
from app.utils.models import DebateState, Article, Argument

class SupervisorAgent:
    def __init__(self, api_key):
        self.llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.1-8b-instant"
        )
        self.prompt = ChatPromptTemplate.from_template("""
        You are a supervisor agent managing a debate about an article.
        
        Current debate state:
        Article: {article_title}
        Current Turn: {current_turn}
        Pro Arguments: {pro_count}
        Con Arguments: {con_count}
        
        Your task: {task}
        
        Respond with only the necessary instructions or information.
        """)
        
    def initialize_debate(self, article: Article) -> DebateState:
        """Initialize the debate with the given article"""
        return DebateState(article=article)
    
    def manage_turn(self, state: DebateState) -> str:
        """Determine the next action in the debate"""
        task = "Determine the next step in the debate process."
        
        response = self.llm.invoke(
            self.prompt.format(
                article_title=state.article.title,
                current_turn=state.current_turn,
                pro_count=state.pro_count,
                con_count=state.con_count,
                task=task
            )
        )
        
        return response.content
        
    def format_argument(self, argument: Argument) -> str:
        """Format a verified argument for display"""
        position = "PRO" if argument.position == "pro" else "CON"
        return f"ARGUMENT #{argument.number} ({position}):\n\n{argument.content}"
        
    def process_user_input(self, state: DebateState, user_input: str) -> DebateState:
        """Process user input and update the debate state"""
        if user_input.lower() in ["done", "exit"]:
            state.is_active = False
            return state
            
        if user_input.lower() != "continue":
            state.user_inputs.append(user_input)
            
        return state