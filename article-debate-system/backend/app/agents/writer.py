from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from app.utils.models import Article, Argument
from typing import List, Literal

class WriterAgent:
    def __init__(self, api_key, position: Literal["pro", "con"]):
        self.llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.1-8b-instant"
        )
        self.position = position
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are a {position} writer agent in a debate about an article.
        
        Article Summary: {article_summary}
        
        Previous Arguments:
        {previous_arguments}
        
        User Input: {user_input}
        
        Your task is to write a compelling argument {stance} the article's position.
        Focus on facts and logical reasoning. Make specific claims that can be fact-checked.
        
        Write a concise, well-structured argument of 3-5 paragraphs.
        """)
        
    def create_argument(self, 
                       article_summary: str,
                       previous_arguments: List[Argument],
                       user_input: str = "",
                       argument_number: int = 1) -> Argument:
        """Create a new argument based on the debate context"""
        
        # Format previous arguments for context
        prev_args_text = "\n\n".join([
            f"Argument #{arg.number} ({'PRO' if arg.position == 'pro' else 'CON'}): {arg.content}"
            for arg in previous_arguments[-3:] if previous_arguments  # Only include last 3 for context
        ]) if previous_arguments else "No previous arguments."
        
        stance = "supporting" if self.position == "pro" else "opposing"
        
        response = self.llm.invoke(
            self.prompt.format(
                position=self.position,
                article_summary=article_summary,
                previous_arguments=prev_args_text,
                user_input=user_input or "No user input provided.",
                stance=stance
            )
        )
        
        return Argument(
            content=response.content,
            position=self.position,
            number=argument_number,
            verified=False  # Will be verified by fact checker
        )
        
    def revise_argument(self, argument: Argument, fact_check_feedback: str) -> Argument:
        """Revise an argument based on fact checking feedback"""
        revision_prompt = ChatPromptTemplate.from_template("""
        You need to revise your argument based on fact-checking feedback.
        
        Your original argument:
        {original_argument}
        
        Fact-checking feedback:
        {feedback}
        
        Please revise your argument to address these issues while maintaining your {position} position.
        Focus on accuracy while keeping your argument persuasive.
        """)
        
        response = self.llm.invoke(
            revision_prompt.format(
                original_argument=argument.content,
                feedback=fact_check_feedback,
                position=self.position
            )
        )
        
        # Return revised argument with same metadata
        argument.content = response.content
        return argument