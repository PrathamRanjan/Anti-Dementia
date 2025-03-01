from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from app.utils.models import Article

class ReaderAgent:
    def __init__(self, api_key):
        self.llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.1-8b-instant"
        )
        self.prompt = ChatPromptTemplate.from_template("""
        You are a reader agent tasked with analyzing an article.
        
        Article Title: {title}
        Article Content: {content}
        
        Please provide:
        1. A concise summary of the article
        2. The main position/stance of the article
        3. Key claims and evidence presented
        4. Potential counterarguments
        
        Be objective and thorough in your analysis.
        """)
        
    def analyze_article(self, article: Article) -> str:
        """Analyze the article and return a structured summary"""
        response = self.llm.invoke(
            self.prompt.format(
                title=article.title,
                content=article.content
            )
        )
        
        return response.content