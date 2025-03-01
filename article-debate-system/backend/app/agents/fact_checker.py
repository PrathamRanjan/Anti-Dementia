from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import requests
from app.utils.models import Argument
from typing import Dict, Any, Tuple

class FactCheckerAgent:
    def __init__(self, groq_api_key, google_fact_check_api_key):
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.1-8b-instant"
        )
        self.google_api_key = google_fact_check_api_key
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are a fact checker agent evaluating an argument in a debate.
        
        Argument: {argument}
        
        Google Fact Check API results: {api_results}
        
        Your task is to:
        1. Identify any factual claims in the argument
        2. Determine if these claims are supported by reliable evidence
        3. Check if any claims contradict established facts
        
        Provide a detailed assessment of the argument's factual accuracy.
        Clearly state whether the argument PASSES or FAILS fact checking.
        If it FAILS, explain what needs to be corrected.
        """)
        
    def check_facts_with_api(self, query: str) -> Dict[str, Any]:
        """Query the Google Fact Check API"""
        base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        
        params = {
            "key": self.google_api_key,
            "query": query
        }
        
        try:
            response = requests.get(base_url, params=params)
            return response.json()
        except Exception as e:
            # Handle API errors gracefully
            return {"error": str(e), "claims": []}
        
    def verify_argument(self, argument: Argument) -> Tuple[bool, str, Argument]:
        """Verify an argument and return (is_verified, feedback, updated_argument)"""
        
        # Get fact check results from Google API
        api_results = self.check_facts_with_api(argument.content)
        
        # Use LLM to evaluate factual accuracy
        response = self.llm.invoke(
            self.prompt.format(
                argument=argument.content,
                api_results=str(api_results)
            )
        )
        
        feedback = response.content
        is_verified = "PASSES" in feedback
        
        # Update argument verification status
        argument.verified = is_verified
        
        return is_verified, feedback, argument