from langchain_community.tools import DuckDuckGoSearchRun
from app.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class TermExplainer:
    def __init__(self):
        self.search = DuckDuckGoSearchRun()
        self.llm = get_llm(temperature=0.3)

    def explain(self, term: str) -> str:
        """
        Lightweight search for term definition.
        """
        try:
            # Limit search to encyclopedias
            query = f'"{term}" definition site:wikipedia.org OR site:baike.baidu.com OR site:wiki.mbalib.com OR site:zhihu.com'
            result = self.search.invoke(query)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a concise dictionary assistant. Based on the context, explain the term '{term}' in Chinese. Keep it under 100 words. Cite the source if possible."),
                ("user", "Context: {context}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            summary = chain.invoke({"term": term, "context": result})
            return summary
        except Exception as e:
            return f"无法获取解释: {e}"

