from keybert import KeyBERT
import os

# Pre-load the model to avoid loading it on every call.
# It's better to load this once at startup.
try:
    _model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print(f"⚠️ [TermEnhancer] Failed to load model: {e}")
    _model = None

class TermEnhancer:
    def extract_key_terms(self, text: str, top_n: int = 5) -> list[str]:
        """
        Extract semantically relevant key terms/phrases from text using KeyBERT.
        """
        if not _model or not text:
            return []
            
        try:
            # keyphrase_ngram_range=(1, 3): Extract phrases of 1-3 words
            # use_mmr=True: Use Maximal Marginal Relevance to diversify keywords
            # diversity=0.2: Low diversity ensures high relevance to the topic
            keywords = _model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 3), 
                use_mmr=True, 
                diversity=0.2,
                top_n=top_n
            )
            # keybert returns list of tuples (keyword, score)
            return [k[0] for k in keywords]
        except Exception as e:
            print(f"❌ [TermEnhancer] Error: {e}")
            return []

term_enhancer = TermEnhancer()

