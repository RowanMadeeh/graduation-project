from summarizer import Summarizer
from spacy.lang.ar import Arabic
import pickle
import nltk
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from rouge import Rouge


class ExtractiveService:
    def __init__(self):
        self.dic = {}
        self.chunk = 0
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        self.arabic_stopwords = set(stopwords.words('arabic'))

    def normalize_arabic(self, text):
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ؤ", "و", text)
        text = re.sub(r"ئ", "ي", text)
        text = re.sub(r"ة", "ه", text)
        # text = re.sub(r"[^\w\s]", "", text)
        text = re.sub("\n", " ", text)
        return text

    def preprocess_sentences(self, text):
        # Split by Arabic sentence-ending punctuation
        raw_sentences = re.split(r'(?<=[.!؟])\s+', text)
        cleaned = []
        for s in raw_sentences:
            s = s.strip()
            if s:
                norm = self.normalize_arabic(s)
                cleaned.append(norm)
        return cleaned

    def arabic_tokenizer(self, text):
        words = text.split()
        return [w for w in words if w not in self.arabic_stopwords]

    def summarize_arabic_text(self, text):
        sentences = self.preprocess_sentences(text)
        # Convert each sentence into TF-IDF vector
        vectorizer = TfidfVectorizer(tokenizer=self.arabic_tokenizer)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Calculate the similarity between each pair of sentences
        # and represent each sentence as a node and the edge between
        # each two nodes represents the cosine similarity between them.
        sim_matrix = cosine_similarity(tfidf_matrix)
        graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(graph)

        # Top-ranked sentence indices
        scored = [(i, scores[i]) for i in range(len(sentences))]
        top_indices = sorted(scored, key=lambda x: x[1], reverse=True)[:int(len(scored) * 0.5) + 1]

        # Sort by original order
        top_indices_sorted = sorted(top_indices, key=lambda x: x[0])
        summary = [sentences[i] for i, _ in top_indices_sorted]

        return "\n".join(summary)

    def getSummary(self, text: str):
        summary = self.summarize_arabic_text(text)
        return summary
