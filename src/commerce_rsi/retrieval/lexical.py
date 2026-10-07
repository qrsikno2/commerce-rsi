"""A dependency-light inverted-index BM25 baseline."""

import re
from collections import Counter, defaultdict

import numpy as np


def tokenize(text):
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


class BM25:
    def __init__(self, texts, k1=1.2, b=0.75):
        self.size = len(texts)
        self.k1 = k1
        self.b = b
        postings = defaultdict(list)
        lengths = []
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths.append(sum(counts.values()))
            for token, count in counts.items():
                postings[token].append((i, count))
        lengths = np.asarray(lengths, dtype=np.float64)
        average_length = max(float(lengths.mean()) if len(lengths) else 0.0, 1.0)
        self.normalizer = k1 * (1 - b + b * lengths / average_length)
        self.postings = {
            token: (np.array([i for i, _ in pairs], dtype=np.int64),
                    np.array([count for _, count in pairs], dtype=np.float64))
            for token, pairs in postings.items()
        }

    def score(self, query):
        scores = np.zeros(self.size, dtype=np.float64)
        for token, query_tf in Counter(tokenize(query)).items():
            if token not in self.postings:
                continue
            positions, term_frequency = self.postings[token]
            document_frequency = len(positions)
            idf = np.log1p((self.size - document_frequency + 0.5) / (document_frequency + 0.5))
            scores[positions] += query_tf * idf * (term_frequency * (self.k1 + 1)) / (
                term_frequency + self.normalizer[positions])
        return scores
