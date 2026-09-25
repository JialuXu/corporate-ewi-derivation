"""Knowledge submodule — local industry-knowledge RAG via TF-IDF.

Corpus is markdown under `corpus/`, with subdirectories that classify each
document's scope (industries / cross_cutting / pitfalls). The retriever
uses scikit-learn's TfidfVectorizer with character n-grams so Chinese
queries don't depend on word segmentation.

Used by `analysis.suspicion` to anchor LLM judgements in business knowledge.
`knowledge` itself does NOT call LLMs.
"""
