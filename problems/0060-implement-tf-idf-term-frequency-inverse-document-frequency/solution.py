import numpy as np

def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.
    
    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """
    def calc_term_frequency_in_document(document: list[str], word: str):
        total = len(document)
        matched = len([corpus_word for corpus_word in document if corpus_word == word])
        return matched / total

    def calc_document_frequency(word: str):
        return len([doc for doc in corpus if calc_term_frequency_in_document(doc, word) > 0])

    term_freq_scores = []
    for query_str in query:
        term_frequencies = [calc_term_frequency_in_document(doc, query_str) for doc in corpus]
        occurences = len([freq for freq in term_frequencies if freq > 0])
        idf_w = np.log((len(corpus) + 1) / (occurences+1)) + 1
        term_frequencies = [freq * idf_w for freq in term_frequencies]
        term_freq_scores.append(term_frequencies)

    return np.asarray(term_freq_scores).T
