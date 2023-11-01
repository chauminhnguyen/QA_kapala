from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np


class Cross_Encoder_Retrieval:
    '''
    Retrieval class for retrieving relevant documents for a given query.
    Infer Input: [[query, doc1], [query, doc2], ...]
    Infer Output: [index of doc1, index of doc2, ...]
    '''
    def __init__(self, model_path, corpus_path):
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.build_model()
        self.build_paragraphs()

    def build_model(self):
        self.model = CrossEncoder(self.model_path)

    def build_paragraphs(self):
        self.paragraphs = []

    def predict(self, query, convert_to_numpy=False, show_progress_bar=False):
        model_input = [[query, para] for para in self.paragraphs]
        pred_scores = self.model.predict(model_input, convert_to_numpy=convert_to_numpy, show_progress_bar=show_progress_bar)
        pred_scores_argsort = np.argsort(-pred_scores)
        return pred_scores_argsort