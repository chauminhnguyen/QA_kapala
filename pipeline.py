from retrieval import Cross_Encoder_Retrieval
from qa_model import Llama2_Vi
import pandas as pd


class Base_Pipeline:
    def __init__(self, retrieval_model, qa_model, topk=5):
        self.retrieval_model_name = retrieval_model
        self.qa_model_name = qa_model
        self.topk = topk
        self.build_models()

    def build_models(self):
        self.retrieval_model = Cross_Encoder_Retrieval(self.retrieval_model_name)
        self.qa_model = Llama2_Vi(self.qa_model_name)
    
    def run(self, question, options):
        pred_scores_argsort = self.retrieval_model.predict(question)
        pred_scores_argsort = pred_scores_argsort[:self.topk]
        pred_scores_argsort = [int(i) for i in pred_scores_argsort]
        paragraphs = [self.retrieval_model.paragraphs[i] for i in pred_scores_argsort]
        
        lst = [ [question, '\n'.join(paragraphs)] ]
        df = pd.DataFrame(lst, columns=['prompt', 'context'])
        answer = self.qa_model.run_model(df)

        for option in options:
            if option in answer:
                return option  
        