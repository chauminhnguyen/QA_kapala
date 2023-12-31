from pipeline import Base_Pipeline
import pandas as pd
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='public_test.csv')
    parser.add_argument('--corpus_path', type=str, default='./MEDICAL/corpus/')
    parser.add_argument('--retrieval_model_name', type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--qa_model_name', type=str, default="TheBloke/vietnamese-llama2-7B-40GB-GPTQ")
    parser.add_argument('--topk', type=int, default=4)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = argparser()
    pipeline = Base_Pipeline(args.retrieval_model_name, args.qa_model_name, args.corpus_path, topk=args.topk)
    data = pd.read_csv(args.data_path)
    # options = data.apply(lambda row: [row[f'option{i}'] for i in range(1, 5)], axis=1)
    data['answer'] = data.apply(lambda row: pipeline.run(row['question'], [row[f'option_{i}'] for i in range(1, 5)]), axis=1)
    data.to_csv('submission.csv', index=False)