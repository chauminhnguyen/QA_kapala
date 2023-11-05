import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from utils import clean_memory


class Llama2_Vi:
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.pipeline_kwargs={
            "temperature": kwargs.get("temperature", 1.0),
            "max_new_tokens": kwargs.get("max_new_tokens", 2500),
            "top_k": kwargs.get("top_k", 1),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        }
        self.max_content = kwargs.get("max_content", 2500)
        self.build_model()

    def build_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="cuda")
        
        # Some gen config knobs
        self.generation_config = GenerationConfig(
            penalty_alpha=0.6,
            do_sample = True,
            top_k=5,
            temperature=0.5,
            repetition_penalty=1.2,
            max_new_tokens=100
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    
    def get_tokens(self, row):
        system_prefix = "Bên dưới là đoạn văn hướng dẫn để làm theo, bạn cần đọc Question và đoạn Context. Viết câu trả lời dựa vào đoạn Context phù hợp với ngữ cảnh nhất.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_prefix}"
        instruction = f"Công việc của bạn là phân tích Question, đoạn Context. Bạn có thể tùy ý sử dụng ngữ cảnh cơ bản từ các bài Context trên thêm như một trợ giúp tiềm năng cho câu trả lời của mình, ngay cả khi chúng không phải lúc nào cũng thích hợp."
        input_prefix = f"Context: {row['context'][:self.max_content]}\nQuestion: {row['prompt']}\nAnswer: "
        prompt_prefix = system_prefix.format(instruction=instruction, input_prefix=input_prefix)

        text_options = []
        question = row['prompt']
        context = row['context']
        for option in row['options']:
            system_prefix = "<START>Bên dưới là đoạn văn hướng dẫn để làm theo, bạn cần đọc và phân tích Câu hỏi, Đoạn văn và Đáp án. Viết Đúng hoặc Sai cho câu Đáp án dựa vào đoạn văn phù hợp với ngữ cảnh nhất.\n\n### Đoạn văn:\n{context}.\n\n### Câu hỏi:\n{question}.\n\n### Đáp án:\n{option}.<SEP>\n\n### {answer}<END>"
            prompt = system_prefix.format(question=question, context=context, option=option.split(')')[-1], answer= "Đúng" if option.split(')')[-1] in row['correct_answers'] else "Sai")
            text_options.append(prompt)
        return text_options

    def run_model(self, df):
        inputs = df.apply(self.get_tokens, axis=1).values
        clean_memory()
        inputs = self.tokenizer(inputs, return_tensors="pt").to('cuda')
        outputs = self.model.generate(**inputs, generation_config=self.generation_config)
        ans = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        clean_memory()
        # outputs.append(ans)
        return ans