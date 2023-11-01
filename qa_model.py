import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )

        model.config.pretraining_tp = 1
        model.eval()

        tokenizer.pad_token = tokenizer.eos_token

        self.pipeline = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            task='text-generation',
            **self.pipeline_kwargs
        )
    
    def get_tokens(self, row):
        system_prefix = "Bên dưới là đoạn văn hướng dẫn để làm theo, bạn cần đọc Question và đoạn Context. Viết câu trả lời dựa vào đoạn Context phù hợp với ngữ cảnh nhất.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_prefix}"
        instruction = f"Công việc của bạn là phân tích Question, đoạn Context. Bạn có thể tùy ý sử dụng ngữ cảnh cơ bản từ các bài Context trên thêm như một trợ giúp tiềm năng cho câu trả lời của mình, ngay cả khi chúng không phải lúc nào cũng thích hợp."
        input_prefix = f"Context: {row['context'][:self.max_content]}\nQuestion: {row['prompt']}\nAnswer: "
        prompt_prefix = system_prefix.format(instruction=instruction, input_prefix=input_prefix)
        return prompt_prefix

    def run_model(self, df):
        inputs = df.apply(self.get_tokens, axis=1).values
        # outputs = []
        # for batch in inputs:
        inputs = '[[INST]] <<SYS>>' + inputs + '<</SYS>>'
        ans = self.pipeline(input)[0]["generated_text"]
        # outputs.append(ans)
        return ans