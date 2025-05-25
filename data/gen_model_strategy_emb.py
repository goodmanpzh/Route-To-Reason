import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

device = ''
encoder_model_path = '' # your path to the encoder model


model_descriptions = {
    "Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct is a lightweight 3B parameter model with fast inference and low resource usage. It is suitable for simple tasks such as basic question answering and short-form text generation, but is limited in handling complex reasoning or multi-step tasks.",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct is a mid-small 7B parameter model that balances speed and performance. It is capable of multi-turn dialogue, basic code and math tasks, and offers improved language understanding over smaller models, while maintaining efficient inference.",
    "Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct is a mid-sized 14B parameter model that excels at complex reasoning, document summarization, and structured mid-length text generation. It demonstrates strong performance on tasks requiring deeper understanding and context retention.",
    "DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-7B is a 7B distilled model with slightly slower but stable inference compared to other models of similar size. It is well-suited for medium-length answers that require deep reasoning, and often generates more detailed and comprehensive responses.",
    "DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-R1-14B is a 14B distilled model with slower inference but strong logical and mathematical capabilities. It is ideal for mathematical proofs and tasks requiring rigorous step-by-step reasoning, offering robust performance in logic-intensive scenarios.",
    "QwQ-32B-AWQ": "QwQ-32B is a large 32B quantized model with slow inference. It excels at complex logic, coding, and multi-step reasoning tasks, though it may produce verbose outputs. Its large capacity enables handling of challenging prompts and long-context tasks."
}

# model_descriptions = {
#     "Qwen3-4B_thinking": "Qwen3-4B in 'thinking' mode generates longer reasoning chains and detailed thought processes. While it has slower inference and higher resource usage, it excels at solving complex logic and reasoning problems, making it suitable for tasks that require step-by-step explanations or in-depth analysis.",
#     "Qwen3-4B_non_thinking": "Qwen3-4B in 'non-thinking' mode is optimized for short, direct answers. It provides fast inference with low resource cost, but is limited in deep reasoning or step-by-step explanations, making it best for straightforward queries or when efficiency is prioritized."
# }



strategy_descriptions = {
    "direct": "Direct prompting retains the original question content without adding any additional prompt information.",
    "cot": "Chain-of-Thought (CoT) prompting guides the model to articulate a step-by-step reasoning process before providing the final answer. This results in longer responses and slightly slower reasoning speed, typically generating the longest answers, but it performs best on complex problems such as mathematical reasoning.",
    "cod": "Chain-of-Draft (CoD) prompts the model to generate only intermediate drafts with explicit constraints on output length, encouraging concise reasoning. These drafts represent the model's thinking process, often containing important calculation steps and key reasoning information. It simplifies the intermediate steps of the reasoning chain while retaining good performance, resulting in shorter answers.",
    "pal": "Program-Aided Language (PAL) PAL transforms the reasoning process into executable code. This approach leverages the determinism of programming languages to ensure logical consistency and high accuracy, making it particularly effective for mathematical, symbolic, or algorithmic tasks. PAL relies on a suitable code execution environment and consistently produces results with high reliability and stable, moderately sized outputs. However, it may not well suited for commonsense reasoning tasks."
}

def generate_model_strategy_embeddings(encoder_model_path, model_strategy_emb_path):
        print("Generating model/strategy embeddings...")
        encoder = SentenceTransformer(encoder_model_path, local_files_only=True, device=device)
        model_emb = {k: encoder.encode(v) for k, v in model_descriptions.items()}
        strategy_emb = {k: encoder.encode(v) for k, v in strategy_descriptions.items()}
        torch.save({"model": model_emb, "strategy": strategy_emb}, model_strategy_emb_path)
        print("Embeddings saved successfully!")
        return {"model": model_emb, "strategy": strategy_emb}

model_strategy_emb_path = "./model_strategy_embeddings.pth"
model_strategy_embeddings = generate_model_strategy_embeddings(encoder_model_path, model_strategy_emb_path)