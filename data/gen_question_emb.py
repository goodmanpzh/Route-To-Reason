from sentence_transformers import SentenceTransformer
import json
import torch
import pandas as pd
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "" # your gpu id
jsonl_path = "./all_questions_merged.jsonl"
questions = []
ids = []

with open(jsonl_path, "r") as f:
    for line in f:
        data = json.loads(line)
        questions.append(data["question"])
        ids.append(data["id"])

model = SentenceTransformer('', local_files_only=True) # your path to the encoder model
embeddings = model.encode(questions, show_progress_bar=True, batch_size=32)
embedding_tensor = torch.tensor(embeddings)
pth_path = "./all_question_embeddings.pth"
torch.save(embedding_tensor, pth_path)