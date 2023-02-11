from transformers import pipeline
from datasets import load_dataset
import numpy as np
import torch
import json
# import evaluate

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("json", data_files = "/home/tjrals/jinseok/js_p-tuning/test_data/test_original_relations.json")



# def tokenization(data):
#     return tokenizer(data["masked_sentence"])

# tokenized_datasets = dataset.map(tokenization, batched=True)

# eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))

# metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)




# print(dataset['train'][0])


unmasker = pipeline('fill-mask', model='bert-base-cased', device=0)

total = 0
correct = 0
for data in dataset['train']:
    total+=1
    masked_sentence = data['masked_sentence']
    predicted_str = unmasker(masked_sentence)[0]['token_str']
    answer_str = data['obj_label']
    if predicted_str == answer_str:
        correct+=1
    if total%1000==0:
        print(total, correct/total)

print(f"total: {total}, P@1: {correct/total}")