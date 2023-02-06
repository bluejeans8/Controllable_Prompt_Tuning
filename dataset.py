from datasets import load_dataset, get_dataset_split_names

dataset = load_dataset("lama")
print(dataset['train'][0])
