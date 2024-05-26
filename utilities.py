import datasets
from transformers import AutoTokenizer


def tokenize_fn(examples,tokenizer, max_length):
    texts = []
    for i in range(len(examples["question"])):
        text = examples["question"][i] + examples["answer"][i]
        texts.append(text)
    print(texts)
    
    tokenized_inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    )

    return tokenized_inputs


def tokenize_and_split_dataset(training_config, tokenizer):
    if training_config['datasets']['use_hf']:
        dataset = datasets.load_dataset(training_config['datasets']['path'])

    max_length = training_config['model']['max_length']
    tokenized_dataset = dataset.map(lambda examples : tokenize_fn(examples, tokenizer, max_length), batched=True, drop_last_batch=True)

    tokenized_dataset = tokenized_dataset['train'].add_column('labels', tokenized_dataset['train']['input_ids'])

    split_dataset = tokenized_dataset.train_test_split(test_size=0.1,shuffle=True)

    return split_dataset['train'], split_dataset['test']