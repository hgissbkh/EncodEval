import os
import random

from datasets import Dataset, DatasetDict
from datasets import load_dataset as _load_dataset
import numpy as np
from tqdm import tqdm


# Wrapper for loading datasets
def load_dataset(*args, **kwargs) -> DatasetDict:
    print(
        f"Loading dataset {args[0]}" + 
        (f", {kwargs['name']}" if "name" in kwargs else "") + 
        (f", {kwargs['split']}" if "split" in kwargs else "") 
    )
    dataset_name = args[0].split("/")[-1]
    
    if "LOCAL_DATASET_DIR" in os.environ:
        print(f"Loading dataset from local storage at {os.environ['LOCAL_DATASET_DIR']}")
        return _load_dataset(f"{os.environ['LOCAL_DATASET_DIR']}/{dataset_name}", *args[1:], **kwargs)
    
    else:
        print("Loading dataset from Hugging Face")
        return _load_dataset(*args, **kwargs)


#========================
# Sequence classification
#========================

def mnli_m() -> DatasetDict:
    dataset = load_dataset("nyu-mll/glue", "mnli")
    dataset["test"] = dataset["validation_matched"]
    dataset_train_val = dataset["train"].train_test_split(train_size=0.95, seed=42)
    dataset["train"], dataset["validation"] = dataset_train_val["train"], dataset_train_val["test"]
    del dataset["validation_matched"], dataset["validation_mismatched"], dataset["test_matched"], dataset["test_mismatched"]
    dataset = dataset.map(
        lambda example: {
            "text": f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}",
        },
        remove_columns=["premise", "hypothesis", "idx"]
    )
    return dataset

def mnli_mm() -> DatasetDict:
    dataset = load_dataset("nyu-mll/glue", "mnli")
    dataset["test"] = dataset["validation_mismatched"]
    dataset_train_val = dataset["train"].train_test_split(train_size=0.95, seed=42)
    dataset["train"], dataset["validation"] = dataset_train_val["train"], dataset_train_val["test"]
    del dataset["validation_matched"], dataset["validation_mismatched"], dataset["test_matched"], dataset["test_mismatched"]
    dataset = dataset.map(
        lambda example: {
            "text": f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}",
        },
        remove_columns=["premise", "hypothesis", "idx"]
    )
    return dataset

def qqp() -> DatasetDict:
    dataset = load_dataset("nyu-mll/glue", "qqp")
    dataset["test"] = dataset["validation"]
    dataset_train_val = dataset["train"].train_test_split(train_size=0.95, seed=42)
    dataset["train"], dataset["validation"] = dataset_train_val["train"], dataset_train_val["test"]
    dataset = dataset.map(
        lambda example: {
            "text": f"Question 1: {example['question1']}\nQuestion 2: {example['question2']}",
        },
        remove_columns=["question1", "question2", "idx"]
    )
    return dataset

def sst2() -> DatasetDict:
    dataset = load_dataset("nyu-mll/glue", "sst2")
    dataset["test"] = dataset["validation"]
    dataset_train_val = dataset["train"].train_test_split(train_size=0.95, seed=42)
    dataset["train"], dataset["validation"] = dataset_train_val["train"], dataset_train_val["test"]
    dataset = dataset.rename_column("sentence", "text")
    dataset = dataset.remove_columns(["idx"])
    return dataset

def stsb() -> DatasetDict:
    dataset = load_dataset("nyu-mll/glue", "stsb")
    dataset["test"] = dataset["validation"]
    dataset_train_val = dataset["train"].train_test_split(train_size=0.95, seed=42)
    dataset["train"], dataset["validation"] = dataset_train_val["train"], dataset_train_val["test"]
    dataset = dataset.map(
        lambda example: {
            "text": f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}",
        },
        remove_columns=["sentence1", "sentence2", "idx"]
    )
    return dataset


#=====================
# Token classification
#=====================

def conll2003_en() -> DatasetDict:
    dataset = load_dataset("hgissbkh/conll2003-en")
    dataset = dataset.rename_column("words", "tokens")
    dataset = dataset.rename_column("ner", "tags")
    return dataset

def ontonotes() -> DatasetDict:
    dataset = load_dataset("hgissbkh/ontonotes5", trust_remote_code=True)
    return dataset

def uner_en() -> DatasetDict:
    dataset = load_dataset("hgissbkh/uner_en")
    dataset = dataset.remove_columns(["idx", "text", "annotator"])
    dataset = dataset.rename_column("ner_tags", "tags")
    return dataset


#===================
# Question answering
#===================

def squad() -> DatasetDict:
    dataset = load_dataset("rajpurkar/squad")
    dataset = dataset.remove_columns(["id", "title"])
    dataset["test"] = dataset["validation"]
    dataset_train_val = dataset["train"].train_test_split(train_size=0.95, seed=42)
    dataset["train"], dataset["validation"] = dataset_train_val["train"], dataset_train_val["test"]
    return dataset

def squad_v2() -> DatasetDict:
    dataset = load_dataset("rajpurkar/squad_v2")
    dataset = dataset.remove_columns(["id", "title"])
    dataset["test"] = dataset["validation"]
    dataset_train_val = dataset["train"].train_test_split(train_size=0.95, seed=42)
    dataset["train"], dataset["validation"] = dataset_train_val["train"], dataset_train_val["test"]
    return dataset

def qed() -> DatasetDict:
    dataset = load_dataset("hgissbkh/qed")
    dataset["test"] = dataset["validation"]
    dataset_train_val = dataset["train"].train_test_split(train_size=0.95, seed=42)
    dataset["train"], dataset["validation"] = dataset_train_val["train"], dataset_train_val["test"]
    return dataset

def record() -> DatasetDict:
    dataset = load_dataset("hgissbkh/record")
    dataset["test"] = dataset["validation"]
    dataset_train_val = dataset["train"].train_test_split(train_size=0.95, seed=42)
    dataset["train"], dataset["validation"] = dataset_train_val["train"], dataset_train_val["test"]
    return dataset


#==========
# Retrieval
#==========

# Training set
def msmarco_pairs() -> DatasetDict:
    dataset = load_dataset("hgissbkh/msmarco-pairs")
    dataset = dataset.rename_column("query", "anchor")
    dataset = dataset.map(
        lambda x: {
            "anchor": f"Query: {x['anchor']}", 
            "positive": f"Document: {x['positive']}", 
        }
    )
    return dataset

# Evaluation datasets
def msmarco() -> DatasetDict:
    queries = load_dataset("BeIR/msmarco", "queries", split="queries", trust_remote_code=True)
    corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus", trust_remote_code=True)
    qrels = {
        "validation": load_dataset("BeIR/msmarco-qrels", split="validation", trust_remote_code=True),
        "test": load_dataset("BeIR/msmarco-qrels", split="test", trust_remote_code=True),
    }
    queries = {
        str(example["_id"]): f"Query: {example['text']}" 
        for example in tqdm(queries, desc="Processing queries")
    }
    corpus = {
        str(example["_id"]): f"Document: {example['text']}" 
        for example in tqdm(corpus, desc="Processing corpus")
    }
    queries_sub, corpus_sub, qrels_dict = {}, {}, {}
    for split in qrels:
        qrels_dict[f"qrels_{split}"] = {}
        for example in tqdm(qrels[split], desc=f"Processing qrels ({split})"):
            query_id, corpus_id = str(example["query-id"]), str(example["corpus-id"])
            queries_sub[query_id], corpus_sub[corpus_id] = queries[query_id], corpus[corpus_id]
            if example["score"] >= 1:
                if query_id in qrels_dict[f"qrels_{split}"]:
                    qrels_dict[f"qrels_{split}"][query_id].append(corpus_id)
                else:
                    qrels_dict[f"qrels_{split}"][query_id] = [corpus_id]
    del queries, corpus, qrels
    dataset = {"queries": queries_sub, "corpus": corpus_sub, **qrels_dict}
    return dataset

def nq() -> DatasetDict:
    queries = load_dataset("BeIR/nq", "queries", split="queries", trust_remote_code=True)
    corpus = load_dataset("BeIR/nq", "corpus", split="corpus", trust_remote_code=True)
    qrels = load_dataset("BeIR/nq-qrels", split="test", trust_remote_code=True).to_pandas()
    query_ids = sorted(list(set(qrels["query-id"])))
    query_ids_val = random.sample(query_ids, round(0.5 * len(query_ids)))
    query_ids_test = list(set(query_ids) - set(query_ids_val))
    qrels = {
        "validation": Dataset.from_pandas(qrels[np.isin(qrels["query-id"], query_ids_val)]),
        "test": Dataset.from_pandas(qrels[np.isin(qrels["query-id"], query_ids_test)]),
    }
    queries = {
        str(example["_id"]): f"Query: {example['text']}" 
        for example in tqdm(queries, desc="Processing queries")
    }
    corpus = {
        str(example["_id"]): f"Document: {example['title']}. {example['text']}" 
        for example in tqdm(corpus, desc="Processing corpus")
    }
    queries_sub, corpus_sub, qrels_dict = {}, {}, {}
    for split in qrels:
        qrels_dict[f"qrels_{split}"] = {}
        for example in tqdm(qrels[split], desc=f"Processing qrels ({split})"):
            query_id, corpus_id = str(example["query-id"]), str(example["corpus-id"])
            queries_sub[query_id], corpus_sub[corpus_id] = queries[query_id], corpus[corpus_id]
            if example["score"] >= 1:
                if query_id in qrels_dict[f"qrels_{split}"]:
                    qrels_dict[f"qrels_{split}"][query_id].append(corpus_id)
                else:
                    qrels_dict[f"qrels_{split}"][query_id] = [corpus_id]
    del queries, corpus, qrels
    dataset = {"queries": queries_sub, "corpus": corpus_sub, **qrels_dict}
    return dataset

def mldr_en() -> DatasetDict:    
    qrels = {
        "validation": load_dataset("hgissbkh/mldr-en", "default", split="dev", trust_remote_code=True),
        "test": load_dataset("hgissbkh/mldr-en", "default", split="test", trust_remote_code=True),
    }
    corpus = load_dataset("hgissbkh/mldr-en", "corpus", split="corpus")
    queries = {}
    for split in qrels:
        for example in tqdm(qrels[split], desc=f"Processing queries ({split})"):
            queries[example["query_id"]] = f"Query: {example['query']}"
    corpus = {example["docid"]: f"Document: {example['text']}" for example in tqdm(corpus, desc="Processing corpus")}
    queries_sub, corpus_sub, qrels_dict = {}, {}, {}
    for split in qrels:
        qrels_dict[f"qrels_{split}"] = {}
        for example in tqdm(qrels[split], desc=f"Processing qrels ({split})"):
            query_id = example["query_id"]
            corpus_ids = [pos["docid"] for pos in example["positive_passages"]]
            for corpus_id in corpus_ids:        
                queries_sub[query_id], corpus_sub[corpus_id] = queries[query_id], corpus[corpus_id]
                if query_id in qrels_dict[f"qrels_{split}"]:
                    qrels_dict[f"qrels_{split}"][query_id].append(corpus_id)
                else:
                    qrels_dict[f"qrels_{split}"][query_id] = [corpus_id]
    del queries, corpus, qrels
    dataset = {"queries": queries_sub, "corpus": corpus_sub, **qrels_dict}
    return dataset
