import argparse
import os

import dill as pickle
import numpy as np
import openai
import tiktoken

from utils.featurize import score_ngram, t_featurize_logprobs
from utils.symbolic import get_words, scalar_functions, train_trigram, vec_functions


MAX_TOKENS = 2047


def load_text(file_path):
    with open(file_path) as handle:
        return handle.read().strip()


def run_ghostbuster(file_path, openai_key=""):
    if openai_key != "":
        openai.api_key = openai_key

    best_features = open("model/features.txt").read().strip().split("\n")
    enc = tiktoken.encoding_for_model("davinci")

    model = pickle.load(open("model/model", "rb"))
    mu = pickle.load(open("model/mu", "rb"))
    sigma = pickle.load(open("model/sigma", "rb"))

    doc = load_text(file_path)
    tokens = enc.encode(doc)[:MAX_TOKENS]
    doc = enc.decode(tokens).strip()
    print(f"Input: {doc}")

    print("Loading Trigram...")
    trigram_model = train_trigram()

    trigram = np.array(score_ngram(doc, trigram_model, enc.encode, n=3, strip_first=False))
    unigram = np.array(score_ngram(doc, trigram_model.base, enc.encode, n=1, strip_first=False))

    response = openai.Completion.create(
        model="ada",
        prompt="<|endoftext|>" + doc,
        max_tokens=0,
        echo=True,
        logprobs=1,
    )
    ada = np.array(
        list(map(lambda x: np.exp(x), response["choices"][0]["logprobs"]["token_logprobs"][1:]))
    )

    response = openai.Completion.create(
        model="davinci",
        prompt="<|endoftext|>" + doc,
        max_tokens=0,
        echo=True,
        logprobs=1,
    )
    davinci = np.array(
        list(map(lambda x: np.exp(x), response["choices"][0]["logprobs"]["token_logprobs"][1:]))
    )

    subwords = response["choices"][0]["logprobs"]["tokens"][1:]
    gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
    for i in range(len(subwords)):
        for key, value in gpt2_map.items():
            subwords[i] = subwords[i].replace(key, value)

    t_features = t_featurize_logprobs(davinci, ada, subwords)

    vector_map = {
        "davinci-logprobs": davinci,
        "ada-logprobs": ada,
        "trigram-logprobs": trigram,
        "unigram-logprobs": unigram,
    }

    exp_features = []
    for exp in best_features:
        exp_tokens = get_words(exp)
        curr = vector_map[exp_tokens[0]]

        for i in range(1, len(exp_tokens)):
            if exp_tokens[i] in vec_functions:
                next_vec = vector_map[exp_tokens[i + 1]]
                curr = vec_functions[exp_tokens[i]](curr, next_vec)
            elif exp_tokens[i] in scalar_functions:
                exp_features.append(scalar_functions[exp_tokens[i]](curr))
                break

    data = (np.array(t_features + exp_features) - mu) / sigma
    preds = model.predict_proba(data.reshape(-1, 1).T)[:, 1]
    print(f"Prediction: {preds}")


def run_roberta(file_path, model_dir):
    import torch
    from transformers import RobertaForSequenceClassification, RobertaTokenizer

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"RoBERTa model directory not found: {model_dir}. Train it first or pass --model_dir."
        )

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=2).to(device)
    model.eval()

    doc = load_text(file_path)
    encoding = tokenizer(
        doc,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    encoding = {key: value.to(device) for key, value in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    probs = torch.softmax(outputs.logits, dim=1)[0][1].item()
    print(f"Prediction: [{probs}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="input.txt")
    parser.add_argument("--mode", choices=["ghostbuster", "roberta"], default="ghostbuster")
    parser.add_argument("--openai_key", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="models/roberta_gpt")
    args = parser.parse_args()

    if args.mode == "ghostbuster":
        run_ghostbuster(args.file, openai_key=args.openai_key)
    elif args.mode == "roberta":
        run_roberta(args.file, args.model_dir)
