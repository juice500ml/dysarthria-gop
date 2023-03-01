import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import entropy, kendalltau
from tqdm import tqdm

from dataset import PhoneRecognitionDataset, reduce_vocab
from model import Wav2Vec2Recognizer, Wav2Vec2ConvRecognizer
from train_phone_recognizer import _get_collator


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=Path)
    parser.add_argument("--commonphone_csv", type=Path)
    parser.add_argument("--gpu", default=None, type=int, help="Default to CPU. Input GPU index (integer) to use GPU.")
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--ignore_nonexisting_vocab", type=bool, default=True)
    return parser.parse_args()


def _get_model_vocab(model_path):
    args = pickle.load(open(model_path / "arguments.pkl", "rb"))
    index_to_vocab = pickle.load(open(model_path / "index_to_vocab.pkl", "rb"))
    model = {True: Wav2Vec2ConvRecognizer, False: Wav2Vec2Recognizer}[args.use_conv_only](args.model, len(index_to_vocab))
    model.load_state_dict(torch.load(model_path / "best.pt"))
    return model, args.model, index_to_vocab, args.reduce_vocab


def _get_data(df, collator):
    ds = PhoneRecognitionDataset(df)

    severity = []
    for audio in ds.audios:
        label = df[df.audio == audio].label.unique()
        assert len(label) == 1
        severity.append(label[0])

    return severity, torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        collate_fn=collator,
    )


def _get_prior(df, vocab_to_index):
    prior = np.zeros(len(vocab_to_index))
    for vocab, prob in df[df.split == "train"].phone.value_counts(normalize=True).to_dict().items():
        prior[vocab_to_index[vocab]] = prob
    return prior


def _infer(model, test_ds, device):
    model.eval()
    logits_acc, labels_acc = [], []
    for x, y in tqdm(test_ds):
        logits, _ = model(x.to(device))
        logits_acc.append(logits.detach().cpu()[0])
        labels_acc.append(y[0])
    return logits_acc, labels_acc


def _remove_nonexisting_vocab(logits_acc, labels_acc, prior, index_to_vocab, vocab_to_index, existing_vocabs):
    indices = np.array([vocab_to_index[v] for v in existing_vocabs])
    mask_acc = [np.isin(labels, indices) for labels in labels_acc]
    new_vocab_to_index = {v: i for i, v in enumerate(existing_vocabs)}

    prior = np.array([prior[i] for i in range(len(prior)) if i in indices])
    prior /= prior.sum()
    logits_acc = [logits[mask][:, indices] for mask, logits in zip(mask_acc, logits_acc)]
    labels_acc = [labels[mask].apply_(lambda i: new_vocab_to_index[index_to_vocab[i]]) for mask, labels in zip(mask_acc, labels_acc)]

    return logits_acc, labels_acc, prior


def _get_scores(logits_acc, labels_acc, gop_scorer, ignore_label):
    scores = []
    for logits, labels in zip(logits_acc, labels_acc):
        mask = labels != ignore_label
        scores.append(gop_scorer(logits[mask], labels[mask]))
    return np.array(scores)


def _phonewise_loop(labels: torch.LongTensor):
    uniq, loc = torch.unique_consecutive(
        labels, return_inverse=True)
    for i, v in enumerate(uniq):
        mask = (loc == i).numpy()
        yield v.item(), mask


def gmm_gop_scorer(logits: torch.FloatTensor, labels: torch.LongTensor) -> float:
    scores = []
    preds = logits.softmax(-1)
    for vocab, mask in _phonewise_loop(labels):
        scores.append(preds[mask, vocab].log().mean().item())
    return np.mean(scores)


def nn_gop_scorer(logits, labels):
    scores = []
    preds = logits.softmax(-1)
    for vocab, mask in _phonewise_loop(labels):
        avg_preds = preds[mask].mean(0)
        scores.append(avg_preds[vocab].item() - avg_preds.max().item())
    return np.mean(scores)


def logit_margin_gop_scorer(logits, labels):
    scores = []
    for vocab, mask in _phonewise_loop(labels):
        avg_preds = logits[mask].mean(0)
        mask = torch.arange(avg_preds.shape[0]) != vocab
        others_avg_preds = avg_preds[mask]
        scores.append(avg_preds[vocab].item() - others_avg_preds.max().item())
    return np.mean(scores)


def margin_gop_scorer(logits, labels):
    return logit_margin_gop_scorer(logits.softmax(-1), labels)


def logit_gop_scorer(logits, labels):
    scores = []
    for vocab, mask in _phonewise_loop(labels):
        scores.append(logits[mask, vocab].mean().item())
    return np.mean(scores)


def mean_prob_gop_scorer(logits, labels):
    return logit_gop_scorer(logits.softmax(-1), labels)


def entropy_gop_scorer(logits, labels):
    scores = []
    preds = logits.softmax(-1)
    for vocab, mask in _phonewise_loop(labels):
        scores.append(preds[mask, vocab].mean().item())
    return entropy(scores)


def normalizer(scorer, prior):
    def _norm_scorer(logits, labels):
        return scorer(logits - np.exp(prior), labels)
    return _norm_scorer


def scaler(scorer, temperature):
    def _scale_scorer(logits, labels):
        return scorer(logits / temperature, labels)
    return _scale_scorer


if __name__ == "__main__":
    args = _get_args()
    print(args)

    device = torch.device("cpu" if args.gpu is None else args.gpu)
    model, model_name, index_to_vocab, do_reduce_vocab = _get_model_vocab(args.model_path)
    vocab_to_index = {v: i for i, v in index_to_vocab.items()}

    df = pd.read_csv(args.dataset_csv, compression="gzip")
    if do_reduce_vocab:
        df = reduce_vocab(df)

    collator = _get_collator(model_name, vocab_to_index, model.get_feat_length)
    severity, test_dl = _get_data(df, collator)

    cp_df = pd.read_csv(args.commonphone_csv, compression="gzip")
    if do_reduce_vocab:
        cp_df = reduce_vocab(cp_df)
    prior = _get_prior(cp_df, vocab_to_index)
    # temperature = _get_temperature(args.commonphone_csv)

    logits_acc, labels_acc = _infer(model.to(device), test_dl, device)
    if args.ignore_nonexisting_vocab:
        logits_acc, labels_acc, prior = _remove_nonexisting_vocab(logits_acc, labels_acc, prior, index_to_vocab, vocab_to_index, df.phone.unique().tolist())

    scorers = {
        "GMM-GoP": gmm_gop_scorer,
        "NN-GoP": nn_gop_scorer,
        "CALL-GoP": normalizer(margin_gop_scorer, prior),
        "DNN-GoP": normalizer(mean_prob_gop_scorer, prior),
        "Entropy-GoP": entropy_gop_scorer,
        "NormEntropy-GoP": normalizer(entropy_gop_scorer, prior),
        # "ScaleEntropy-GoP": scaler(entropy_gop_scorer, temperature),
        "Margin-GoP": margin_gop_scorer,
        "LogitMargin-GoP": logit_margin_gop_scorer,
        "Logit-GoP": logit_gop_scorer,
        "NormLogit-GoP": normalizer(logit_gop_scorer, prior),
        "NormLogitMargin-GoP": normalizer(logit_margin_gop_scorer, prior),
        "Prob-GoP": mean_prob_gop_scorer,
        "NormProb-GoP": normalizer(mean_prob_gop_scorer, prior),
        # "ScaleProb-GoP": scaler(mean_prob_gop_scorer, temperature),
    }
    ood_scorers = {}

    for name, scorer in scorers.items():
        scores = _get_scores(logits_acc, labels_acc, scorer, ignore_label=vocab_to_index["(...)"])
        print(f"{name}: {kendalltau(scores, severity).statistic:.4f}")
