import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from dataset import PhoneRecognitionDataset, reduce_vocab
from model import Wav2Vec2Recognizer, Wav2Vec2ConvRecognizer
from train_phone_recognizer import _get_collator


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=Path)
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--gpu", default=None, type=int, help="Default to CPU. Input GPU index (integer) to use GPU.")
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def _get_model_vocab(model_path):
    args = pickle.load(open(model_path / "arguments.pkl", "rb"))
    index_to_vocab = pickle.load(open(model_path / "index_to_vocab.pkl", "rb"))
    model = {True: Wav2Vec2ConvRecognizer, False: Wav2Vec2Recognizer}[args.use_conv_only](args.model, len(index_to_vocab))
    model.load_state_dict(torch.load(model_path / "best.pt"))
    return model, args.model, index_to_vocab, args.reduce_vocab


def _get_data(df, batch_size, split):
    return torch.utils.data.DataLoader(
        PhoneRecognitionDataset(df[df.split == split]),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=16,
        collate_fn=collator,
    )


def _infer(model, test_ds, device):
    model.eval()
    logits_acc, labels_acc = [], []
    for x, labels in tqdm(test_ds):
        logits, _ = model(x.to(device))

        logits = logits.detach().cpu().flatten(0, 1)
        labels = labels.flatten(0, 1)

        logits = logits[labels >= 0]
        labels = labels[labels >= 0]

        logits_acc.append(logits)
        labels_acc.append(labels)

    return torch.cat(logits_acc), torch.cat(labels_acc)


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
    dl = _get_data(df, args.batch_size, args.split)

    logits, labels = _infer(model.to(device), dl, device)

    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=400)
    nll_criterion = torch.nn.CrossEntropyLoss()

    def run():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        print("Loss", loss.item())
        return loss

    optimizer.step(run)
    print("Final temperature: ", temperature)
