import argparse
import pickle
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import Wav2Vec2FeatureExtractor

from dataset import PhoneRecognitionDataset, get_vocab
from model import Wav2Vec2Recognizer, Wav2Vec2ConvRecognizer
from loss import get_loss


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)

    # Training Settings
    parser.add_argument("--gpu", default=None, type=int, help="Default to CPU. Input GPU index (integer) to use GPU.")
    parser.add_argument("--bestkeep_metric", default="accuracy", type=str)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--loss", type=str, default="ctc_like")

    # Model Settings
    parser.add_argument("--model", default="facebook/wav2vec2-xls-r-300m", type=str)
    parser.add_argument("--use_conv_only", default=True, type=bool)

    # Optimizer Settings
    parser.add_argument("--optim", default="AdamW", type=str)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Dataset settings
    parser.add_argument("--commonphone_csv", required=True, type=Path)

    return parser.parse_args()


def _prepare_model(model_name, vocab_size, use_conv_only):
    ModelClass = {False: Wav2Vec2Recognizer, True: Wav2Vec2ConvRecognizer}[use_conv_only]
    model = ModelClass(model_name, vocab_size)
    model.freeze_conv_features()
    return model


def _get_logger(tb_path):
    writer = SummaryWriter(log_dir=tb_path)
    step_acc = defaultdict(int)
    def _log(name, value):
        writer.add_scalar(name, value, step_acc[name])
        step_acc[name] += 1
    return _log


def _get_collator(model, vocab_to_index, _get_feat_extract_output_lengths):
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model)
    def _collate(batch):
        audios = [b[0] for b in batch]
        audios = processor(raw_speech=audios, sampling_rate=16000, padding=True)
        audios = torch.FloatTensor(audios["input_values"])

        _, max_length = audios.shape
        max_length = _get_feat_extract_output_lengths(max_length).item()

        labels = []
        for _, _df in batch:
            label = []
            prev_loc = 0

            for _, row in _df.iterrows():
                index = vocab_to_index[row["phone"]]
                loc = _get_feat_extract_output_lengths(int(row["max"] * 16000)).item()
                if loc > prev_loc:
                    label += [index] * (loc - prev_loc)
                    prev_loc = loc

            label += [-100] * (max_length - len(label))
            labels.append(label)
        labels = torch.LongTensor(labels)

        return audios, labels

    return _collate


def _prepare_data(df, batch_size, collator):
    train_ds = torch.utils.data.DataLoader(
        PhoneRecognitionDataset(df[df.split == "train"]),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        collate_fn=collator,
    )
    valid_ds = torch.utils.data.DataLoader(
        PhoneRecognitionDataset(df[df.split == "dev"]),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        collate_fn=collator,
    )
    test_ds = torch.utils.data.DataLoader(
        PhoneRecognitionDataset(df[df.split == "test"]),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        collate_fn=collator,
    )
    return train_ds, valid_ds, test_ds


def _train(model, device, optim, loss_fn, dataloader, logger):
    model.train()

    correct_acc = 0
    wrong_acc = 0
    loss_acc = 0.0

    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)

        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()

        logger("train/loss", loss.item())
        loss_acc += loss.item()

        correct_acc += ((logits.argmax(-1) == y) * (y >= 0)).sum()
        wrong_acc += ((logits.argmax(-1) != y) * (y >= 0)).sum()

    logger("train/avg_loss", loss_acc / len(dataloader))
    logger("train/accuracy", correct_acc / (correct_acc + wrong_acc))


def _eval(model, device, dataloader, logger, metric_funcs, mode):
    model.eval()

    preds_acc = []
    labels_acc = []
    for x, y in tqdm(dataloader):
        preds = model(x.to(device))
        preds_acc.append(preds.detach().cpu().numpy())
        labels_acc.append(y.numpy())
    preds_acc = np.concatenate(preds_acc)
    labels_acc = np.concatenate(labels_acc)

    eval_results = {}
    for name, func in metric_funcs.items():
        eval_results[name] = func(labels_acc, preds_acc)
        logger(f"{mode}/{name}", eval_results[name])

    return {"preds": preds_acc, "labels": labels_acc, "metrics": eval_results}


def _discretize_metric(metric):
    def _metric(y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.argmax(-1).flatten()
        mask = y_true >= 0
        return metric(y_true[mask], y_pred[mask])
    return _metric


_metrics = {
    "accuracy": _discretize_metric(metrics.accuracy_score),
}


if __name__ == "__main__":
    args = _get_args()
    print(args)

    exp_dir = Path("exp") / f"{args.exp_name}_{datetime.today().isoformat()}"
    epoch_dir = exp_dir / "epochs"
    epoch_dir.mkdir(exist_ok=False, parents=True)

    logger = _get_logger(exp_dir / "logs")
    device = torch.device("cpu" if args.gpu is None else args.gpu)

    df = pd.read_csv(args.commonphone_csv, compression="gzip")
    index_to_vocab, vocab_to_index = get_vocab(df)

    model = _prepare_model(args.model, len(index_to_vocab), args.use_conv_only).to(device)

    collator = _get_collator(args.model, vocab_to_index, model.get_feat_length)
    train_dataloader, valid_dataloader, test_dataloader = _prepare_data(df, args.batch_size, collator)

    optim = getattr(torch.optim, args.optim)(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = get_loss(args.loss)

    best_epoch, best_metric = None, None
    for epoch in range(args.num_epochs):
        _train(model, device, optim, loss_fn, train_dataloader, logger)
        _eval_results = _eval(model, device, valid_dataloader, logger, _metrics, "valid")

        print(f"Epoch {epoch}")
        print(_eval_results["metrics"])

        epochwise_dir = epoch_dir / f"{epoch:04d}"
        epochwise_dir.mkdir(exist_ok=False, parents=True)
        pickle.dump(_eval_results, open(epochwise_dir / "eval_results.pkl", "wb"))

        if best_epoch is None or best_metric < _eval_results["metrics"][args.bestkeep_metric]:
            best_epoch = epoch
            best_metric = _eval_results["metrics"][args.bestkeep_metric]
            torch.save(model.state_dict(), exp_dir / "best.pt")

    model.load_state_dict(torch.load(exp_dir / "best.pt"))
    _test_results = _eval(model, device, test_dataloader, logger, _metrics, "test")
    pickle.dump(_test_results, open(exp_dir / "test_results.pkl", "wb"))

    print("Training Finished!")
    print(_test_results)
