import argparse
from itertools import product
from pathlib import Path

import librosa
import pandas as pd
import praatio.textgrid
import textgrids
from tqdm import tqdm

import torch


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset.")
    parser.add_argument("--dataset_type", type=str, choices=["commonphone", "ssnce", "torgo", "qolt"])
    parser.add_argument("--output_path", type=Path, help="Output csv folder")
    return parser.parse_args()


def get_vocab(df):
    vocabs = df["phone"].unique()
    index_to_vocab = dict(enumerate(sorted(vocabs)))
    vocab_to_index = {v: i for i, v in index_to_vocab.items()}
    return index_to_vocab, vocab_to_index


def _prepare_commonphone(commonphone_path: Path):
    langs = ("de", "en", "es", "fr", "it", "ru")
    splits = ("train", "dev", "test")
    rows = []
    for lang, split in tqdm(product(langs, splits)):
        cp_path = commonphone_path / lang
        df = pd.read_csv(cp_path / f"{split}.csv")

        for _, row in df.iterrows():
            audio_file_name = cp_path / "wav" / row["audio file"].replace(".mp3", ".wav")
            grid_file_name = cp_path / "grids" / f"{audio_file_name.stem}.TextGrid"

            grid = textgrids.TextGrid(grid_file_name)
            for g in grid["MAU"]:
                rows.append({
                    "audio": audio_file_name,
                    "language": lang,
                    "id": row["id"],
                    "split": split,
                    "sentence": row["text"],
                    "min": g.xmin,
                    "max": g.xmax,
                    "phone": g.text,
                })
    return pd.DataFrame(rows)


def _prepare_ssnce(ssnce_path: Path):
    labels = ("0_healthy", "1_mild", "2_moderate", "3_severe")
    vocab = {'#': '(...)', 'SIL': '(...)', 'a': 'a', 'aa': 'aː', 'ai': 'aɪ', 'b': 'b', 'c': 'tʃ', 'dx': 'd', 'd': 'ð', 'e': 'e', 'ee': 'eː', 'g': 'g', 'h': 'h', 'i': 'i', 'ii': 'iː', 'j': 'dʒ', 'k': 'k', 'l': 'l', 'lx': 'l', 'm': 'm', 'n': 'n', 'nx': 'n', 'nd': 'n', 'nj': 'ɲ', 'ng': 'ŋ', 'o': 'o', 'oo': 'oː', 'p': 'p', 'r': 'r', 'rx': 'r', 's': 's', 'sx': 'ʃ', 'tx': 't', 't': 'θ', 'u': 'u', 'uu': 'uː', 'y': 'ʝ', 'eu': 'ɨ', 'zh': 'r', 'w': 'w'}
    rows = []
    for label in labels:
        index, label_name = label.split("_")
        index = int(index)

        for audio_path in (ssnce_path / label).glob("*.wav"):
            grid_path = audio_path.with_suffix(".TextGrid")
            grid = praatio.textgrid.openTextgrid(grid_path, includeEmptyIntervals=False)
            for entry in grid.getTier("phonemes").entries:
                rows.append({
                    "audio": audio_path,
                    "label": index,
                    "label_name": label_name,
                    "min": entry.start,
                    "max": entry.end,
                    "phone": vocab[entry.label],
                })
    return pd.DataFrame(rows)


def _prepare_torgo(torgo_path: Path):
    labels = ("0_healthy", "1_mild", "2_moderate", "3_severe")
    vocab = {'#': '(...)', '@': '(...)', 'OY1': 'ɔɪ', 'T': 't', 'UH1': 'ʊ', 'ZH': 'ʒ', 'A01': 'ɒ', 'SH': 'ʃ', 'G': 'g', 'B': 'b', 'k': 'k', 'AA2': 'ɑ', 'EY1': 'ei', 'AA': 'ɑ', 'EI': 'ei', 'AW1': 'aʊ', 'ER0': 'eə', 'Y': 'y', 'IH2': 'ɪ', 'AE2': 'æ', 'CH': 'tʃ', 'A02': 'ɒ', 'AE': 'æ', 'UW2': 'ʉ', 'DH': 'ð', 'EH0': 'ɛ', 'P': 'p', 'F': 'f', 'AH0': 'ə', 'E': '#', 'd': 'd', 'EY2': 'ei', 'OW0': 'əʊ', 'IY0': 'i', 'NG': 'ŋ', 'AE0': 'æ', 'JH': 'dʒ', 'AY1': 'aɪ', 'AO1': 'ɒ', 'OW2': 'əʊ', 'EY': 'æ', 'sp': '#', 'sil': '#', 'IY1': 'i', 'AH2': 'ɐ', 'ER1': 'eə', 'OW1': 'əʊ', 'K': 'k', 'L': 'l', 'EH1': 'ɛ', 'OU': 'əʊ', 'AA1': 'ɑ', 'AY': 'aɪ', 'AW2': 'aʊ', 'R': 'r', 'HH': 'h', 'H': 'h', 'IH1': 'ɪ', 'UW1': 'ʉ', 'S': 's', '3': '#', 'EY0': 'ei', 'IH0': 'ɪ', 'Z': 'z', 'W': 'w', 'AH1': 'ɐ', 'AY2': 'aɪ', 'N': 'n', 'AO2': 'ɒ', 'UW': 'ʉ', 'EH2': 'ɛ', 'V': 'v', 'U': 'u', 'IY2': 'i', 'A0': 'ɒ', 'AE1': 'æ', 'UW0': 'ʉ', 'TH': 'θ', 'M': 'm', 'A': 'ɑ', 'D': 'd', 'UH': 'ɐ', 'IH': 'ɪ', 'ER': 'eə', 'AH': '#', 'OW': 'aʊ', 'IY': 'i', 'EH': 'ɛ', 'AI': 'aɪ', 'AW': 'aʊ'}
    rows = []
    for label in labels:
        index, label_name = label.split("_")
        index = int(index)

        for audio_path in (torgo_path / label).glob("*.wav"):
            grid_path = audio_path.with_suffix(".TextGrid")
            grid = praatio.textgrid.openTextgrid(grid_path, includeEmptyIntervals=False)
            for entry in grid.getTier("phones").entries:
                rows.append({
                    "audio": audio_path,
                    "label": index,
                    "label_name": label_name,
                    "min": entry.start,
                    "max": entry.end,
                    "phone": vocab[entry.label],
                })
    return pd.DataFrame(rows)


def _prepare_qolt(qolt_path: Path):
    labels = ("0_healthy", "1_mild", "2_mildmod", "3_modsev", "4_sev")
    vocab = {'d': 'd', 'dʑ': 'ts', 'e': 'e', 'h': 'h', 'i': 'i', 'j': 'j', 'k': 'k', 'k̚': 'k', 'k͈': 'k', 'm': 'm', 'n': 'n', 'o': 'o', 'oː': 'o', 'p': 'p', 'pʰ': 'p', 'p̚': 'p', 'p͈': 'p', 'spn': '#', 'sʰ': 's', 's͈': 's', 't': 't', 'tɕ': 'ts', 'tɕʰ': 'ts', 'tʰ': 't', 't̚': 't', 'u': 'u', 'w': 'w', 'ŋ': 'ŋ', 'ɐ': 'ɐ', 'ɕʰ': 's', 'ɛː': 'ɛ', 'ɡ': 'ɡ', 'ɨ': 'ɨ', 'ɭ': 'l', 'ɸ': 'h', 'ɾ': 'r', 'ʌ': 'ʌ'}
    rows = []
    for label in labels:
        index, label_name = label.split("_")
        index = int(index)

        for audio_path in (qolt_path / label).glob("*.wav"):
            grid_path = audio_path.with_suffix(".TextGrid")
            if not grid_path.exists():
                continue

            grid = praatio.textgrid.openTextgrid(grid_path, includeEmptyIntervals=False)
            for entry in grid.getTier("phones").entries:
                rows.append({
                    "audio": audio_path,
                    "label": index,
                    "label_name": label_name,
                    "min": entry.start,
                    "max": entry.end,
                    "phone": vocab[entry.label],
                })
    return pd.DataFrame(rows)


class PhoneRecognitionDataset(torch.utils.data.Dataset):
    """
    * What should be saved in CSV file?
    We read the "audio" column and "label" column via pandas.read_csv.
    "audio" column should contain the absolute path to the audio, and "label" should be integer: 0, 1, 2.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sample_rate: int = 16000
    ):
        self.df = df
        self.audios = df.audio.unique()
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, i):
        x, _ = librosa.load(self.audios[i], sr=self.sample_rate, mono=True)
        y = self.df[self.df["audio"] == self.audios[i]]

        return x, y[["phone", "min", "max"]]


if __name__ == "__main__":
    args = _get_args()
    _prepare = {"commonphone": _prepare_commonphone, "ssnce": _prepare_ssnce, "torgo": _prepare_torgo, "qolt": _prepare_qolt}[args.dataset_type]
    df = _prepare(args.dataset_path)
    df.to_csv(args.output_path / f"{args.dataset_type}.csv.gz", index=False, compression="gzip")
