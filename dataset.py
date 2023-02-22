import argparse
from itertools import product
from pathlib import Path

import librosa
import pandas as pd
import textgrids
from tqdm import tqdm

import torch


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commonphone_path", type=Path, help="Path to CommonPhone Dataset.")
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
        self.audio_df = df
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_df)

    def __getitem__(self, i):
        x, _ = librosa.load(self.audio_df.iloc[i].audio, sr=self.sample_rate, mono=True)
        y = self.df[self.df["audio"] == self.audio_df.iloc[i].audio]

        return x, y[["phone", "min", "max"]]


if __name__ == "__main__":
    args = _get_args()
    df = _prepare_commonphone(args.commonphone_path)
    df.to_csv(args.output_path / "commonphone.csv.gz", index=False, compression="gzip")
