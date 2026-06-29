#!/usr/bin/env python
"""Aplica denoise + equalização offline em todas as imagens do dataset.

Uso:
    python preprocess_dataset.py --csv dataset.csv --outdir preprocessed

Isso gera:
    preprocessed/  (imagens processadas)
    dataset_preprocessed.csv  (CSV com paths atualizados)

Depois no treino:
    python main.py data.preprocessed_dir=preprocessed
"""

import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from application.preprocessing.PreProcessing import OpenCVPreprocessing


def main():
    parser = argparse.ArgumentParser(description="Preprocessamento offline de imagens")
    parser.add_argument("--csv", default="dataset.csv", help="Caminho para o CSV com paths das imagens")
    parser.add_argument("--outdir", default="preprocessed", help="Diretório de saída")
    parser.add_argument("--img-col", default="img_name", help="Nome da coluna com o path da imagem")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    proc = OpenCVPreprocessing()
    new_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        img_path = row[args.img_col]
        if not os.path.exists(img_path):
            print(f"[AVISO] Imagem não encontrada: {img_path}")
            new_paths.append(img_path)
            continue

        img = Image.open(img_path).convert("RGB")
        processed = proc(img)

        rel_path = os.path.relpath(img_path, start=os.path.dirname(args.csv) if os.path.isfile(args.csv) else ".")
        out_path = os.path.join(args.outdir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        processed.save(out_path, quality=95)
        new_paths.append(out_path)

    df[args.img_col] = new_paths

    out_csv = args.csv.replace(".csv", "_preprocessed.csv")
    df.to_csv(out_csv, index=False)
    print(f"CSV salvo em: {out_csv}")
    print(f"{len(df)} imagens processadas em: {args.outdir}")


if __name__ == "__main__":
    main()
