import os
import sys
import warnings

import pandas as pd
import rdkit
from rdkit import Chem
from transformers import AutoTokenizer

rdkit.RDLogger.DisableLog("rdApp.*")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import canonicalize, seed_everything

warnings.filterwarnings("ignore")


def remove_space(row, num_beams):
    for i in range(num_beams):
        if f"{i}th" in row:
            row[f"{i}th"] = row[f"{i}th"].replace(" ", "")
    return row


def calculate_accuracy_function(
    input_data_path: str,
    target_data_path: str,
    target_col: str,
    model_name_or_path: str = "sagawa/ReactionT5v2-retrosynthesis",
    num_beams: int = 5,
    seed: int = 42,
):
    """
    Calculates reaction prediction accuracy and invalidity.

    Args:
        input_data_path (str): Path to the input data (predictions).
        target_data_path (str): Path to the target data (ground truth).
        target_col (str): Name of the target column in target_data_path.
        model_name_or_path (str, optional): Model name or path for tokenizer.
                                            Defaults to "sagawa/ReactionT5v2-retrosynthesis".
        num_beams (int, optional): Number of beams used for beam search. Defaults to 5.
        seed (int, optional): Seed for reproducibility. Defaults to 42.

    Returns:
        dict: A dictionary containing top-k accuracies and invalidity score.
    """
    seed_everything(seed=seed)



    df = pd.read_csv(input_data_path)
    #print(df.columns)
    df[[f"{i}th" for i in range(num_beams)]] = df[
        [f"{i}th" for i in range(num_beams)]
    ].fillna(" ")
    df["target"] = pd.read_csv(target_data_path)[target_col].values
    df = df.apply(lambda row: remove_space(row, num_beams), axis=1)

    top_k_invalidity = num_beams

    top1_list, top2_list, top3_list, top5_list = [], [], [], []
    invalidity_list = []

    for idx, row in df.iterrows():
        target = canonicalize(row["target"])
        
        found_in_top1 = False
        found_in_top2 = False
        found_in_top3 = False
        found_in_top5 = False

        current_invalid_count = 0
        for i in range(num_beams):
            pred_col = f"{i}th"
            if pred_col in row:
                prediction = canonicalize(row[pred_col])
                
                # Check for accuracy
                if prediction == target:
                    if i == 0:
                        found_in_top1 = True
                    if i < 2: # 0th or 1st
                        found_in_top2 = True
                    if i < 3: # 0th, 1st, or 2nd
                        found_in_top3 = True
                    if i < 5: # 0th to 4th
                        found_in_top5 = True
                
                # Check for invalidity
                mol = Chem.MolFromSmiles(row[pred_col].rstrip("."))
                if not isinstance(mol, Chem.rdchem.Mol):
                    current_invalid_count += 1
            else:
                # If column doesn't exist, it's an invalid prediction for this beam
                current_invalid_count += 1 # Or handle as a missing prediction

        top1_list.append(1 if found_in_top1 else 0)
        top2_list.append(1 if found_in_top2 else 0)
        top3_list.append(1 if found_in_top3 else 0)
        top5_list.append(1 if found_in_top5 else 0)
        invalidity_list.append(current_invalid_count)

    num_samples = len(df)
    results = {}
    if num_samples > 0:
        results["top1_accuracy"] = sum(top1_list) / num_samples
        results["top2_accuracy"] = sum(top2_list) / num_samples
        results["top3_accuracy"] = sum(top3_list) / num_samples
        results["top5_accuracy"] = sum(top5_list) / num_samples
        results["top_k_invalidity_percentage"] = sum(invalidity_list) / (num_samples * top_k_invalidity) * 100
    else:
        results["top1_accuracy"] = 0.0
        results["top2_accuracy"] = 0.0
        results["top3_accuracy"] = 0.0
        results["top5_accuracy"] = 0.0
        results["top_k_invalidity_percentage"] = 0.0


    print(f"Top 1 accuracy: {results['top1_accuracy']}")
    print(f"Top 2 accuracy: {results['top2_accuracy']}")
    print(f"Top 3 accuracy: {results['top3_accuracy']}")
    print(f"Top 5 accuracy: {results['top5_accuracy']}")
    print(
        f"Top {top_k_invalidity} Invalidity: {results['top_k_invalidity_percentage']}"
    )

    return results