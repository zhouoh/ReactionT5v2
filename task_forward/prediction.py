import gc
import os
import sys
import warnings

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation_utils import (
    ReactionT5Dataset,
    decode_output,
    save_multiple_predictions,
)
from task_forward.train import preprocess_df
from utils import seed_everything

warnings.filterwarnings("ignore")


def predict(
    input_data,
    model_name_or_path="sagawa/ReactionT5v2-forward",
    input_max_length=400,
    output_min_length=1,
    output_max_length=300,
    num_beams=5,
    num_return_sequences=5,
    batch_size=5,
    output_dir="./",
    seed=42,
    debug=False,
):
    """
    Performs reaction product prediction.

    Args:
        input_data (pd.DataFrame or str): Path to the input CSV file or a pandas DataFrame.
        model_name_or_path (str): Name or path of the finetuned model.
        input_max_length (int): Maximum token length of input.
        output_min_length (int): Minimum token length of output.
        output_max_length (int): Maximum token length of output.
        num_beams (int): Number of beams for beam search.
        num_return_sequences (int): Number of predictions to return.
        batch_size (int): Batch size for prediction.
        output_dir (str): Directory to save predictions.
        seed (int): Seed for reproducibility.
        debug (bool): Debug mode.

    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seed_everything(seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.abspath(model_name_or_path)
        if os.path.exists(model_name_or_path)
        else model_name_or_path,
        return_tensors="pt",
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        os.path.abspath(model_name_or_path)
        if os.path.exists(model_name_or_path)
        else model_name_or_path
    ).to(device)
    model.eval()
    model = torch.compile(model)

    if isinstance(input_data, str):
        input_df = pd.read_csv(input_data)
    else:
        input_df = input_data

    input_df = preprocess_df(input_df, drop_duplicates=False)

    # Create a temporary config object for ReactionT5Dataset
    class TempCFG:
        def __init__(self):
            self.input_max_length = input_max_length
            self.tokenizer = tokenizer
            self.debug = debug

    dataset_cfg = TempCFG()
    dataset = ReactionT5Dataset(dataset_cfg, input_df)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    all_sequences, all_scores = [], []
    for inputs in tqdm(dataloader, total=len(dataloader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                min_length=output_min_length,
                max_length=output_max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Create a temporary config object for decode_output
        class DecodeCFG:
            def __init__(self):
                self.num_return_sequences = num_return_sequences
                self.tokenizer = tokenizer
                self.num_beams = num_beams

        decode_cfg = DecodeCFG()
        sequences, scores = decode_output(output, decode_cfg)
        all_sequences.extend(sequences)
        if scores:
            all_scores.extend(scores)
        del output

    # Create a temporary config object for save_multiple_predictions
    class SaveCFG:
        def __init__(self):
            self.num_return_sequences = num_return_sequences
            self.debug = debug

    save_cfg = SaveCFG()
    output_df = save_multiple_predictions(input_df, all_sequences, all_scores, save_cfg)

    output_df.to_csv(os.path.join(output_dir, "output.csv"), index=False)
    return output_df


def predict_single(
    reaction_input_string,
    model=None,
    tokenizer=None,
    model_name_or_path="sagawa/ReactionT5v2-forward",
    input_max_length=400,
    output_min_length=1,
    output_max_length=300,
    num_beams=5,
    num_return_sequences=5,
    device=None,
):
    """
    Performs reaction product prediction for a single reaction input string.

    Args:
        reaction_input_string (str): The reaction input string, e.g., "REACTANT:c1ccccc1.O=C=O.ClREAGENT:c1ccccc1".
        model (AutoModelForSeq2SeqLM, optional): Pre-loaded model. Defaults to None.
        tokenizer (AutoTokenizer, optional): Pre-loaded tokenizer. Defaults to None.
        model_name_or_path (str): Name or path of the finetuned model.
        input_max_length (int): Maximum token length of input.
        output_min_length (int): Minimum token length of output.
        output_max_length (int): Maximum token length of output.
        num_beams (int): Number of beams for beam search.
        num_return_sequences (int): Number of predictions to return.
        device (torch.device, optional): Device to run the model on. Defaults to None.

    Returns:
        list: A list of predicted product SMILES.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.abspath(model_name_or_path)
            if os.path.exists(model_name_or_path)
            else model_name_or_path,
            return_tensors="pt",
        )

    if model is None:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            os.path.abspath(model_name_or_path)
            if os.path.exists(model_name_or_path)
            else model_name_or_path
        ).to(device)
        model.eval()
        model = torch.compile(model)

    input_df = pd.DataFrame({"input": [reaction_input_string]})

    # Create a temporary config object for ReactionT5Dataset
    class TempCFG:
        def __init__(self):
            self.input_max_length = input_max_length
            self.tokenizer = tokenizer
            self.debug = False

    dataset_cfg = TempCFG()
    dataset = ReactionT5Dataset(dataset_cfg, input_df)
    inputs = dataset[0]
    inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            min_length=output_min_length,
            max_length=output_max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Create a temporary config object for decode_output
    class DecodeCFG:
        def __init__(self):
            self.num_return_sequences = num_return_sequences
            self.tokenizer = tokenizer
            self.num_beams = num_beams

    decode_cfg = DecodeCFG()
    #print(decode_cfg.num_beams)
    sequences, scores = decode_output(output, decode_cfg)

    return sequences
