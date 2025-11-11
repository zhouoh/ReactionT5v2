import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation_utils import ReactionT5Dataset
from task_forward.train import preprocess_df, preprocess_USPTO
from utils import filter_out, seed_everything

warnings.filterwarnings("ignore")


def create_embedding(dataloader, model, device):
    outputs_mean = []
    model.eval()
    model.to(device)
    for inputs in dataloader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        last_hidden_states = output[0]
        input_mask_expanded = (
            inputs["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_states.size())
            .float()
        )
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-6)
        mean_embeddings = sum_embeddings / sum_mask
        outputs_mean.append(mean_embeddings.detach().cpu().numpy())

    return np.concatenate(outputs_mean, axis=0)


def generate_embedding(
    input_data,
    test_data=None,
    input_max_length=400,
    model_name_or_path="sagawa/ReactionT5v2-forward",
    batch_size=5,
    output_dir="./",
    debug=False,
    seed=42,
):
    """
    Generates embeddings for reaction data.
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
    model = T5EncoderModel.from_pretrained(model_name_or_path).to(device)
    model.eval()

    if isinstance(input_data, str):
        input_df = pd.read_csv(input_data)
    else:
        input_df = input_data
        
    input_df = filter_out(input_df, ["REACTANT", "PRODUCT"])
    input_df = preprocess_df(input_df, drop_duplicates=False)

    if test_data:
        input_data_copy = preprocess_USPTO(input_df.copy())
        if isinstance(test_data, str):
            test_df = pd.read_csv(test_data)
        else:
            test_df = test_data
        test_df = filter_out(test_df, ["REACTANT", "PRODUCT"])
        USPTO_test = preprocess_USPTO(test_df)
        input_df = input_df[
            ~input_data_copy["pair"].isin(USPTO_test["pair"])
        ].reset_index(drop=True)

    input_df.to_csv(os.path.join(output_dir, "input_data.csv"), index=False)

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

    outputs = create_embedding(dataloader, model, device)
    np.save(os.path.join(output_dir, "embedding_mean.npy"), outputs)
    return outputs
