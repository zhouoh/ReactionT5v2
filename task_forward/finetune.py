import os
import sys
import warnings

import datasets
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from task_forward.train import preprocess_df
from utils import filter_out, get_accuracy_score, preprocess_dataset, seed_everything

# Suppress warnings and disable progress bars
warnings.filterwarnings("ignore")
datasets.utils.logging.disable_progress_bar()


def finetune(
    train_data_path,
    valid_data_path,
    similar_reaction_data_path=None,
    output_dir="t5",
    model_name_or_path="sagawa/ReactionT5v2-forward",
    debug=False,
    epochs=3,
    lr=2e-5,
    batch_size=32,
    input_max_length=200,
    target_max_length=150,
    eval_beams=5,
    target_column="PRODUCT",
    weight_decay=0.01,
    evaluation_strategy="epoch",
    eval_steps=None,
    save_strategy="epoch",
    save_steps=500,
    logging_strategy="epoch",
    logging_steps=500,
    save_total_limit=2,
    fp16=False,
    disable_tqdm=False,
    seed=42,
    sampling_num=-1,
):
    """
    Finetunes a reaction prediction model.
    """
    disable_tqdm = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=seed)

    train = preprocess_df(
        filter_out(pd.read_csv(train_data_path), ["REACTANT", "PRODUCT"])
    )
    valid = preprocess_df(
        filter_out(pd.read_csv(valid_data_path), ["REACTANT", "PRODUCT"])
    )
    if sampling_num > 0:
        train = train.sample(n=sampling_num, random_state=seed).reset_index(
            drop=True
        )

    if similar_reaction_data_path:
        similar = preprocess_df(
            filter_out(
                pd.read_csv(similar_reaction_data_path), ["REACTANT", "PRODUCT"]
            )
        )
        train = pd.concat([train, similar], ignore_index=True)

    for col in ["REAGENT"]:
        train[col] = train[col].fillna(" ")
        valid[col] = valid[col].fillna(" ")
    train["input"] = "REACTANT:" + train["REACTANT"] + "REAGENT:" + train["REAGENT"]
    valid["input"] = "REACTANT:" + valid["REACTANT"] + "REAGENT:" + valid["REAGENT"]

    if debug:
        train = train[: int(len(train) / 40)].reset_index(drop=True)
        valid = valid[: int(len(valid) / 40)].reset_index(drop=True)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train[["input", "PRODUCT"]]),
            "validation": Dataset.from_pandas(valid[["input", "PRODUCT"]]),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.abspath(model_name_or_path)
        if os.path.exists(model_name_or_path)
        else model_name_or_path,
        return_tensors="pt",
        local_files_only=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        os.path.abspath(model_name_or_path)
        if os.path.exists(model_name_or_path)
        else model_name_or_path,
        local_files_only=True,
    ).to(device)

    class TempCFG:
        def __init__(self):
            self.input_max_length = input_max_length
            self.target_max_length = target_max_length
            self.target_column = target_column
            self.tokenizer = tokenizer

    cfg = TempCFG()

    tokenized_datasets = dataset.map(
        lambda examples: preprocess_dataset(examples, cfg),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir,
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_strategy=logging_strategy,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 4,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=fp16,
        disable_tqdm=disable_tqdm,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
    )

    model.config.eval_beams = eval_beams
    model.config.max_length = target_max_length

    class EvalCFG:
        def __init__(self):
            self.tokenizer = tokenizer
            self.target_column = target_column

    eval_cfg = EvalCFG()

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_preds: get_accuracy_score(eval_preds, eval_cfg),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train()
    trainer.save_model("./best_model")
    return trainer
