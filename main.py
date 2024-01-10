"""
Model training & hyperparameter optimization

Author: M.Cihat UNAL
"""

# !pip install transformers optuna nltk scikit-learn pandas

import gc
import os
import random
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from preprocess import preprocessing
from bert_model import BertModel
from utils import create_bert_dataset, evaluate

warnings.filterwarnings("ignore")

gc.collect()
torch.cuda.empty_cache()


SEED_VAL = 2
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
os.environ['PYTHONHASHSEED'] = str(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("cmp711_5class.csv")
df = df.reset_index(drop=True)

# Check how many labels are there in the dataset
unique_labels = df.label.unique().tolist()

# Map each label into its id representation and vice versa
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

df["text"] = df["text"].apply(preprocessing)
df["label"] = df["label"].apply(lambda x: labels_to_ids[x])

# Split data into train & test
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=SEED_VAL),
                                     [int(.70 * len(df)), int(.80 * len(df))])


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    # accuracy_score(pred_flat, labels_flat)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train_loop(model, optimizer, batch_size, epochs, es_patience):
    """Train loop - validation"""
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=batch_size)

    # put device into GPU
    model = model.to(device)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * epochs)

    last_loss = 100
    patience = 0
    for epoch in range(epochs):

        # print out active_run
        # print("Epoch: %s\n" % (epoch + 1))

        model.train()
        loss_train_total = 0
        acc_train_total = 0

        loop = tqdm(enumerate(train_dataloader), leave=False, total=len(train_dataloader))
        for step, batch in loop:
            # clear previously calculated gradients (Zero the gradients to start fresh next time.)
            optimizer.zero_grad()

            batch = [r.to(device) for r in batch]
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Calculate loss & backpropagation
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            # Calculate accuracy
            logits = outputs[1]
            classes = torch.argmax(logits, dim=1)
            acc_train_total += torch.mean((classes == labels).float())

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Show progress while training
            loop.set_description(f'Epoch = {epoch + 1}/{epochs}, training_loss: {(loss.item() / len(batch)):.3f}')

        train_loss = loss_train_total / len(train_dataloader)

        if train_loss >= last_loss:
            patience += 1

            if patience == es_patience:
                print("Early Stopping!\n")
                with open("es-log.txt", "a+", encoding="utf-8") as es:
                    es.write(f"{MODEL_NAME} - stopped at {epoch} epoch\n")

                return model

        else:
            patience = 0

        last_loss = train_loss

        # # Validation
        model.eval()

        val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=32)

        loss_val_total = 0
        total_eval_accuracy = 0

        # print("\nValidating...")
        for batch in tqdm(val_dataloader, total=len(val_dataloader), leave=False):
            batch = tuple(b.to(device) for b in batch)

            input_ids, attention_mask, labels = batch

            with torch.no_grad():
                outputs = model(input_ids, attention_mask, labels)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = labels.cpu().numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        if epoch == epochs - 1:
            print(f'Training Loss: {train_loss: .3f}')
            print(f'Train Acc.: {acc_train_total / len(train_dataloader)}')

            # Report the final accuracy for this validation run.
            val_acc_avg = total_eval_accuracy / len(val_dataloader)
            loss_val_avg = loss_val_total / len(val_dataloader)

            print('Val. Average loss: {:.3f}'.format(loss_val_avg))
            print('Val. Average Acc.: {:.3f}\n'.format(val_acc_avg))

    return model


# Hyperparameter Optimization
def objective(trial):
    gc.collect()
    torch.cuda.empty_cache()

    raw_model = BertModel(MODEL_NAME, unique_labels)
    print(f"{MODEL_NAME} - {trial.number} is training...")

    epochs = trial.suggest_int("epochs", low=4, high=7)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 48, 64])
    LR = trial.suggest_categorical("LR", [1e-3, 3e-3, 3e-4, 1e-5, 3e-5, 5e-5])
    WD = trial.suggest_categorical("WD", [1e-4, 1e-5, 2e-5])
    OPT = trial.suggest_categorical("OPT", ["AdamW", "SGD", "RMSprop"])

    optimizer = getattr(optim, OPT)(raw_model.parameters(), lr=LR, weight_decay=WD)

    parameters = {
        "model": raw_model,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "epochs": epochs,
        "es_patience": 2
    }

    trained_model = train_loop(**parameters)
    _, f1, _ = evaluate(trained_model, test_dataset)

    if f1 > 0.63:
        trained_model.save(f"./{MODEL_NAME}-{trial.number}")

    return f1


if __name__ == "__main__":
    model_names = ["distilbert-base-uncased", "albert-base-v2", "huawei-noah/TinyBERT_General_6L_768D", "roberta-base",
                   "bert-base-uncased", "google/electra-base-discriminator", "YituTech/conv-bert-base"]

    for model in model_names:
        MODEL_NAME = model

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Create datasets For BERT
        train_dataset = create_bert_dataset(df_train, tokenizer)
        val_dataset = create_bert_dataset(df_val, tokenizer)
        test_dataset = create_bert_dataset(df_test, tokenizer)

        # We want to maximize the f1
        study = optuna.create_study(study_name='airbert-hyperopt',
                                    direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED_VAL))

        # Optimize the objective using 50 different trials
        study.optimize(objective, n_trials=50)

        with open("param_new.txt", "a+", encoding="utf-8") as file:
            trial = study.best_trial
            file.write(f"Model Name: {MODEL_NAME}\n")
            file.write(f"Best Score: {trial.value}\n")
            file.write("Best Params: \n")
            for key, value in trial.params.items():
                file.write(f'  {key}: {value}\n')
            file.write("*" * 60)
            file.write("\n")

        del train_dataset, val_dataset, test_dataset, tokenizer, MODEL_NAME

    # # See the evaluation of any model
    # load_model = BertModel("<model-name>", unique_labels)
    # load_model = load_model.to(device)
    # tokenizer = AutoTokenizer.from_pretrained("<model-name>")
    # test_dataset = createBertDataset(df_test, tokenizer)
    # evaluate(load_model, test_dataset)

    # test_loss, test_f1, test_acc = evaluate(load_model, test_dataset)

    # print(f'Test loss: {test_loss}')
    # print(f'F1 Score (Weighted): {test_f1}')
    # print(f'Acc Score: {test_acc}')
