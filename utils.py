from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm


def createBertDataset(df, tokenizer, max_length=256):
    encoded_data_train = tokenizer.batch_encode_plus(
        df.text.values.tolist(),
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=True,
        pad_to_max_length=True,
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df.label.values)

    return TensorDataset(input_ids_train, attention_masks_train, labels_train)


def evaluate(model, test_dataset, device="cpu"):
    model.eval()

    test_dataloader = DataLoader(test_dataset,
                                 sampler=SequentialSampler(test_dataset),
                                 batch_size=32)

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in tqdm(test_dataloader):
        batch = tuple(b.to(device) for b in batch)

        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, labels)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = labels.cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(test_dataloader)

    predictions = np.concatenate(predictions, axis=0)
    preds_flat = np.argmax(predictions, axis=1).flatten()

    true_vals = np.concatenate(true_vals, axis=0)
    labels_flat = true_vals.flatten()

    f1 = f1_score(preds_flat, labels_flat, average='macro')
    acc = accuracy_score(preds_flat, labels_flat)

    return loss_val_avg, f1, acc
