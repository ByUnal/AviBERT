from torch import nn


class BertModel(nn.Module):
    def __init__(self, checkpoint, unique_labels):
        super(BertModel, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                                       num_labels=len(unique_labels),
                                                                       output_attentions=False,
                                                                       output_hidden_states=False)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        return output

    def save(self, path="model"):
        self.bert.save_pretrained(f"./{path}")
