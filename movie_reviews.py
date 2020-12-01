from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels




def fine_tune_model(dir):

    train_texts, train_labels = read_imdb_split(dir + '/train/')
    test_texts, test_labels = read_imdb_split(dir + '/test/')
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    # Use DistilBert tokenizer. Usually to match the pretrained models, we need to use the same tokenization and
    # numericalization as the model. Fortunately, the tokenizer class from transformers provides the correct
    # pre-process tools that correspond to each pre-trained models.



    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # PAss truncation=True and padding=True, which ensure that all sequences are padded to the same length
    # and are truncated to be no longer maximum input length.

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # Now, let's turn our labels and encodings into a Dataset object.In PyTorch, this is done by subclassing
    # a torch.utils.data.Dataset object and implementing __len__ and __getitem__.

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()

if __name__=="__main__":
    #data_dir=r"/Users/....../Documents/Projects/Bert/Transformers_HuggingFace/data/aclImdb" Directory of the aclImdb data
    fine_tune_model(data_dir)


