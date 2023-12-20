import torch

from datasets import load_dataset_builder
from datasets import load_dataset

ds_builder = load_dataset_builder("conll2000")
dataset = load_dataset("conll2000", split="train")

# Возьмем коды тэгов из датасета
pos_numbers = {'0': "''''",
            '1': "#",
            '2': "$",
            '3': "(",
            '4': ")",
            '5': ",",
            '6': ".",
            '7': ":",
            '8': "``",
            '9': "CC",
            '10':"CD",
            '11':"DT",
            '12': "EX",
            '13': "FW",
            '14': "IN",
            '15': "JJ",
            '16': "JJR",
            '17': "JJS",
            '18': "MD",
            '19': "NN",
            '20': "NNP",
            '21': "NNPS",
            '22': "NNS",
            '23': "PDT",
            '24': "POS",
            '25': "PRP",
            '26': "PRP$",
            '27': "RB",
            '28': "RBR",
            '29': "RBS",
            '30': "RP",
            '31': "SYM",
            '32': "TO",
            '33': "UH",
            '34': "VB",
            '35': "VBD",
            '36': "VBG",
            '37': "VBN",
            '38': "VBP",
            '39': "VBZ",
            '40': "WDT",
            '41': "WP",
            '42': "WP$",
            '43': "WRB"}


sentences = dataset['tokens'][0:2000]
tags = dataset['pos_tags'][0:2000]


def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

training_data = merge(sentences, tags)

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

num_words = len(word_to_ix)
num_tags = len(pos_numbers)

torch.manual_seed(123)

EMBEDDING_DIM = 100
HIDDEN_DIM = 100

class LSTMTagger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(7872, EMBEDDING_DIM)
        self.lstm = torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)
        self.pos_predictor = torch.nn.Linear(HIDDEN_DIM, 44)

    def forward(self, token_ids):
        embeds = self.embedding_layer(token_ids)
        lstm_out, _ = self.lstm(embeds.view(len(token_ids), 1, -1))
        logits = self.pos_predictor(lstm_out.view(len(token_ids), -1))
        probs = torch.nn.functional.softmax(logits, dim=1)

        return probs

model = LSTMTagger()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

for epoch in range(20):
    running_loss = 0
    for sentence, tags in training_data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor(tags, dtype=torch.long)

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        running_loss += loss

        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, loss = {running_loss}')

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    print(training_data[0][0], tag_scores)