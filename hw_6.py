# -*- coding: utf-8 -*-
"""HW_6

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yu2OxmN0l4ginB42KwfVvLVEkFgO_-Ji
"""

!pip install transformers

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

class DistractorsDataset(Dataset):
    def __init__(self, dataset_path = 'trian.json'):
        super().__init__()

        with open('train.json') as json_file:
          train_data = json.load(json_file)

        self.input_list = []
        self.end_of_text_token = '<|endoftext|>'

        for line in train_data:
          dist1 = f"<question>{line['question']}<key>{line['correct_answer']}<distractor>{line['distractor1']}{self.end_of_text_token}"
          self.input_list.append(dist1)
          dist2 = f"<question>{line['question']}<key>{line['correct_answer']}<distractor>{line['distractor2']}{self.end_of_text_token}"
          self.input_list.append(dist2)
          dist3 = f"<question>{line['question']}<key>{line['correct_answer']}<distractor>{line['distractor3']}{self.end_of_text_token}"
          self.input_list.append(dist3)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, item):
        return self.input_list[item]

dataset = DistractorsDataset()
distractors_loader = DataLoader(dataset, batch_size=1, shuffle=True)

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

model = model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_distractor_tens = None
models_folder = "trained_models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(EPOCHS):

    print(f"EPOCH {epoch} started" + '=' * 30)

    for idx, distractor in enumerate(distractors_loader):

        distractor_tens = torch.tensor(tokenizer.encode(distractor[0])).unsqueeze(0).to(device)
        if distractor_tens.size()[1] > MAX_SEQ_LEN:
            continue

        if not torch.is_tensor(tmp_distractor_tens):
            tmp_distractor_tens = distractor_tens
            continue
        else:
            if tmp_distractor_tens.size()[1] + distractor_tens.size()[1] > MAX_SEQ_LEN:
                work_distractor_tens = tmp_distractor_tens
                tmp_distractor_tens = distractor_tens
            else:

                tmp_distractor_tens = torch.cat([tmp_distractor_tens, distractor_tens[:,1:]], dim=1)
                continue


        outputs = model(work_distractor_tens, labels=work_distractor_tens)
        loss, logits = outputs[:2]
        loss.backward()
        sum_loss = sum_loss + loss.detach().data

        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0
            batch_count += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0

    torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_medium_distractor_{epoch}.pt"))

MODEL_EPOCH = 4

models_folder = "trained_models"

model_path = os.path.join(models_folder, f"gpt2_medium_distractor_{MODEL_EPOCH}.pt")
model.load_state_dict(torch.load(model_path))

distractors_output_file_path = f'generated_{MODEL_EPOCH}.distractors'

model.eval()

if os.path.exists(distractors_output_file_path):
    os.remove(distractors_output_file_path)

distractor_num = 0

questions = ['<question>What is the process by which plants convert light energy into chemical energy?<key>Photosynthesis<distractor>',
             '<question>What is the process by which plants convert light energy into chemical energy?<key>Photosynthesis<distractor>',
             '<question>What is the process by which plants convert light energy into chemical energy?<key>Photosynthesis<distractor>',
             '<question>What is the force that causes apples to fall to the ground and governs the motion of celestial bodies?<key>Gravity<distractor>',
             '<question>What is the force that causes apples to fall to the ground and governs the motion of celestial bodies?<key>Gravity<distractor>',
             '<question>What is the force that causes apples to fall to the ground and governs the motion of celestial bodies?<key>Gravity<distractor>',
             '<question>Which scientist is credited with the theory of relativity and the famous equation E=mc^2?<key>Albert Einstein<distractor>',
             '<question>Which scientist is credited with the theory of relativity and the famous equation E=mc^2?<key>Albert Einstein<distractor>',
             '<question>Which scientist is credited with the theory of relativity and the famous equation E=mc^2?<key>Albert Einstein<distractor>'
 ]
with torch.no_grad():

        for question in questions:

            distractor_generation_finished = False

            cur_ids = torch.tensor(tokenizer.encode(question)).unsqueeze(0).to(device)

            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
                if i < 3:
                    n = 20
                else:
                    n = 3
                next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    distractor_generation_finished = True
                    break


            if distractor_generation_finished:

                distractor_num = distractor_num + 1

                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)

                with open('generated_distractors.txt', 'a') as f:
                    f.write(f"{output_text} \n\n")