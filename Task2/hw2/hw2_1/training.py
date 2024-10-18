

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import os
import json
import re
import pickle
from torch.utils.data import DataLoader, Dataset
from scipy.special import expit

# ## Data Preprocessing

def data_preprocess():
    """Processes the data and builds word dictionaries."""
    filepath = '/home/agolla/HW2/MLDS_hw2_1_data/'
    with open(filepath + 'training_label.json', 'r') as f:
        data = json.load(f)

    word_count = {}
    for item in data:
        for caption in item['caption']:
            words = re.sub(r'[.!,;?]', ' ', caption).split()
            for word in words:
                word = word.replace('.', '') if '.' in word else word
                word_count[word] = word_count.get(word, 0) + 1

    word_dict = {word: count for word, count in word_count.items() if count > 4}
    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(useful_tokens): w for i, w in enumerate(word_dict)}
    w2i = {w: i + len(useful_tokens) for i, w in enumerate(word_dict)}

    for token, index in useful_tokens:
        i2w[index] = token
        w2i[token] = index
        
    return i2w, w2i, word_dict

def s_split(sentence, word_dict, w2i):
    """Splits and encodes a sentence based on word dictionaries."""
    words = re.sub(r'[.!,;?]', ' ', sentence).split()
    encoded_sentence = [w2i.get(word, 3) for word in words]  # 3 is the <UNK> token
    return [1] + encoded_sentence + [2]  # <SOS> = 1, <EOS> = 2

def annotate(label_file, word_dict, w2i):
    """Annotates the data with encoded sentences."""
    filepath = '/home/agolla/HW2/MLDS_hw2_1_data/' + label_file
    annotated_captions = []
    with open(filepath, 'r') as f:
        data = json.load(f)
    for item in data:
        for caption in item['caption']:
            encoded_caption = s_split(caption, word_dict, w2i)
            annotated_captions.append((item['id'], encoded_caption))
    return annotated_captions

def load_avi_files(files_dir):
    """Loads the avi features from directory."""
    avi_data = {}
    filepath = '/home/agolla/HW2/MLDS_hw2_1_data/' + files_dir
    for file in os.listdir(filepath):
        avi_data[file.split('.npy')[0]] = np.load(os.path.join(filepath, file))
    return avi_data

def create_minibatch(data):
    """Prepares a minibatch of data."""
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data)
    avi_data = torch.stack(avi_data, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        targets[i, :len(cap)] = torch.Tensor(cap[:len(cap)])
    return avi_data, targets, lengths

# ## Dataset Classes

class TrainingData(Dataset):
    """Dataset class for training."""
    def __init__(self, label_file, files_dir, word_dict, w2i):
        self.avi_data = load_avi_files(label_file)
        self.data_pairs = annotate(files_dir, word_dict, w2i)
        
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        video_id, sentence = self.data_pairs[idx]
        avi_features = torch.Tensor(self.avi_data[video_id])
        avi_features += torch.Tensor(avi_features.size()).random_(0, 2000) / 10000.
        return avi_features, torch.Tensor(sentence)

class TestData(Dataset):
    """Dataset class for testing."""
    def __init__(self, test_data_path):
        self.avi_data = [(file.split('.npy')[0], np.load(os.path.join(test_data_path, file)))
                         for file in os.listdir(test_data_path)]
    
    def __len__(self):
        return len(self.avi_data)
    
    def __getitem__(self, idx):
        return self.avi_data[idx]

# ## Model Definitions

class Attention(nn.Module):
    """Attention mechanism."""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_layers = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    
    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, self.hidden_size).repeat(1, seq_len, 1)
        combined_input = torch.cat((encoder_outputs, hidden_state), 2)
        attention_weights = F.softmax(self.attention_layers(combined_input.view(-1, 2 * self.hidden_size)).view(batch_size, seq_len), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context

class EncoderRNN(nn.Module):
    """Encoder RNN."""
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(512, 512, batch_first=True)

    def forward(self, input):
        compressed_input = self.compress(input.view(-1, input.size(-1)))
        compressed_input = self.dropout(compressed_input.view(input.size(0), input.size(1), -1))
        return self.gru(compressed_input)

class DecoderRNN(nn.Module):
    """Decoder RNN."""
    def __init__(self, hidden_size, output_size, word_dim, vocab_size, dropout_p=0.3):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, word_dim)
        self.gru = nn.GRU(hidden_size + word_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_hidden, encoder_outputs, targets=None, training=True, step=None):
        batch_size = encoder_hidden.size(1)
        decoder_input = Variable(torch.ones(batch_size, 1).long().cuda())  # <SOS>
        hidden_state = encoder_hidden

        seq_log_probs, seq_preds = [], []
        targets = self.embedding(targets)
        for i in range(targets.size(1) - 1):
            use_teacher_forcing = random.random() < self.teacher_forcing_ratio(step)
            if training and use_teacher_forcing:
                current_input = targets[:, i]
            else:
                current_input = self.embedding(decoder_input).squeeze(1)

            context = self.attention(hidden_state, encoder_outputs)
            gru_input = torch.cat((current_input, context), dim=1).unsqueeze(1)
            output, hidden_state = self.gru(gru_input, hidden_state)
            log_prob = self.output(output.squeeze(1))
            seq_log_probs.append(log_prob.unsqueeze(1))
            decoder_input = log_prob.max(1)[1].unsqueeze(1)

        return torch.cat(seq_log_probs, dim=1), torch.cat(seq_log_probs, dim=1).max(2)[1]

    def teacher_forcing_ratio(self, step):
        return expit(step / 20 + 0.85)

class Model(nn.Module):
    """Combined encoder-decoder model."""
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, avi_feats, mode, target_sentences=None, step=None):
        encoder_outputs, hidden_state = self.encoder(avi_feats)
        if mode == 'train':
            return self.decoder(hidden_state, encoder_outputs, targets=target_sentences, training=True, step=step)
        elif mode == 'inference':
            return self.decoder.infer(hidden_state, encoder_outputs)

# ## Training Function

def train_epoch(model, optimizer, loss_fn, data_loader, epoch):
    """Train the model for one epoch."""
    model.train()
    for batch_idx, (avi_feats, targets, lengths) in enumerate(data_loader):
        avi_feats, targets = avi_feats.cuda(), targets.cuda()
        optimizer.zero_grad()
        seq_log_probs, _ = model(avi_feats, mode='train', target_sentences=targets, step=epoch)
        targets = targets[:, 1:]  # Ignore <SOS>
        loss = sum([loss_fn(seq_log_probs[:, i], targets[:, i]) for i in range(targets.size(1))])
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

# ## Hyperparameters
hidden_size = 512
vocab_size = len(w2i)
word_dim = 512
output_size = vocab_size
avi_feat_dim = 4096
num_epochs = 20
batch_size = 64

# ## Main Execution
def main():
    i2w, w2i, word_dict = data_preprocess()
    train_data = TrainingData('training_label.json', 'training_data', word_dict, w2i)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=create_minibatch)
    
    encoder = EncoderRNN()
    decoder = DecoderRNN(hidden_size, output_size, word_dim, vocab_size)
    model = Model(encoder, decoder).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        train_epoch(model, optimizer, loss_fn, train_loader, epoch)

if __name__ == "__main__":
    main()
