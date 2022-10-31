import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import metrics
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from dataset import CHAR_SMI_SET_LEN
from sklearn.metrics import roc_curve, precision_recall_curve, auc

vocab_size = CHAR_SMI_SET_LEN
embedding_size = 110
hidden_dim = 128
num_class = 2
device = torch.device("cuda:0")

class LSTM_Attention(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,n_layers):
        super(LSTM_Attention,self).__init__()

        self.W_Q = nn.Linear(hidden_dim,hidden_dim,bias =False)
        self.W_K = nn.Linear(hidden_dim,hidden_dim,bias =False)
        self.W_V = nn.Linear(hidden_dim,hidden_dim,bias =False)

        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)

        self.rnn = nn.LSTM(input_size = embedding_dim,hidden_size = hidden_dim,num_layers = n_layers,batch_first=True)

        self.fc = nn.Linear(hidden_dim,1)

        self.dropout = nn.Dropout(0.5)

        self.layernorm = nn.LayerNorm(hidden_dim)


    def attention(self,Q,K,V):
        d_k = K.size(-1)
        scores = torch.matmul(Q,K.transpose(1,2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores,dim=-1)
        context = torch.matmul(alpha_n,V)
        output = context.sum(1)
        
        return output,alpha_n


    def forward(self, input_ids, input_lengths):
        embedding = self.dropout(self.embedding(input_ids))    #embedding.shape = [batch_size,smi_len,embedding_dim = 100]
        embedding_data = pack_padded_sequence(input=embedding ,lengths=input_lengths, batch_first=True, enforce_sorted=False)
        output,(h_n,c) = self.rnn(embedding_data)           #out.shape = [batch_size,smi_len,hidden_dim=128]
        total_length = embedding.size(1)
        out_pad, out_len = pad_packed_sequence(sequence=output, batch_first=True,total_length=total_length,padding_value=0)

        Q = self.W_Q(out_pad)        # [batch_size,smi_len,hidden_dim]
        K = self.W_K(out_pad)
        V = self.W_V(out_pad)

        attn_output,alpha_n = self.attention(Q, K, V)
        out_ln = self.layernorm(attn_output)

        out = self.dropout(self.fc(out_ln))   #out.shape = [batch_size,num_class]
        return out


def test(flag, model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx,  (labels, input_ids, input_lengths) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            
            y = torch.tensor(labels).to(device)
            input_ids = torch.tensor(input_ids).to(device)
            input_lengths = input_lengths

            y_hat = model(input_ids, input_lengths)
            y_hat_temp = y_hat
            test_loss += loss_function(y_hat_temp.view(-1), y.view(-1).float()).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    b = torch.sigmoid(torch.tensor(outputs))
    pre_label = []
    for i in b:
        if i<=0.6187:
            pre_label.append(0)
        else:
            pre_label.append(1)

    outputs = np.array(pre_label)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'ACC': metrics.get_ACC(targets, outputs),
        'Precision': metrics.get_Precision(targets, outputs),
        'Recall': metrics.get_Recall(targets, outputs),
        'get_F1': metrics.get_F1(targets, outputs),
        'AUROC': metrics.get_ROC(targets, np.array(b)),
    }

    return evaluation
