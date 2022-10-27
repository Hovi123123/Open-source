import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import metrics
import torch.nn.functional as F
import math

embedding_size = 110
hidden_dim = 128
num_class = 2
device = torch.device("cuda:0")

class MyModel(nn.Module):
    def __init__(self,hidden_dim, dp_len, maccs_len, ecfp4_len, num_out_dim):
        super(MyModel,self).__init__()

        self.W_Q2 = nn.Linear(num_out_dim,num_out_dim,bias =False)
        self.W_K2 = nn.Linear(num_out_dim,num_out_dim,bias =False)
        self.W_V2 = nn.Linear(num_out_dim,num_out_dim,bias =False)

        self.fc = nn.Linear(num_out_dim,1)
        #dropout
        self.dropout = nn.Dropout(0.6)
        #layerNorm
        self.layernorm = nn.LayerNorm(num_out_dim)

        self.dpfc = nn.Linear(dp_len, num_out_dim)
        self.maccsfc = nn.Linear(maccs_len, num_out_dim)
        self.ecfp4fc = nn.Linear(ecfp4_len, num_out_dim)


    def attention(self,Q,K,V):
        d_k = K.size(-1)

        scores = torch.matmul(Q,K.transpose(1,2)) / math.sqrt(d_k)

        alpha_n = F.softmax(scores,dim=-1)
        context = torch.matmul(alpha_n,V)

        output = context.sum(1)
        
        return output,alpha_n


    def forward(self, DP, MACCS, ECFP4):

        out_dp = torch.relu(self.dropout(self.dpfc(torch.as_tensor(DP,dtype=torch.float32))))
        out_maccs = torch.relu(self.dropout(self.maccsfc(torch.as_tensor(MACCS,dtype=torch.float32))))
        out_ecfp4 = torch.relu(self.dropout(self.ecfp4fc(torch.as_tensor(ECFP4,dtype=torch.float32))))

        com_vector = torch.stack((out_dp,out_maccs,out_ecfp4),1)

        Q2 = self.dropout(self.W_Q2(com_vector))
        K2 = self.dropout(self.W_K2(com_vector))
        V2 = self.dropout(self.W_V2(com_vector))

        attn_output,alpha_n = self.attention(Q2, K2, V2)
        out_ln = self.layernorm(attn_output)

        out = self.dropout(self.fc(out_ln))
        return out


def test(flag, model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx,  (labels, DP, MACCS, ECFP4) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            
            y = torch.tensor(labels).to(device)
            DP = torch.tensor(DP).to(device)
            MACCS = torch.tensor(MACCS).to(device)
            ECFP4 = torch.tensor(ECFP4).to(device)

            y_hat = model(DP, MACCS, ECFP4)
            y_hat_temp = y_hat
            test_loss += loss_function(y_hat_temp.view(-1), y.view(-1).float()).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    b = torch.sigmoid(torch.tensor(outputs))
    pre_label = []
    for i in b:
        if i <= 0.35:
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
