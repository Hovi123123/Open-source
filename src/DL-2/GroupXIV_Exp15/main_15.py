import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataset import MyDataset, CHAR_SMI_SET_LEN
from model import LSTM_Attention, test

def dataset_collate(batch):
    label = []
    smi = []
    lenth = []
    for i in batch:
        a,b,d = i
        label.append(a)
        smi.append(b)
        lenth.append(d)
    
    return (np.array(label,dtype=np.int32),
            np.array(smi,dtype=np.int32),
            np.array(lenth,dtype=np.int32))

if __name__ == '__main__':
    print(sys.argv)

    SHOW_PROCESS_BAR = True
    data_path = '../data_cleaned_enum.xlsx'

    seed = 32416

    path = Path(f'./logs/logs_{datetime.now().strftime("%Y%m%d%H%M%S")}_{seed}')
    device = torch.device("cuda:0")  # or torch.device('cpu')

    max_smi_len = 195

    embedding_dim = 100
    batch_size = 128
    n_epoch = 20
    interrupt = None
    save_best_epoch = 6

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = SummaryWriter(path)
    f_param = open(path / 'parameters.txt', 'w')
    print(f'==========================================')
    print(f'seed={seed}')
    f_param.write(f'device={device}\n' 
            f'seed={seed}\n'
            f'write to {path}\n')

    f_param.write(f'max_smi_len={max_smi_len}\n')


    assert 0<save_best_epoch<n_epoch

    model = LSTM_Attention(vocab_size=CHAR_SMI_SET_LEN,embedding_dim=embedding_dim,hidden_dim=128,n_layers=4)
    model = model.to(device)
    f_param.write('model: \n')
    f_param.write(str(model)+'\n')
    f_param.close()

    data_loaders = {phase_name:
                        DataLoader(MyDataset(phase_name, data_path, max_smi_len),
                                batch_size=batch_size,
                                pin_memory=True,
                                shuffle=False,
                                collate_fn=dataset_collate)
                    for phase_name in ['training', 'validation', 'test']}
    optimizer = optim.AdamW(model.parameters(), weight_decay=1e-2)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-4, epochs=n_epoch,
                                            steps_per_epoch=len(data_loaders['training']))

    loss_function = nn.BCEWithLogitsLoss(reduction='sum')

    start = datetime.now()

    best_epoch = -1
    best_val_F1 = 0
    for epoch in range(1, n_epoch + 1):
        tbar = tqdm(enumerate(data_loaders['training']), disable=not SHOW_PROCESS_BAR, total=len(data_loaders['training']))
        for idx, (labels, input_ids, input_lengths) in tbar:

            model.train()

            y = torch.tensor(labels).to(device)
            input_ids = torch.tensor(input_ids).to(device)
            input_lengths = input_lengths

            optimizer.zero_grad()
            output = model(input_ids, input_lengths)
            loss = loss_function(output.view(-1), y.view(-1).float())
            loss.backward()

            optimizer.step()
            scheduler.step()

            tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item() / len(y):.3f}')

        for _p in ['training', 'validation']:
            flag = 0
            performance = test(flag, model, data_loaders[_p], loss_function, device, False)
            for i in performance:
                writer.add_scalar(f'{_p} {i}', performance[i], global_step=epoch)
            if _p=='validation' and epoch>=save_best_epoch and performance['get_F1']>best_val_F1:
                best_val_F1 = performance['get_F1']
                best_epoch = epoch
                torch.save(model.state_dict(), path / 'best_model.pt')


    model.load_state_dict(torch.load(path / 'best_model.pt'))
    flag = 1
    with open(path / 'result.txt', 'w') as f:
        f.write(f'best model found at epoch NO.{best_epoch}\n')
        _p = 'test'
        performance = test(flag, model, data_loaders[_p], loss_function, device, SHOW_PROCESS_BAR)
        f.write(f'{_p}:\n')
        print(f'{_p}:')
        for k, v in performance.items():
            f.write(f'{k}: {v}\n')
            print(f'{k}: {v}\n')
        f.write('\n')
        print()

