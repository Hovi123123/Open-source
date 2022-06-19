import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from dataset import MyDataset
from model import MyModel, test


def dataset_collate(batch):
    label = []
    dp = []
    maccs = []
    ecfp4 = []
    for i in batch:
        a,e,f,g = i
        label.append(a)
        dp.append(e)
        maccs.append(f)
        ecfp4.append(g)
    
    return (np.array(label,dtype=np.int32),
            np.array(dp),
            np.array(maccs),
            np.array(ecfp4))


if __name__ == '__main__':
    print(sys.argv)
    SHOW_PROCESS_BAR = True
    data_path = '../data_cleaned.xlsx'
    seed_list = [32416,31764,31861,32342,32486,32249,32313,31691,
                 32289,32538,32487,31673,32140,31632,31732,31607,
                 31786,31687,32397,31948,31924,32543,32479,31956,
                 31690,31677,32200,32168,32230,31692]

    for i in seed_list:

        seed = i
        path = Path(f'./logs/logs_{datetime.now().strftime("%Y%m%d%H%M%S")}_{seed}')
        device = torch.device("cuda:0")  # or torch.device('cpu')

        dp_len = 128
        maccs_len = 128
        ecfp4_len = 128
        num_out_dim = 128

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

        assert 0<save_best_epoch<n_epoch

        model = MyModel(hidden_dim=128, dp_len=dp_len, maccs_len=maccs_len, ecfp4_len=ecfp4_len, num_out_dim=num_out_dim)
        model = model.to(device)
        f_param.write('model: \n')
        f_param.write(str(model)+'\n')
        f_param.close()

        data_loaders = {phase_name:
                            DataLoader(MyDataset(phase_name, data_path, dp_len, maccs_len, ecfp4_len),
                                    batch_size=batch_size,
                                    pin_memory=True,
                                    shuffle=True,
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
            for idx, (labels, DP, MACCS, ECFP4) in tbar:
                model.train()

                y = torch.tensor(labels).to(device)
                DP = torch.tensor(DP).to(device)
                MACCS = torch.tensor(MACCS).to(device)
                ECFP4 = torch.tensor(ECFP4).to(device)

                optimizer.zero_grad()
                output = model(DP, MACCS, ECFP4)
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
                if _p=='validation' and epoch>=save_best_epoch and performance['get_F1'] > best_val_F1:
                    best_val_F1 = performance['get_F1']
                    best_epoch = epoch
                    torch.save(model.state_dict(), path / 'best_model.pt')


        model.load_state_dict(torch.load(path / 'best_model.pt'))
        flag = 1
        with open(path / 'result.txt', 'w') as f:
            f.write(f'best model found at epoch NO.{best_epoch}\n')
            for _p in ['training', 'validation', 'test',]:
                performance = test(flag, model, data_loaders[_p], loss_function, device, SHOW_PROCESS_BAR)
                f.write(f'{_p}:\n')
                print(f'{_p}:')
                for k, v in performance.items():
                    f.write(f'{k}: {v}\n')
                    print(f'{k}: {v}\n')
                f.write('\n')
                print()


