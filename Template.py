import os
import torch
import numpy
from Tools import ProgressBar


def Template_Basic_Train_Method(
        model, train_loader, val_loader, save_path, cuda_flag=True, episode_number=100, learning_rate=1E-4):
    if os.path.exists(save_path): return
    os.makedirs(save_path)

    if cuda_flag: model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.zero_grad()
    pbar = ProgressBar(n_total=len(train_loader) * episode_number, desc='Training')
    for episode_index in range(episode_number):
        episode_loss = 0.0

        model.train()
        loss_file = open(os.path.join(save_path, '%04d-Loss.csv' % episode_index), 'w')
        for batch_index, [batch_data, batch_seq, batch_label] in enumerate(train_loader):
            if cuda_flag:
                batch_data, batch_seq, batch_label = batch_data.cuda(), batch_seq.cuda(), batch_label.cuda()
            loss = model(batch_data, batch_seq, batch_label)

            pbar(episode_index * len(train_loader) + batch_index, {'loss': loss.item()})
            loss.backward()
            optimizer.step()
            model.zero_grad()

            episode_loss += loss.item()
            loss_file.write(str(loss.item()) + '\n')
        print('\nEpisode', episode_index, 'Total Loss =', episode_loss)
        loss_file.close()

        torch.save({'epoch': episode_index, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   os.path.join(save_path, '%04d-Parameter.pkl' % episode_index))

        model.eval()
        for eval_part in ['Eval']:
            predict_file = open(os.path.join(save_path, '%04d-%s.csv' % (episode_index, eval_part)), 'w')
            loader = val_loader

            for batch_index, [batch_data, batch_seq, batch_label] in enumerate(loader):
                if cuda_flag:
                    batch_data, batch_seq = batch_data.cuda(), batch_seq.cuda()
                batch_label = batch_label.squeeze()
                predict = model(batch_data, batch_seq)
                predict = predict.detach().cpu().numpy()
                batch_label = batch_label.detach().cpu().numpy()
                for indexX in range(numpy.shape(predict)[0]):
                    predict_file.write(str(batch_label))
                    for indexY in range(numpy.shape(predict)[1]):
                        predict_file.write(',' + str(predict[indexX][indexY]))
                    predict_file.write('\n')
            predict_file.close()
