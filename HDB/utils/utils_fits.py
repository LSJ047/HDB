import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, scaler, save_period,
                  save_dir, local_rank=0):
    total_loss = 0

    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        Msbs, Lsbs = batch

        with torch.no_grad():
            if cuda:
                Msbs = Msbs.cuda(local_rank)
                Lsbs = Lsbs.cuda(local_rank)

        optimizer.zero_grad()
        # ----------------------#
        #   前向传播
        # ----------------------#
        outputs = model_train(Msbs, Lsbs)
        # ----------------------#
        #   损失计算
        # ----------------------#
        log_likelihood = -torch.log(outputs)
        loss = log_likelihood.mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        Msbs, Lsbs = batch

        with torch.no_grad():
            if cuda:
                Msbs = Msbs.cuda(local_rank)
                Lsbs = Lsbs.cuda(local_rank)

            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(Msbs, Lsbs)
            # ----------------------#
            #   损失计算
            # ----------------------#
            log_likelihood = -torch.log(outputs)
            loss = log_likelihood.mean()
            # -------------------------------#

            val_loss += loss.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        # pbar.close()
        # print('Finish Validation')
        # loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        # eval_callback.on_epoch_end(epoch + 1, model_train)
        # print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        # print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        print(f'Epoch:{Epoch}')
        print(f'save_period:{save_period}')
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
