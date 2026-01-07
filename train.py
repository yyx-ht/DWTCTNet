import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import ImageFolder
from metrix import validate
from utils import DiceLoss, FocalLoss
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import os
from ht.model.unetfamily.unet import UNet
from ht.model.ours.wtdmcatunet import wtdmcatunet
from tqdm import tqdm
from ht.model.wtnet.ID_UNet import ID_UNet
from ht.model.dnanet import DNANet
from ht.model.hcfnet import HCFnet
from ht.model.dear import DEFER,Res_CBAM_Block



def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('-size', type=int, default=224, help='size of images')
    parser.add_argument('-epoch', default=200)
    parser.add_argument('-set_num_threads', default=4, help='number of cpu thread')
    parser.add_argument('-device', default='cuda')
    parser.add_argument('-numclass', default=2)
    parser.add_argument('-lr', default=0.001, help='Learning rate')
    parser.add_argument('-cfg', type=str, default=r'D:\PyCharm\Py_Projects\deeplabv3-plus-py',
                        help='path to config file')
    parser.add_argument('-root_dir', default=r"D:\PyCharm\ht\ht\CTdata")
    parser.add_argument('-model', type=str, default='DEFER')
    parser.add_argument('-MutliLoss', default=False)
    parser.add_argument('-DATASET', default=r'CT')
    parser.add_argument('-SoftMax', default=True)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    model = DEFER(block=Res_CBAM_Block, nb_filter=[16, 32, 64, 128, 256], block_nums=[2, 2, 2, 2]).cuda()

    # 设置训练环境
    torch.set_num_threads(args.set_num_threads)
    rootdir = args.root_dir

    dataloader = ImageFolder(rootdir, mode='train', size=args.size)
    verify_dataloader = ImageFolder(rootdir, mode='test', size=args.size)
    save_dir = os.path.join('Experimental_result_' + args.DATASET + '/' + args.model)
    os.makedirs(save_dir, exist_ok=True)
    Evaluation_dir = os.path.join(save_dir + '/Evaluation_everyEpoch.txt')

    with open(Evaluation_dir, 'w') as file:
        file.write('epoch      mean_acc      mean_iu       recall        precision     f1_score      loss\n')

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # 启用 num_workers 进行数据并行加载
    train_loader = DataLoader(dataloader, batch_size=4, shuffle=True, pin_memory=False, num_workers=4)
    verify_loader = DataLoader(verify_dataloader, batch_size=1, shuffle=False)

    # 实例化损失函数，使用激进的参数设置来解决小目标问题
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(2)
    focal_loss = FocalLoss(alpha=0.99, gamma=1.0)

    cur_itrs = 0
    best_score = 0
    model.train()

    # 开始训练
    for epoch in range(args.epoch):
        train_loss = 0.0
        # 学习率打印逻辑
        if epoch < 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
                print(param_group['lr'])
        else:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])

        # === 添加 tqdm 进度条 ===
        tqdm_loader = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1}/{args.epoch}',
            leave=True,
            dynamic_ncols=True
        )

        for i_batch, sample_batched in enumerate(tqdm_loader):
            cur_itrs += 1
            images_batch, target_labels = sample_batched[0], sample_batched[1]

            # 将数据移动到 CUDA
            if args.device == 'cuda':
                images_batch, target_labels = images_batch.cuda(), target_labels.cuda()

            m_label_out_ = model(images_batch)

            # SoftMax=True 时将标签转换为 One-Hot 形式，与 C=2 的输出匹配
            if args.SoftMax is True:
                target_labels = torch.cat([target_labels, target_labels], dim=1)
            # else: target_labels 保持 C=1

            # === 损失计算 (组合损失) ===
            if args.MutliLoss:
                # 多输出模型的 MultiLoss
                loss_dice1 = dice_loss(m_label_out_[1], target_labels, softmax=True)
                loss_dice0 = dice_loss(m_label_out_[0], target_labels, softmax=True)
                loss = 0.1 * loss_dice1 + 0.1 * loss_dice0
            else:
                # Single Loss 模式：L = Focal + 0.5 * Dice/BCE
                loss_focal = focal_loss(m_label_out_, target_labels)

                if args.SoftMax is True:
                    loss_other = dice_loss(m_label_out_, target_labels, softmax=True)
                else:
                    loss_other = bce_loss(m_label_out_, target_labels)

                # 最终组合损失：Focal Loss (权重 1.0) + 其他损失 (权重 0.5)
                loss = loss_focal + 0.5 * loss_other
                # === 损失计算结束 ===

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            train_loss += current_loss

            # 在 tqdm 进度条上实时显示当前损失
            tqdm_loader.set_postfix(batch_loss=f'{current_loss:.6f}')

        scheduler.step()
        train_loss = train_loss / len(train_loader)

        # 打印 Epoch 损失
        print(f'\nEpoch: {epoch}/{args.epoch} \tTraining Loss: {train_loss:.6f}')

        save_mode_path = os.path.join(f'Experimental_result_{args.DATASET}/{args.model}/{args.model}_epoch{epoch}.pt')
        torch.save(model.state_dict(), save_mode_path)

        print(f"validation... cur_itrs: {cur_itrs}")
        model.eval()
        # 验证
        mean_acc, mean_iu, recall, precision, f1_score = validate(n_classes=2, model=model, loader=verify_loader)
        print(
            f"acc: {mean_acc:.6f}, iou: {mean_iu:.6f}, recall: {recall:.6f}, precision: {precision:.6f}, f1_score: {f1_score:.6f}")

        with open(Evaluation_dir, 'a') as file:
            file.write(
                f"    {epoch}      {mean_acc:.6f}      {mean_iu:.6f}      {recall:.6f}      {precision:.6f}      {f1_score:.6f}      {train_loss:.6f}\n")

        if mean_iu > best_score:
            best_score = mean_iu
            save_best_mode_path = os.path.join(f'Experimental_result_{args.DATASET}/{args.model}/best.pt')
            torch.save(model.state_dict(), save_best_mode_path)


if __name__ == '__main__':
    main()