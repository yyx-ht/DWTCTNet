import numpy as np
import torch
import os
import cv2 as cv
from torch.utils.data import DataLoader
import torch.nn.functional as F
from metrix import validate
from dataset import ImageFolder
import numpy
import argparse
from PIL import Image
from ht.model.unetfamily.unet import UNet
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')            #--size 意思是配置中输入
    parser.add_argument('-size', type=int,default=2048 ,help='size of images')
    parser.add_argument('-epoch', default=50)
    parser.add_argument('-set_num_threads', default=4,help='number of cpu thread')
    parser.add_argument('-device', default='cuda')
    parser.add_argument('-numclass', default=2)
    parser.add_argument('-cfg', type=str,default= r'D:\tony\python\unet_family\res+swin\config.py', help='path to config file' )
    parser.add_argument('-root_dir', default=r"D:\PyCharm\ht\ht\DRdata\data")
    parser.add_argument('-model', type=str, default='UNet')
    parser.add_argument('-DATASET', default=r'DR')
    args = parser.parse_args()
    return args

def main():
    torch.set_num_threads(4)
    args = parse_args()

    model =UNet().cuda()

    # save_best_mode_path = os.path.join('Experimental_result_' + args.DATASET + '/' + args.model + '/best' + '.pt')
    save_best_mode_path = r'D:\PyCharm\ht\Experimental_result_DR\UNet\best.pt'
    model.load_state_dict(torch.load(save_best_mode_path))

    test_dir = args.root_dir

    test_dataloader = ImageFolder(test_dir,mode='test1',size=args.size)
    test_loader = DataLoader(
        test_dataloader, batch_size=1, shuffle=False)

    model.eval()
    mean_acc, mean_iu, recall,precision, f1_score = validate(n_classes=2, model=model, loader=test_loader)

    #model.train()
    print("acc:{:.6f},iou:{:.6f}, recall:{:.6f},precision:{:.6f},f1_score:{:.6f}".format(mean_acc, mean_iu, recall,
                                                                                         precision, f1_score))

    file_list = os.listdir(test_dir+r'\test1\labels')
    folder_path = r'D:\PyCharm\ht\ht\DRdata\data\predictbig\UNet'

    test_img_dir = os.path.join(folder_path, r'testimg')
    test_lab_dir = os.path.join(folder_path, r'testlab')
    label_dir = os.path.join(folder_path, r'labels')
    image_dir = os.path.join(folder_path, r'images')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(test_img_dir):
        os.makedirs(test_img_dir)
    if not os.path.exists(test_lab_dir):
        os.makedirs(test_lab_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    model.eval()
    i = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            i = i + 1
            images_batch, target_labels = \
                sample_batched[0], sample_batched[1]
            images_batch, target_labels = images_batch.cuda(), target_labels.cuda()
            x = model(images_batch)

            probs = F.softmax(x, dim=1)
            m_label_out = probs.max(1)[1].cpu().numpy()  # 不应该直接取第2层，而是判断两层中哪个较大，取较大的层，并且每层确定一个类别
            m_label_out = m_label_out.squeeze(0)
            m_label_out = (m_label_out).astype(numpy.uint8) * 255
            m_label_out_label = m_label_out

            #原图与标签图结合
            m_label_out = Image.fromarray(m_label_out)
            m_label_out = m_label_out.convert("RGB")


            for x in range(m_label_out.width):
                for y in range(m_label_out.height):
                    gray_value = m_label_out.getpixel((x,y))[0]
                    if gray_value > 0 :
                        m_label_out.putpixel((x,y),(255,0,0))
                    else:
                        m_label_out.putpixel((x, y), (gray_value, gray_value, gray_value))

            images = (images_batch.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(numpy.uint8)
            images = Image.fromarray(images)
            m_label_out = Image.blend(images,m_label_out,0.5)
            m_label_out = np.array(m_label_out)
            images = np.array(images)

            save_path = os.path.join(test_img_dir,file_list[i - 1])
            cv.imwrite(save_path, m_label_out)
            target_labels = target_labels.squeeze(0).squeeze(0).cpu().numpy().astype(numpy.uint8) * 255
            save_path = os.path.join(label_dir,file_list[i - 1])
            cv.imwrite(save_path, target_labels)
            #images = (images_batch.squeeze(0).cpu().numpy().transpose(1,2,0)* 255).astype(numpy.uint8)
            save_path = os.path.join(image_dir,file_list[i - 1])
            cv.imwrite(save_path, images)
            save_path = os.path.join(test_lab_dir, file_list[i - 1])
            cv.imwrite(save_path, m_label_out_label)

if __name__ == '__main__':
    main()