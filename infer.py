import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from conf import settings
from utils import get_network, get_test_dataloader
import shutil
from models.resnet_pre import base_resnet
from p_r import p_r
from glob import glob
import numpy as np
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-source', type=str, required=True, help='the path to img file')
    args = parser.parse_args()

    net = base_resnet()
    net = net.cuda()
    net.load_state_dict(torch.load(args.weights))
    net.eval()

    transform_test = transforms.Compose([
        transforms.Resize((756, 1008), interpolation=2),
        transforms.ToTensor(),
    ])

    image_files = glob(args.source + '/*.*')
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    txt_file = os.path.join(result_dir, 'result.txt')
    txt_f = open(txt_file, 'w')

    with torch.no_grad():
        for image_file in sorted(image_files):
            image = Image.open(image_file).convert('RGB')
            image = transform_test(image).unsqueeze(0)
            image = image.cuda()
            output = net(image)
            _, pred = output.topk(1, 1, largest=True, sorted=True)
            pred = pred.item()
            txt_f.write(image_file + '   :' + str(pred) + '\n')

            # Image.fromarray(image_framed).save(output_file)
            # print("Mission complete, it took {:.3f}s".format(time.time() - t))
            # print("\nRecognition Result:\n")
            # for key in result:
            #     print(result[key][1])
            #     txt_f.write(result[key][1]+'\n')

    txt_f.close()
