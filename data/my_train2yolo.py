import os.path
import cv2
import numpy as np
import torch
import torch.utils.data as data

class WeldingDetection(data.Dataset):
    # E:\TEMP\output\new\distrubute\dataset\images\1.txt
    def __init__(self,txt_path, preproc=None):
        self.preproc = preproc
        self.image_path = []
        self.word = []
        f = open(txt_path,'r')
        lines = f.readlines()
        for line in lines:
            path = txt_path[:-len('1.txt')] + line
            path = path.rstrip()
            label__path = txt_path[:txt_path.index("images")] + 'labels/' + line.replace('.jpg','').replace('\n','') + '.txt'
            f_label = open(label__path,'r')
            l_lines = f_label.readlines()
            count = 0
            temp_word = []
            for l_line in l_lines:
                l_line = l_line.rstrip()
                if 1<count<5:
                    inline = l_line.split(' ')
                    label = [float(x) for x in inline]
                    temp_word.append(label)
                count += 1
            x_min = min(temp_word[:][:][0])
            y_min = min(temp_word[:][:][1])
            w = max(temp_word[:][:][0]) - min(temp_word[:][:][0])
            h = max(temp_word[:][:][1]) - min(temp_word[:][:][1])
            self.word.append([x_min, y_min, w, h, temp_word[0], temp_word[1], temp_word[2]])
            self.image_path.append(path)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        img = cv2.imread(self.image_path[item])
        height, width, _ = img.shape
        labels = self.word[item]
        return torch.from_numpy(img),labels


img_path_txt = "E:/TEMP/output/new/distrubute/dataset/images/1.txt"
aa = WeldingDetection(img_path_txt)
for i in range(aa.__len__()):
    save_path = os.path.join(img_path_txt[0:img_path_txt.index('images')],'train/')
    save_label_path =aa.image_path[i].replace('.jpg', '.txt')
    str_label = "0  "
    with open(save_label_path,'w') as f:
        str_label += str(aa.word[i])
        str_label = str_label.replace('[','').replace(']','').replace(',',' ')
        f.write(str_label)

