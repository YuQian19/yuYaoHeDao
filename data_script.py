# coding=utf-8

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

img_list = []
for root, dirs, files in os.walk("E:\\余姚河道\\data\\test"):
    for file in files:
        img_list.append(os.path.join(root, file).replace('\\', '/'))




with open('E:\\余姚河道\\data\\test_data.txt', 'w') as f:
    for i in range(len(img_list)):
        strr = img_list[i].replace('E:/余姚河道/','./')
        flag = (strr.find('clear')==-1)
        label = 1
        if flag:
            label = 0
        f.write(strr+" "+str(label)+"\r")
print("123")
