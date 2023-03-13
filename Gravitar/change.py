# -*- coding: utf-8 -*-
# @Time    : 2021/2/27 18:54
# @Author  : youngleesin
# @FileName: change.py
# @Software: PyCharm
import numpy as np
with open('D:\迅雷下载\Incredibles.2.2018.1080p.BluRay.H264.AAC-RARBG\Subs\[zmk.pw]Incredibles.2.2018.1080p.WEB-DL.DD5.1.H264-FGT.srt', 'r', encoding='UTF-8') as f:
    with open('D:/迅雷下载/Incredibles.2.2018.1080p.BluRay.H264.AAC-RARBG/Subs/new.srt', 'w', encoding='UTF-8') as w:
        for line in f.readlines():
            if line[0] == '0':
                print(int(line[7]) + 1)
                print(int(line[24]) + 1)
                w.write(line[:7]+str(int(line[7]) + 1)+line[8:24]+str(int(line[24]) + 1)+line[25:])
            else:
                w.write(line)
                # pass
# sl = "1234"
# sl[0] = '2'
# print(sl)