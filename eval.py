import os
import numpy as np
import matplotlib.pyplot as plt

def correction(a,threhold):
    index = np.where(a==1)[0]
    for i in np.arange(len(index)-1):
        begin = index[i]
        end = index[i+1]
        if end-threhold<begin:
            a[begin:end]=1
    return a

threhold = 0.3

def detect(a,pred,threhold,corr=False):
    a=np.squeeze(a,axis=1)
    print(a.shape,pred.shape)
    item_index = np.where(a==1)[0]
    event = []
    non = []
    for i in np.arange(len(item_index)):
        if(i == 0):
            event.append(item_index[i])
        else:
            if(i != len(item_index)-1):
                if(item_index[i+1]-1 != item_index[i]):
                    event.append(item_index[i])
                    event.append(item_index[i+1])
            else:
                event.append(item_index[i])
    event = np.array(event)
    print(len(event) // 2)
    newevent = event.copy()

    if len(newevent) == 0:
        print("⚠️ Warning: Ground truth 中没有正样本，跳过评估")
        return
    for i in np.arange(int(len(newevent)//2)):
        if(i==0):
            non.append(0)
            non.append(newevent[i]-1)
        else:
            non.append(newevent[2*i-1]+1)
            non.append(newevent[2*i]-1)
    if(newevent[-1] != len(a)-1):
        if(newevent[-1] != len(a)-2):
            non.append(newevent[-1]+1)
            non.append(len(a)-1)
    FP = 0
    TN = 0
    for i in np.arange(int(len(non)//2)):
        start =non[2*i]
        ending =non[2*i+1]+1
        real_non = ending-start
        pred_non1 =sum(pred[start:ending]==0)
        iog_non1 = pred_non1 / real_non
        if(iog_non1>0.9):
            TN += 1
        else:
            FP += 1
    TP = 0
    FN = 0
    for i in np.arange(len(newevent)//2):
        begin = newevent[2*i]
        end = newevent[2*i+1]+1
        real_bubble=newevent[2*i+1]-newevent[2*i]+1
        if corr :
            pred_correction = correction(pred[begin:end],70)
            pred[begin:end] = pred_correction
        pred_bubble = sum(pred[begin:end])
        IoG_event = pred_bubble/real_bubble
        if(IoG_event>threhold):
            TP += 1
        else:
            FN += 1
    print('threshold',threhold)
    print('TP',TP,'FN',FN,'FP',FP,'TN',TN)

    R = TP/(TP+FN)
    P = TP/(TP+FP)
    f1 = (2*R*P)/(R+P)
    print("Recall",round(R,3))
    print("Precision",round(P,3))
    print("f1",round(f1,3))


if __name__ == '__main__':
    pred_inc = np.load(r"E:\等离子泡\交叉计算\test\inc_2003new_y_18.npy")
    pred_10s = np.load(r"E:\等离子泡\原始公式对比_数据集\dayandnight\2003night\2003_night_10s_formula.npy")
    a = np.load(r"E:\等离子泡\原始公式对比_数据集\dayandnight\2003night\2003_night_y_label.npy")
    thres = 0.5
    detect(a,pred_runet,thres,True)
    detect(a,pred_unet,thres,True)
    detect(a,pred_inc,thres,True)
    detect(a,pred_10s,thres,True)
