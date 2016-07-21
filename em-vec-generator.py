#!/usr/bin/env python

from  polyglot2 import Polyglot
from sets import Set
import numpy as np
from sets import Set

model_path="saved.model"
data_path="trainParsed.txt"
nn_input_path="input"

taglist=['PRP$', 'VBG', 'VBD', 'VBN', 'VBP', 'WDT', 'JJ', 'WP', 'VBZ', 'DT', 'RP', 'NN', 'FW', 'POS', 'TO', 'PRP', 'RB', 'NNS', 'NNP', 'VB', 'WRB', 'CC', 'PDT', 'RBS', 'RBR', 'CD', 'EX', 'IN', 'WP$', 'MD', 'NNPS', 'JJS', 'JJR', 'SYM', 'UH']
tagset=Set()

def read_file(path):
    sentence=[]
    with open(path,"r") as r:
        for x in r:
            sentence.append(x.strip().split())
    return sentence

def load_model(path):
    model = Polyglot.load_word2vec_format(model_path)
    return model

def tag_hot_vec(val):
    # print "Tag----------> ",val
    tag_len=len(taglist)
    y=[0]*tag_len
    for i in range(len(taglist)):
        if taglist[i] == val:
            y[i]=1
    return y

def make_sequence(model,sentence,X,Y,line,window=1):
    end_pad=model["</PAD>"]
    window_mid=window/2

    only_sen=[]
    only_tag=[]


    for i in range(len(sentence)):
        if i%2 == 0:
            only_sen.append(sentence[i])
        else:
            only_tag.append(sentence[i])
            tagset.add(sentence[i])
            # if sentence[i] =="#" or sentence[i] =="." or sentence[i] ==":" or sentence[i] ==",":
                # print "Vul Tag: ",sentence[i], line,i

    sen_len=len(only_sen)
    for i in range(sen_len-window_mid):
        seq_x=[]
        tag=""
        for j in range(window):
            if i+j<sen_len:
                word=only_sen[i+j]
                seq_x.append(model[word])
            else:
                seq_x.append(end_pad)

            if j == window_mid:
                tag = only_tag[i+j]
            pass
        seq_x=np.reshape(seq_x,-1)
        seq_y=tag_hot_vec(tag)

        X.append(seq_x)
        Y.append(seq_y)
        pass

def generate_em_vec(model,path):
    sentence=read_file(path)
    X=[]
    Y=[]
    cnt=1 # it's used for debug
    for x in sentence:
       make_sequence(model,x,X,Y,cnt,1)
       cnt=cnt+1

    X_arr=np.asarray(X).astype(np.float32)
    Y_arr=np.asarray(Y).astype(np.float32)

    return X_arr,Y_arr

def save_em_vector(X,Y,path):
    p1=path+"/testX.npy"
    p2=path+"/testY.npy"
    np.save(p1,X)
    np.save(p2,Y)

def load_npy(path):
    p1=path+"/testX.npy"
    p2=path+"/testY.npy"
    X=np.load(p1)
    Y=np.load(p2)
    return X,Y

def main():
    model=load_model(model_path)
    X,Y=generate_em_vec(model,data_path)
    print X.shape
    print Y.shape

    save_em_vector(X,Y,nn_input_path)
    X,Y=load_npy(nn_input_path)

    print "Load-X-shape: ",X.shape
    print "Load-Y-shape: ",Y.shape

if __name__=="__main__":
    main()
