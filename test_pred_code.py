import random

import numpy as np

from loglinear import softmax
STUDENT={'name': 'Yonatan Zoarets',
         'ID': '207596818'}

def file(f_name):
    with open(f_name,"r",encoding="utf-8") as f:
     return f.read().splitlines()


def read_data(fname):
    data = []

    for line in file(fname):
        label, text = line.strip().lower().split("\t",1)
        data.append((label, text))

    return data


def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

TRAIN = [(l, text_to_bigrams(t)) for l,t in read_data("train")]
DEV   = [(l, text_to_bigrams(t)) for l,t in read_data("dev")]#untill here, all by you

from collections import Counter
fcEn = Counter()
fcEs = Counter()
fcDe = Counter()
fcFr = Counter()
fcIt = Counter()
fcNl = Counter() #counter for each language

en_loss,es_loss,de_loss,fr_loss,it_loss,nl_loss=[],[],[],[],[],[]

fces=fcDe,fcEn,fcEs,fcFr,fcIt,fcNl

L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
# feature strings (bigrams) to IDs
losses= de_loss,en_loss,es_loss,fr_loss,it_loss,nl_loss # doing all in same sequence like the L2I

for SET in DEV,TRAIN:
 for l,feats in SET :
   for id,fc,loss in zip(L2I,fces,losses):#taking it together
     if (l==id ):
        fc.update(feats) #updating the counter of the current language
        if SET==TRAIN: #start compute the loss after we have something about the bigrams distribution
         fcl=Counter() #new empty counter
         fcl.update(feats)
         current_data=[]
         total_data=[]
         for i in fcl: #in current line
             if i not in (' @', '..', ' ;', ' (', '(:', '“@', ': '): #removing what could be in every language
               current_data.append(fcl[i])
               total_data.append(fc[i])
         total_data=total_data/np.sum(total_data)
         current_data=current_data/np.sum(current_data) #the percents of each one
         loss.append(np.sum(np.subtract(total_data,current_data)**2)) #pow 2 for being positive


de_ave_loss=np.average(de_loss)
en_ave_loss=np.average(en_loss)
es_ave_loss=np.average(es_loss)
fr_ave_loss=np.average(fr_loss)
it_ave_loss=np.average(fr_loss)
nl_ave_loss=np.average(nl_loss)

grads= de_ave_loss, en_ave_loss, es_ave_loss, fr_ave_loss, it_ave_loss, nl_ave_loss #same sequence


#until here loss for protocol

# label strings to IDs

# 600 most common bigrams in the training set.
vocabEn = set([x for x, c in fcEn.most_common(600)])
vocabEs = set([x for x, c in fcEs.most_common(600)])
vocabDe = set([x for x, c in fcDe.most_common(600)])
vocabFr = set([x for x, c in fcFr.most_common(600)])
vocabIt = set([x for x, c in fcIt.most_common(600)])
vocabNl = set([x for x, c in fcNl.most_common(600)])#taking 600 most common reduces the loss



vocabs=vocabDe,vocabEn,vocabEs,vocabFr,vocabIt,vocabNl #taking the last update of the vocabs


def scanVocab(vocabs, fces, fc_current_line):
 result=1 #unachieveable loss
 for vocab,fc,id,gr in zip(vocabs, fces, L2I, grads):#pasing all the vocabs according to fc and id of langueage
    data_current_vocab,class_outcurrent_data = [],[]#always starting from empty. it much easier for softmax to use array and not counter
    for bigram in vocab :
       if bigram not in (' @','..',' ;', ' (','(:','“@',': '):#ignoring all the uneffecting bigrams for improve accurate. kind of derividev
           class_outcurrent_data.append(fc_current_line[bigram])#test.pred if bigram appears in current row
           data_current_vocab.append(fc[bigram])#test.pred the appearence in the langueage. the places of each bigram it syncronized with the other array

    if np.sum(class_outcurrent_data)!=0:# for not divide by 0
     class_outcurrent_data=(class_outcurrent_data/np.sum(class_outcurrent_data))
    data_current_vocab=data_current_vocab/np.sum(data_current_vocab) # here is the softmax of all the 600 most common bigrams-it always will be 600

#as I said before, I didnt add loss here cause it just ruined
    if np.sum(np.subtract(class_outcurrent_data,data_current_vocab)**2)-gr*0.01<result:# looking for the smallest loss -,power 2 like before
        result = np.sum(np.subtract(class_outcurrent_data,data_current_vocab)**2)-gr*0.01 #0.01 is the learning rate
        ID=id
 # print(ID) for the checking
 with open("test.pred", "a", encoding="utf-8") as tp:# I used append, cause I didnt succeed with write here, but i used later
     tp.write(ID)
     tp.write("\n") # row down as asked

F2I = {f:i for i,f in enumerate(list(sorted(vocabEn)))}

def test(fname):
    text_dev = [(l, text_to_bigrams(t)) for l, t in read_data(fname)]
    with open(fname,"r",encoding="utf-8") as fp:
     line = fp.readline()
     i=0 #to num the lines
     with open("test.pred", "w", encoding="utf-8") as tp:
         tp.write("")#here is to clear
     while line:
      fcLastLine = Counter()#restarting
      line = fp.readline()#readline
      dev=text_dev[i]#taking only the current line bigrams
      i+=1 #method to count the line number
      for feats in dev:
          fcLastLine.update(feats)#updating and puting to the method
      scanVocab(vocabs, fces, fcLastLine)

if __name__ == '__main__':
    test("test")