import  sys
import pandas as pd
import math
import numpy as np
import time
from numpy import linalg as LA
from random import randint

INPUT_FILE_LOCATION = "D:\Train\\Dataset\\census\\census-5-5tra.csv"
OUTPUT_FILE_LOCATION = "D:\Train\\Dataset\\census\\census-5-5tra_conventedE015.csv"

CLASS_COLUMN_NAME = "Class"

NumberofFireflies = 10 #number of fireflies
MaxGeneration = 10  #number of round

#Firefly Parameters
alpha = 0.5 # randomness 0-1
betamin = 0.2 # mininum value of beta
gamma = 1 # Abssorption conffient

def alpha_new (alpha, NGen):
    delta = 1-(10**(-4)/0.9)**(1/NGen)
    alpha = (1-delta)*alpha
    return alpha

def firefly_move (pdf,current):
    bestdist = 0.0
    bestpos = 0
    i=0
    for index,  row in pdf.iterrows():
        dist = CalculateDistance(current, pdf)
        if bestdist > dist:
            bestdist = dist
            bestpos = i
        i = i + 1
        return bestpos,bestdist



def ConvertCategoricalToNumericColumn(df):
    g = df.columns.to_series().groupby(df.dtypes).groups
    check = None
    for k, v in g.items():
        if k.name == "object":
            check = True
            for index in v:
                df[index] = df[index].astype('category')
    if check:
        cat_columns = df.select_dtypes(include=['category']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    return df

def SamplingData (df):
    size = len(df)
    error_rate = 0.1
    sample_size = int(size /(1 + (size * math.pow(error_rate,2))))
    sample_list = np.random.choice(xrange(0,size-1),sample_size,replace=False)

    return sample_size, sample_list

def CalculateDistance (point,cpdf):
    sumdist = 0
    for index, row in cpdf.iterrows():
        #print index, list(row)
      dist = LA.norm(point[:-1]-row[:-1])
#      print "distance :", dist
      sumdist += dist

    return  sumdist

def GetInstance(cpdf, n, alpha):
    size = len(cpdf)

    bestdist = sys.maxint
    bestpos = 0
    i = 0


    psize = size / n
    if psize == 0:
        if size <= 3:
            n = 1
            psize = size
        else:
            n = 2
            psize = size / 2
    remain = size - (psize * n)
    start = np.ones(n, dtype=int)
    end = np.ones(n, dtype=int)
    Lightn = np.ones(n)
    Lightn.fill(float("inf"))
    Positionn = np.ones(n,dtype=int)

    for i in range(0,n):
        if i > 0:
            start[i] = end[i-1]
        else:
            start[i] = 0
        if remain > 0:
            end[i] = start[i] + psize + 1
            remain -= 1
        else :
            end[i] = start[i] + psize
        Positionn[i] = start[i]
    for k in range (0,MaxGeneration):
        alpha = alpha_new(alpha,MaxGeneration)
        print "round # %d" % k
        for i in range (0,n):
            pdf = cpdf.iloc[start[i]:end[i],:]
            pos = int(Positionn[i]) - start [i]
            point = next(pdf[pos:pos+1].iterrows())[1]
            dist = CalculateDistance(point,cpdf)
            Lightn[i] = dist
            Positionn[i] = start[i]+pos
        #Ranking fireflies by their light intensity
        sortn = np.sort(Lightn)
        print sortn
        currentbest = sortn[0]


        index = np.where(Lightn == currentbest)
        if isinstance(index,(list,tuple,np.array)):
            index = int(index[0][0])
        currentpos = int(Positionn[index])
        currentrow = cpdf.iloc[currentpos:currentpos + 1,:]
        for i in range(0,n):
            pdf = cpdf.iloc[start[i]:end[i], :]
            Positionn[i],dist = firefly_move (pdf,currentrow)
            Positionn[i] = start[i] + Positionn[i]
        print "curent best : %d " %currentpos
        if  bestdist >currentbest :
            bestdist = currentbest
            bestpos = currentpos
        i = i+1

    print "bestpos = ",bestpos , "dist = " ,bestdist
    return bestpos

    dist = CalculateDistance(point,cpdf)


df = pd.read_csv(INPUT_FILE_LOCATION)
df = df.sort_values(CLASS_COLUMN_NAME, ascending = True)

cdf = df.copy(deep=True)

cdf = ConvertCategoricalToNumericColumn(cdf)

start_time = time.time()

print cdf

'''

cdf.to_csv(OUTPUT_FILE_LOCATION, encoding='utf-8', \
           index=False)
'''

n, sample_list = SamplingData(df)


print "size of sample : ",n

psize = len(df) / n
remain_size = len(df) - (psize * n)
start = 0
end = 0
sample_list = []
#remain_size = 0
#start =(psize * 1942) + remain_size
for i in range(0,n):
    print "partition # ", i
    if remain_size > 0:
        end = start + psize +1
        remain_size = remain_size - 1
    else:
        end = start + psize
    print "start : " , start , " end : ", end
    pdf = cdf.iloc[start:end,:]
    class_list = pdf[CLASS_COLUMN_NAME].unique().tolist()
    for c in class_list :
        cpdf = pdf[pdf[CLASS_COLUMN_NAME] == c]
        #print "class ", c, " has ", len(cpdf), " rows."
        num = GetInstance(cpdf,NumberofFireflies,alpha)
        sample_list.append(start + int(num))
    start = end

print sample_list



sample_df = df.iloc[sample_list,:]
print sample_df
finish_time = time.time()
process_time = finish_time - start_time
print "processing time (seconds) : ",process_time
sample_df.to_csv(OUTPUT_FILE_LOCATION, encoding='utf-8', \
           index=False)