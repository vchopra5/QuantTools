import pandas as pd
import numpy as np
from collections import OrderedDict
import glob
from astropy.stats.funcs import median_absolute_deviation
import math
from sklearn.cluster import KMeans

#Idea taken from https://arxiv.org/pdf/1607.04883.pdf, which uses K-Means and risk adjusted returns to classify stocks

#TODO: find optimal # of clusters to prevent stocks from all being classified in the same cluster
#TODO: find optimal lookback to determine risk profiles - may not be worth it

def k_means(frames, num_clusters, d_points, save_loc):
    #Using scipy for now, will switch to custom for more accurate classification
    
    labels = []
    data = []
    for f in frames:
        frame = frames[f]
        d = frame[0]
        if len(d) > d_points:
            labels.append(f.replace('tsx/', '').replace('.csv', ''))
            d = d[len(d) - d_points:]
            
        d = np.array(d)
        
        data.append(d)
    
    
    new_arr = np.stack((data[0], data[1]), axis=0)

    for d in range(2, len(data), 2):
        data[d] = data[d].reshape(1, -1)
        new_arr = np.concatenate((new_arr, data[d]), axis=0)
    
    k_mean = KMeans(n_clusters=num_clusters, random_state=1)
    k_mean.fit(new_arr)
    
    dict_to_csv = OrderedDict()
    
    for l in range(len(k_mean.labels_)):
        if k_mean.labels_[l] not in dict_to_csv:
            dict_to_csv[k_mean.labels_[l]] = []
        dict_to_csv[k_mean.labels_[l]].append(labels[l])
    
    df = pd.DataFrame.from_dict(dict_to_csv, orient='index')
    df = df.transpose()
    df.to_csv(save_loc + 'classify.csv', index=False)
    


def normalize_data(frames, v):
    for f in frames:
        frame = frames[f]
        std = frame[1]
        u = frame[1]/v
        ret = frame[0]
        new_ret = []
    
        for r in ret:
            new_ret.append(r/(u*std))
        
        ret = new_ret
        
        frame[0] = ret
        
        frames[f] = frame
    
    return frames

def calculate_cross_sec(frames):
    stds = []
    for f in frames:
        frame = frames[f]
        stds.append(np.log(frame[1]))
        
    v = math.exp(np.median(stds) - 3*median_absolute_deviation(stds))
    
    return v
    
    
def calculate_ret(frames):
    
    ret_dict = OrderedDict()
    
    for f in frames:
        frame = frames[f]
        
        returns = []
        
        for j in range(1, len(frame.index.values)):
            if not np.isnan(frame.loc[j, 'Close']) and not np.isnan(frame.loc[j-1, 'Close']):
                returns.append((frame.loc[j, 'Close']/frame.loc[j-1, 'Close']) - 1)
            
        std = np.std(returns)
        
        if not np.isnan(std):
            ret_dict[f] = []
            ret_dict[f].append(returns)
            ret_dict[f].append(np.std(returns))
    
    return ret_dict

def run():
    frames = OrderedDict()
    
    files = glob.glob('tsx/*')
    
    for f in files:
        frames[f] = pd.read_csv(f)
        
    ret_std = calculate_ret(frames)
    v = calculate_cross_sec(ret_std)
    norm = normalize_data(ret_std, v)
    k_means(norm, 20, 100, '')
    
    

run()