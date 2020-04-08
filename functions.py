from scipy.io import wavfile
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import librosa
import numpy as np
import pandas as pd
import librosa.display
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import seaborn as sns
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.signal import butter,filtfilt,lfilter
from scipy.fftpack import fft,fftfreq,ifft
from spleeter.separator import Separator
from spleeter.audio.adapter import get_default_audio_adapter
from os import listdir
from os.path import isfile, join

separator = Separator('spleeter:2stems')


def extractFeaturesAllFilesnf(atype):
    for f1 in listdir("sources/"+atype+"/"):
        i=0
        for f2 in listdir("sources/"+atype+"/"+f1+"/"):
            if (f2==".gitignore"):
                continue
            audio=librosa.load("sources/"+atype+"/"+f1+"/"+f2,sr=44100)
            dur=len(audio[0])/audio[1]
            data=extractFeatures(audio[0],dur,int(audio[1]))
            data.to_csv("nofilter"+"/"+atype+"/data/"+f1+"/"+"audio"+str(i+1)+".csv",index=False)
            i=i+1    


            
    
def extractFeaturesAllFiles(filt,atype):
    for f1 in listdir(filt+"/"+atype+"/sources/"):
        i=0
        for f2 in listdir(filt+"/"+atype+"/sources/"+f1+"/"):     
            if (f2==".gitignore"):
                continue
            if (filt=="neural"):
                audio=librosa.load(filt+"/"+atype+"/sources/"+f1+"/"+f2+"/vocals.wav",sr=44100)
            else:
                if (filt=="podcast22k"):
                    audio=librosa.load(filt+"/"+atype+"/sources/"+f1+"/"+f2,sr=44100/2)
                elif (filt=="podcast11k"):
                    audio=librosa.load(filt+"/"+atype+"/sources/"+f1+"/"+f2,sr=44100/4)
                else:
                    audio=librosa.load(filt+"/"+atype+"/sources/"+f1+"/"+f2,sr=44100)
            dur=len(audio[0])/audio[1]
            data=extractFeatures(audio[0],dur,int(audio[1]))
            data.to_csv(filt+"/"+atype+"/data/"+f1+"/"+"audio"+str(i+1)+".csv",index=False)
            i=i+1

            
def filtAllFiles(filt,atype,cut=1):
    for f1 in listdir("sources/"+atype+"/"):
        for f2 in listdir("sources/"+atype+"/"+f1):
            if (f2==".gitignore"):
                continue
            if (filt=="butter"):
                butt_filterfile("sources/"+atype+"/"+f1+"/"+f2,atype,cut)
            elif (filt=="neural"):
                separator.separate_to_file('sources/'+atype+'/'+f1+'/'+f2, 'neural/'+atype+'/sources/'+f1)
                    
def deleteOutliers(data):
    size=len(data)
    data=data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
    print("Reduced to:",len(data)/size)
    return data

def calculatePCA(data):
    pca = PCA(n_components=4)
    pca.fit(data)
    Y=pd.DataFrame(pca.transform(data))
    print(np.cumsum(pca.explained_variance_ratio_))
    return Y

def extractFeatures(audio, dur, samplerate):
    w=1
    data=pd.DataFrame()
    for  i in range(1, int(dur/w)):
        mfccs=pd.DataFrame(librosa.feature.mfcc(audio[(i-1)*w*samplerate:i*w*samplerate], sr=samplerate,n_mfcc=13).T)
        data=data.append(mfccs.median(),ignore_index=True)
    return data


def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff /(0.5*fs)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data,axis=0)
    return y

def butt_filterfile(source,atype,cut=1):
    if (atype=="music"):
        offset=source[14:]
    else:
        offset=source[16:]
    samplerate, data=wavfile.read(source)
    fs = int(samplerate/cut)
    cutoff = 3500   
    nyq = 0.5 * fs  
    order = 4 
    n = len(data[::cut])
    T=n/samplerate
    ndata=(data[::cut])/2**15
    dataF=fft(ndata)
    y = butter_lowpass_filter(dataF, cutoff, fs, order)   
    y=ifft(y).real
    if (cut==1):
        write("butter/"+atype+"/sources/"+offset, fs, y)
    elif (cut==2):
        write("butter/"+atype+"22k/sources/"+offset, fs, y)
    elif (cut==4):
        write("butter/"+atype+"11k/sources/"+offset, fs, y)
   # plt.plot(np.arange(14300, 14410), ndata[:,1][14300:14410], 'b-', label='data')
   # plt.plot(np.arange(14300, 14410), y[:,1][14300:14410], 'g-', linewidth=2, label='filtered data')
   # plt.xlabel('Time [sec]')
   # plt.grid()
   # plt.legend()
   # plt.show()
    
    
def readFiles(filt,atype,nperson):
    data=[]
    for f in listdir(filt+"/"+atype+"/data/"+nperson):
        tmp=deleteOutliers(pd.read_csv(filt+"/"+atype+"/data/"+nperson+"/"+f))
        data.append(calculatePCA(tmp))
    return data


def savePlots(data,filt,atype,nperson,dist):
    tmp=data.copy()
    for i in range(0,len(tmp)):
        clustering = AgglomerativeClustering(linkage="ward",n_clusters=None,distance_threshold=dist).fit(tmp[i])
        tmp[i]["C"]=clustering.labels_
        plt.figure(figsize=(12,8))
        sns.scatterplot(tmp[i][0],tmp[i][1],hue=tmp[i]["C"],palette='Set1',legend="full")
        plt.savefig(filt+"/"+atype+"/plots/"+nperson+"/audio"+str(i+1)+"_sp.png")
        plt.title("audio n."+str(i+1)+"|||youtube video/podcast|||"+"n. person:  "+nperson)
    plt.close('all')

    
def bestDistance(data1,data2,data3,data4,filt,atype,start,stop):
    scores=[]
    if (atype[:5]!="music"):
        n=10
    else:
        n=3
    for i in range (start,stop):
        e1=0
        e2=0
        e3=0
        e4=0
        for j in range (0,len(data1)):
            cl1 = AgglomerativeClustering(linkage="ward",n_clusters=None,distance_threshold=i).fit(data1[j])
            e1+=np.abs(np.max(cl1.labels_)+1-1)
            cl2 = AgglomerativeClustering(linkage="ward",n_clusters=None,distance_threshold=i).fit(data2[j])
            e2+=np.abs(np.max(cl2.labels_)+1-2)
            if (atype[:5]!="music"):
                cl3 = AgglomerativeClustering(linkage="ward",n_clusters=None,distance_threshold=i).fit(data3[j])
                e3+=np.abs(np.max(cl3.labels_)+1-3)
                cl4 = AgglomerativeClustering(linkage="ward",n_clusters=None,distance_threshold=i).fit(data4[j])
                e4+=np.abs(np.max(cl4.labels_)+1-4)
        scores.append(e1+e2+e3+e4)
    best=start+scores.index(np.min(scores))
    print("Best distance:",best)
    print("Errors:",np.min(scores))
    print("Accuracy:",100-np.min(scores)*100/(len(data1)*(n)),"%")
    plt.title("Best distance: "+str(best)+"   Accuracy: "+str(100-np.min(scores)*100/(len(data1)*(n)))+"%")
    plt.plot(np.linspace(start,stop,(stop-start)),scores)
    plt.savefig(filt+"/"+atype+"/plots/dist_plot.png")
    return best


def plotErrors(data1,data2,data3,data4,dist,filt,atype):
    e1=[]
    e2=[]
    e3=[]
    e4=[]
    for i in range (0,len(data1)):
        cl1 = AgglomerativeClustering(linkage="ward",n_clusters=None,distance_threshold=dist).fit(data1[i])
        e1.append(np.max(cl1.labels_)+1-1)
        cl2 = AgglomerativeClustering(linkage="ward",n_clusters=None,distance_threshold=dist).fit(data2[i])
        e2.append(np.max(cl2.labels_)+1-2)
        if (atype[:5]!="music"):
            cl3 = AgglomerativeClustering(linkage="ward",n_clusters=None,distance_threshold=dist).fit(data3[i])
            e3.append(np.max(cl3.labels_)+1-3)
            cl4 = AgglomerativeClustering(linkage="ward",n_clusters=None,distance_threshold=dist).fit(data4[i])
            e4.append(np.max(cl4.labels_)+1-4)

    plt.figure(figsize=(20,10))
    plt.subplot(2,4,1)
    plt.title("1 Person")
    plt.hist(e1,bins=np.arange(-3,5)-0.5,density=True,edgecolor='black', linewidth=1.2)
    plt.xlim(-3,3)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylim(0,1)
    plt.subplot(2,4,2)
    plt.title("2 Persons")
    plt.hist(e2,bins=np.arange(-3,5)-0.5,density=True,edgecolor='black', linewidth=1.2)
    plt.xlim(-3,3)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylim(0,1)
    if (atype[:5]!="music"):
        plt.subplot(2,4,3)
        plt.title("3 Persons")
        plt.hist(e3,bins=np.arange(-3,5)-0.5,density=True,edgecolor='black', linewidth=1.2)
        plt.xlim(-3,3)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.ylim(0,1)
        plt.subplot(2,4,4)
        plt.title("4 Persons")
        plt.hist(e4,bins=np.arange(-3,5)-0.5,density=True,edgecolor='black', linewidth=1.2)
        plt.xlim(-3,3)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.ylim(0,1)
        plt.subplot(2,4,5)
        plt.title("General error")
        plt.hist(e1+e2+e3+e4,bins=np.arange(-3,5)-0.5,density=True,edgecolor='black', linewidth=1.2)
        plt.xlim(-3,3)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.ylim(0,1)
    else:
        plt.subplot(2,4,3)
        plt.title("General error")
        plt.hist(e1+e2+e3+e4,bins=np.arange(-3,5)-0.5,density=True,edgecolor='black', linewidth=1.2)
        plt.xlim(-3,3)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.ylim(0,1)
    plt.savefig(filt+"/"+atype+"/plots/error_plot.png")