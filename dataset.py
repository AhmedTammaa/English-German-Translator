import os 
import pandas as pd
import yaml 
from tensorflow.keras.preprocessing.sequence import pad_sequences
class DataLoader:
    def __init__(self, config):
        self.config = config
        self.lang_src = config['language']['src']
        self.lang_dest = config['language']['dest']
    
    def loadData(self,datapath):
        
        with open(datapath
                  +self.config['dataset_path']
                  +self.lang_src
                  ,encoding='utf-8') as sentence:
            self.src = sentence.read().split('\n')
        with open(datapath
                  +self.config['dataset_path']
                  +self.lang_dest
                  ,encoding='utf-8') as sentence:
            self.dest = sentence.read().split('\n')
        print("Loaded dataset at directory:\n"+datapath+self.config['dataset_path']+self.lang_src)
        return self.src, self.dest
        
    def removeEndLine(self, data):
        for index, word in enumerate(data):
            data[index] = word[:-1]
        return data
    def addStartEndPad(self, data, mark_start='aaaa ',mark_end= ' oooo'):
  
        for index, word in enumerate(data):
            
            data[index] = mark_start + word + mark_end  
        return data
    def createDataFrameSrcDest(self, src,dest):
        df = pd.DataFrame({self.lang_src: src, self.lang_dest: dest}, columns = [self.lang_src,self.lang_dest])
        #df.to_csv(path+'/questions_easy.csv', index=False)
        return df
class DataPreprocessing:
    def __init(self,src,dest):
        self.src = src
        self.dest = dest
    
        