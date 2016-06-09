# -*- coding: utf-8 -*- 

from TensorBuild import loadEmbeddingFile, loadEmbeddingFile_extend
from TensorBuild import cosVector, cosine_simi,dot_prod
from scipy.stats import spearmanr
import numpy
from scipy import linalg, mat, dot
from scipy.spatial.distance import cdist
from operator import itemgetter
from scipy import stats
from numpy import linalg as LA
import sys
from scipy.spatial import distance
from scipy import spatial

def load_sebastian():
    readFile=open('/mounts/data/proj/fadebac/sa/data_manipulated/corpora/word_similarity/de/gur65/gur65')
    wordPair2label={}
    line_count=0
    for line in readFile:
        if line_count==0:
            line_count+=1
            continue
        else:
            tokens=line.strip().split('\t')
            wordPair2label[(tokens[0], tokens[1])]=float(tokens[2])
    print 'Sebastian loaded over.'
    return wordPair2label    
def Sebastian(embeddingDict):
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/conc_0_1_2_3_4.txt')
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Tensor/result/tensor_ppmi/word2embedding_giga_wiki_20141201_300d.txt')
    #embeddingDict=loadEmbeddingFile(filename)
    #emb_matrix=mat(embeddingDict.values())
    #norm_vector=LA.norm(emb_matrix, ord=2,axis=0)
    #emb_normalized=numpy.divide(emb_matrix,norm_vector)
    #key_list=embeddingDict.keys()
    wordPairs2label=load_sebastian()
    wordPair2simi={}
    labels=[]
    predicts=[]
    unknown=0
    for (word1, word2), label in wordPairs2label.iteritems():
        #labels.append(label)
        embedding1=embeddingDict.get(word1)
        embedding2=embeddingDict.get(word2)
        if embedding1 is not None and embedding2 is not None:            
            #predict=1.0/(1.0+distance.euclidean(embedding1, embedding2))
            predict=1 - spatial.distance.cosine(embedding1, embedding2)
            wordPair2simi[(word1, word2)]=predict
            predicts.append(predict)
            labels.append(label)   
        else:
            unknown+=1
    sp,p_value=spearmanr(labels, predicts)
    print 'Spearmanr is: '+str(sp)+', unknown pairs: '+str(unknown)  
def load_SimLex999():
    readFile=open('/mounts/data/proj/wenpeng/Dataset/SimLex-999/SimLex-999.txt')
    wordPair2label={}
    line_count=0
    for line in readFile:
        if line_count==0:
            line_count+=1
            continue
        else:
            tokens=line.strip().split('\t')
            wordPair2label[(tokens[0], tokens[1])]=float(tokens[3])
    print 'SimLex loaded over.'
    return wordPair2label    

def Simlex999(embeddingDict):
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/conc_0_1_2_3_4.txt')
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Tensor/result/tensor_ppmi/word2embedding_giga_wiki_20141201_300d.txt')
    #embeddingDict=loadEmbeddingFile(filename)
    #emb_matrix=mat(embeddingDict.values())
    #norm_vector=LA.norm(emb_matrix, ord=2,axis=0)
    #emb_normalized=numpy.divide(emb_matrix,norm_vector)
    #key_list=embeddingDict.keys()
    wordPairs2label=load_SimLex999()
    wordPair2simi={}
    labels=[]
    predicts=[]
    unknown=0
    for (word1, word2), label in wordPairs2label.iteritems():
        #labels.append(label)
        embedding1=embeddingDict.get(word1)
        embedding2=embeddingDict.get(word2)
        if embedding1 is not None and embedding2 is not None:            
            #predict=1.0/(1.0+distance.euclidean(embedding1, embedding2))
            predict=1 - spatial.distance.cosine(embedding1, embedding2)
            wordPair2simi[(word1, word2)]=predict
            predicts.append(predict)
            labels.append(label)   
        else:
            unknown+=1
    sp,p_value=spearmanr(labels, predicts)
    print 'Spearmanr is: '+str(sp)+', unknown pairs: '+str(unknown)  
    return sp

def load_wordsimi353():
    readFile=open('/mounts/data/proj/wenpeng/Dataset/wordsimi353/combined.tab')
    wordPair2label={}
    line_count=0
    for line in readFile:
        if line_count==0:
            line_count+=1
            continue
        else:
            tokens=line.strip().split('\t')
            wordPair2label[(tokens[0], tokens[1])]=float(tokens[2])
    print 'wordsimi353 loaded over.'
    return wordPair2label

    

def wordsimi353(embeddingDict):
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/conc_0_1_2_3_4.txt')
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Tensor/result/tensor_ppmi/word2embedding_giga_wiki_20141201_300d.txt')
    #embeddingDict=loadEmbeddingFile(filename)
    #emb_matrix=mat(embeddingDict.values())
    #norm_vector=LA.norm(emb_matrix, ord=2,axis=0)
    #emb_normalized=numpy.divide(emb_matrix,norm_vector)
    #key_list=embeddingDict.keys()
    wordPairs2label=load_wordsimi353()
    wordPair2simi={}
    labels=[]
    predicts=[]
    unknown=0
    for (word1, word2), label in wordPairs2label.iteritems():
        #labels.append(label)
        embedding1=embeddingDict.get(word1,0)
        embedding2=embeddingDict.get(word2,0)
        if embedding1!=0 and embedding2!=0:            
#             predict=1.0/(1.0+distance.euclidean(embedding1, embedding2))
            predict=1 - spatial.distance.cosine(embedding1, embedding2)
            
            wordPair2simi[(word1, word2)]=predict
            predicts.append(predict)
            labels.append(label)   
        else:
            unknown+=1
    sp,p_value=spearmanr(labels, predicts)
    print 'Spearmanr is: '+str(sp)+', unknown pairs: '+str(unknown)     
    
    
    
def RBF_kernel(x, y):
    a = numpy.array(x)
    b = numpy.array(y)

    dist = numpy.linalg.norm(a-b)
    simi= numpy.exp(-dist/4)
    return simi    
    
def findTopTenNeighbors():
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Tensor/result/word2embedding_calculus.txt')
    embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Emb_Extend/ensamble_map2one.txt')
    for word in ['apple', 'microsoft', 'monday', 'Monday', 'like']:
        print 'Computing similarities for: '+ word
        embedding=embeddingDict[word]
        neighbor2simi={}
        count=0
        for neighbor in  embeddingDict.keys():
            count+=1
            if count%500000 ==0:
                print count
            if cmp(neighbor, word) != 0:
                neighbor2simi[neighbor]=cosVector(embedding, embeddingDict[neighbor])
        dict= sorted(neighbor2simi.iteritems(), key=lambda d:d[1],reverse = True)
        for i in range(10):
            print dict[i]


def load_MC30():
    readFile=open('/mounts/data/proj/wenpeng/Dataset/MC_30.txt')
    wordPair2label={}
    for line in readFile:
        tokens=line.strip().split('\t')
        wordPair2label[(tokens[0], tokens[1])]=float(tokens[2])
    print 'MC30 loaded over.'
    return wordPair2label

def load_MEN():
    readFile=open('/mounts/data/proj/wenpeng/Dataset/MEN/MEN_dataset_natural_form_full')
    wordPair2label={}
    for line in readFile:
        tokens=line.strip().split()
        wordPair2label[(tokens[0], tokens[1])]=float(tokens[2])
    print 'MEN loaded over.'
    return wordPair2label
def load_RG65():
    readFile=open('/mounts/data/proj/wenpeng/Dataset/RG_65.txt')
    wordPair2label={}
    for line in readFile:
        tokens=line.strip().split('\t')
        wordPair2label[(tokens[0], tokens[1])]=float(tokens[2])
    print 'RG65 loaded over.'
    return wordPair2label

def load_SCWS():
    readFile=open('/mounts/data/proj/wenpeng/Dataset/SCWS/ratings.txt')
    wordPair2label={}
    for line in readFile:
        tokens=line.strip().split('\t')
        wordPair2label[(tokens[1], tokens[3])]=float(tokens[7])
    print 'SCWS loaded over.'
    return wordPair2label


def load_RW():
    readFile=open('/mounts/data/proj/wenpeng/Dataset/rw/rw.txt')
    wordPair2label={}
    for line in readFile:
        tokens=line.strip().split('\t')
        wordPair2label[(tokens[0], tokens[1])]=float(tokens[2])
    print 'Rare words loaded over.'
    return wordPair2label

def MC30(embeddingDict):
    #embeddingDict=loadEmbeddingFile(filename)
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Tensor/result/tensor_ppmi/word2embedding_giga_wiki_20141201_300d.txt')
    #embeddingDict=loadEmbeddingFile(filename) #hlbl-embeddings-original.EMBEDDING_SIZE=100.txt')
    wordPairs2label=load_MC30()
    wordPair2simi={}
    labels=[]
    predicts=[]
    unknown=0
    for (word1, word2), label in wordPairs2label.iteritems():
        #labels.append(label)
        embedding1=embeddingDict.get(word1,0)
        embedding2=embeddingDict.get(word2,0)
        if embedding1!=0 and embedding2!=0:
#             predict=1.0/(1.0+distance.euclidean(embedding1, embedding2))
            predict=1 - spatial.distance.cosine(embedding1, embedding2)
            wordPair2simi[(word1, word2)]=predict
            predicts.append(predict)
            labels.append(label)   
        else:
            unknown+=1
    sp,p_value=spearmanr(labels, predicts)
    print 'Spearmanr is: '+str(sp)+', unknown pairs: '+str(unknown)      

def MEN(embeddingDict):
    #embeddingDict=loadEmbeddingFile(filename)
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Tensor/result/tensor_ppmi/word2embedding_giga_wiki_20141201_300d.txt')
    #embeddingDict=loadEmbeddingFile(filename) #hlbl-embeddings-original.EMBEDDING_SIZE=100.txt')
    wordPairs2label=load_MEN()
    wordPair2simi={}
    labels=[]
    predicts=[]
    unknown=0
    for (word1, word2), label in wordPairs2label.iteritems():
        #labels.append(label)
        embedding1=embeddingDict.get(word1,0)
        embedding2=embeddingDict.get(word2,0)
        if embedding1!=0 and embedding2!=0:
#             predict=1.0/(1.0+distance.euclidean(embedding1, embedding2))
            predict=1 - spatial.distance.cosine(embedding1, embedding2)
            wordPair2simi[(word1, word2)]=predict
            predicts.append(predict)
            labels.append(label)   
        else:
            unknown+=1
    sp,p_value=spearmanr(labels, predicts)
    print 'Spearmanr is: '+str(sp)+', unknown pairs: '+str(unknown)    
    
def RG65(embeddingDict):
    #embeddingDict=loadEmbeddingFile(filename)
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Tensor/result/tensor_ppmi/word2embedding_giga_wiki_20141201_300d.txt')
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Dataset/glove.6B.300d.txt') #hlbl-embeddings-original.EMBEDDING_SIZE=100.txt')
    wordPairs2label=load_RG65()
    wordPair2simi={}
    labels=[]
    predicts=[]
    unknown=0
    for (word1, word2), label in wordPairs2label.iteritems():
        #labels.append(label)
        embedding1=embeddingDict.get(word1,0)
        embedding2=embeddingDict.get(word2,0)
        if embedding1!=0 and embedding2!=0:
#             predict=1.0/(1.0+distance.euclidean(embedding1, embedding2))
            predict=1 - spatial.distance.cosine(embedding1, embedding2)
            wordPair2simi[(word1, word2)]=predict
            predicts.append(predict)
            labels.append(label)   
        else:
            unknown+=1
    sp,p_value=spearmanr(labels, predicts)
    print 'Spearmanr is: '+str(sp)+', unknown pairs: '+str(unknown)   

def SCWS(embeddingDict):
    #embeddingDict=loadEmbeddingFile(filename)
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Tensor/result/tensor_ppmi/word2embedding_giga_wiki_20141201_300d.txt')
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Dataset/glove.6B.300d.txt') #hlbl-embeddings-original.EMBEDDING_SIZE=100.txt')
    wordPairs2label=load_SCWS()
    wordPair2simi={}
    labels=[]
    predicts=[]
    unknown=0
    for (word1, word2), label in wordPairs2label.iteritems():
        #labels.append(label)
        embedding1=embeddingDict.get(word1,0)
        embedding2=embeddingDict.get(word2,0)
        if embedding1!=0 and embedding2!=0:
            predict=1.0/(1.0+distance.euclidean(embedding1, embedding2))
            wordPair2simi[(word1, word2)]=predict
            predicts.append(predict)
            labels.append(label)   
        else:
            unknown+=1
    sp,p_value=spearmanr(labels, predicts)
    print 'Spearmanr is: '+str(sp)+', unknown pairs: '+str(unknown)   

def RW(embeddingDict):
    #embeddingDict=loadEmbeddingFile(filename)
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Tensor/result/tensor_ppmi/word2embedding_giga_wiki_20141201_300d.txt')
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Dataset/glove.42B.300d.txt') #hlbl-embeddings-original.EMBEDDING_SIZE=100.txt')
    wordPairs2label=load_RW()
    wordPair2simi={}
    labels=[]
    predicts=[]
    unknown=0
    for (word1, word2), label in wordPairs2label.iteritems():
        #labels.append(label)
        embedding1=embeddingDict.get(word1,0)
        embedding2=embeddingDict.get(word2,0)
        if embedding1!=0 and embedding2!=0:
#             predict=1.0/(1.0+distance.euclidean(embedding1, embedding2))
            predict=1 - spatial.distance.cosine(embedding1, embedding2)
            wordPair2simi[(word1, word2)]=predict
            predicts.append(predict)
            labels.append(label)   
        else:
            unknown+=1
    sp,p_value=spearmanr(labels, predicts)
    print 'Spearmanr is: '+str(sp)+', unknown pairs: '+str(unknown)   

def load_word_analogy():
    seman_file=open('/mounts/data/proj/wenpeng/Dataset/word_analogy/semantic.txt', 'r')
    synt_file=open('/mounts/data/proj/wenpeng/Dataset/word_analogy/syntatic.txt', 'r')   
    seman_lists=[]
    for line in seman_file:
        row=line.strip().split()
        seman_lists.append(row)
    synt_lists=[]
    for line in synt_file:
        row=line.strip().split()
        synt_lists.append(row)
    seman_file.close()
    synt_file.close()
    print 'word analogy data loaded over.'
    return seman_lists, synt_lists


def search_word_analogy(lists, embeddingDict):
    rows=len(lists)
    fail=0
    unknown_rows=0
    row_count=0
    for row in range(rows):
        row_count+=1
        if row_count%200==0:
            print row_count, '....'
        emb_1=embeddingDict.get(lists[row][0], 0)
        emb_2=embeddingDict.get(lists[row][1], 0)
        emb_3=embeddingDict.get(lists[row][2], 0)
        emb_4=embeddingDict.get(lists[row][3], 0)
        if emb_1!=0 and emb_2!=0 and emb_3!=0 and emb_4!=0:#a valid row
            pred_emb=list(numpy.array(emb_2)-numpy.array(emb_1)+numpy.array(emb_3))
            max_simi=cosine_simi(pred_emb, emb_4)
            for key, emb in embeddingDict.iteritems():
                simi=cosine_simi(emb,pred_emb)
                if simi>max_simi:
                    fail+=1
                     #print ' known rows ', rows-unknown_rows, ' fail: ', fail 
                    break
        else:
            unknown_rows+=1
    print 'totally rows: ', rows, ' known rows ', rows-unknown_rows, ' fail: ', fail, ' acc: ', 1.0-fail*1.0/(rows-unknown_rows)
    return rows-unknown_rows, rows-unknown_rows-fail
                    
def search_word_analogy_new(lists, embeddingDict):
    rows=len(lists)
    unknown_rows=0
    
    
    word_list=embeddingDict.keys()
    emb_matrix=mat(embeddingDict.values())
    pred_matrix=[]
    gold_word_list=[]
    
    data_matrix=[]
    for row in range(rows):
        a=lists[row][0]
        b=lists[row][1]
        c=lists[row][2]
        d=lists[row][3]
        emb_1=embeddingDict.get(a)
        emb_2=embeddingDict.get(b)
        emb_3=embeddingDict.get(c)
        emb_4=embeddingDict.get(d)
        if emb_1 is None:
            emb_1=embeddingDict.get(lists[row][0].lower())
            a=lists[row][0].lower()
        if emb_2 is None:
            emb_2=embeddingDict.get(lists[row][1].lower())
            b=lists[row][1].lower()
        if emb_3 is None:
            emb_3=embeddingDict.get(lists[row][2].lower())
            c=lists[row][2].lower()
        if emb_4 is None:
            emb_4=embeddingDict.get(lists[row][3].lower())
            d=lists[row][3].lower()
        
        
        if emb_1 is not None and emb_2 is not None and emb_3 is not None and emb_4 is not None:#a valid row
            data_row=[a,b,c,d]
            data_matrix.append(data_row)
            pred_emb=list(numpy.array(emb_2)-numpy.array(emb_1)+numpy.array(emb_3))
            pred_matrix.append(pred_emb)
            gold_word_list.append(d)
        else:
            unknown_rows+=1
    pred_matrix=mat(pred_matrix)
    simi_matrix=1-cdist(pred_matrix,emb_matrix, 'cosine')
    #max_index_list=simi_matrix.argsort()[:,emb_matrix.shape[0]-1]
    max_index_matrix=simi_matrix.argsort()[:,-4:]
    pred_word_matrix=[]
    for row in range(max_index_matrix.shape[0]):
        pred_list=[word_list[i].lower() for i in max_index_matrix[row]] 
        pred_word_matrix.append(pred_list)
    match_count=0
    row_count=len(gold_word_list)
    for i in range(row_count):
        for j in [3,2,1,0]:
            if pred_word_matrix[i][j]==data_matrix[i][0].lower() or pred_word_matrix[i][j]==data_matrix[i][1].lower() or pred_word_matrix[i][j]==data_matrix[i][2].lower():
                continue
            elif pred_word_matrix[i][j]==data_matrix[i][3].lower():
                match_count+=1
            else:
                break
    print 'totally rows: ', rows, ' known rows ', row_count, ' unknown rows ',  unknown_rows, ' acc: ', match_count*1.0/row_count
    return row_count, match_count           


def search_word_analogy_approximate(lists, embeddingDict, freq_words):
    rows=len(lists)
    unknown_rows=0
    
    
    word_list=embeddingDict.keys()
    emb_matrix=mat(embeddingDict.values())
    pred_matrix=[]
    gold_word_list=[]
    
    data_matrix=[]
    max_simi_hope=[]
    for row in range(rows):
        a=lists[row][0]
        b=lists[row][1]
        c=lists[row][2]
        d=lists[row][3]
        emb_1=embeddingDict.get(a)
        emb_2=embeddingDict.get(b)
        emb_3=embeddingDict.get(c)
        emb_4=embeddingDict.get(d)
        if emb_1 is None:
            emb_1=embeddingDict.get(lists[row][0].lower())
            a=lists[row][0].lower()
        if emb_2 is None:
            emb_2=embeddingDict.get(lists[row][1].lower())
            b=lists[row][1].lower()
        if emb_3 is None:
            emb_3=embeddingDict.get(lists[row][2].lower())
            c=lists[row][2].lower()
        if emb_4 is None:
            emb_4=embeddingDict.get(lists[row][3].lower())
            d=lists[row][3].lower()
        
        
        if emb_1 is not None and emb_2 is not None and emb_3 is not None and emb_4 is not None:#a valid row
            data_row=[a,b,c,d]
            data_matrix.append(data_row)
            pred_emb=list(numpy.array(emb_2)-numpy.array(emb_1)+numpy.array(emb_3))
            max_simi_hope.append(cosine_simi(emb_4, pred_emb))
            pred_matrix.append(pred_emb)
            gold_word_list.append(d)
        else:
            unknown_rows+=1
    
    #word_list_plus_d=
    freq_emb_matrix=mat(itemgetter(*freq_words)(embeddingDict))
    
    pred_matrix=mat(pred_matrix)
    simi_matrix=1-cdist(pred_matrix,freq_emb_matrix, 'cosine')
    #max_index_list=simi_matrix.argsort()[:,emb_matrix.shape[0]-1]
    max_index_matrix=simi_matrix.argsort()[:,-4:]
    pred_word_matrix=[]
    for row in range(max_index_matrix.shape[0]):
        pred_list=[freq_words[i].lower() for i in max_index_matrix[row]] 
        pred_word_matrix.append(pred_list)
    match_count=0
    row_count=len(gold_word_list)
    for i in range(row_count):
        for j in [3,2,1,0]:
            if pred_word_matrix[i][j]==data_matrix[i][0].lower() or pred_word_matrix[i][j]==data_matrix[i][1].lower() or pred_word_matrix[i][j]==data_matrix[i][2].lower():
                #print pred_word_matrix[i][j], data_matrix[i][0].lower(), data_matrix[i][1].lower(), data_matrix[i][2].lower()
                continue
            elif pred_word_matrix[i][j]==data_matrix[i][3].lower() or simi_matrix[i][max_index_matrix[i][j]]<max_simi_hope[i]:
                match_count+=1
                break
            else:
                #print simi_matrix[i][max_index_matrix[i][j]], max_simi_hope[i]
                break
    print 'totally rows: ', rows, ' known rows ', row_count, ' unknown rows ',  unknown_rows, ' acc: ', match_count*1.0/row_count
    return row_count, match_count 
    
def word_analogy(embeddingDict):
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Emb_Extend/new_results/svd_True_d300_0_1_2_3_4.txt')
    #embeddingDict=loadEmbeddingFile('/mounts/data/proj/wenpeng/Tensor/result/tensor_ppmi/word2embedding_giga_wiki_20141201_300d.txt')
    #embeddingDict=loadEmbeddingFile(filename) #hlbl-embeddings-original.EMBEDDING_SIZE=100.txt')
    

    seman_lists, synt_lists=load_word_analogy()
    '''
    rows_seman, suc_seman=search_word_analogy_new(seman_lists, embeddingDict)
    rows_synt, suc_synt=search_word_analogy_new(synt_lists, embeddingDict)
    '''
    
    freq_words=load_freq_words()
    rows_seman, suc_seman=search_word_analogy_approximate(seman_lists, embeddingDict, freq_words)
    rows_synt, suc_synt=search_word_analogy_approximate(synt_lists, embeddingDict, freq_words)    
    
    
    print 'seman acc: ', suc_seman*1.0/rows_seman, ' synt acc: ', suc_synt*1.0/rows_synt, ' total acc: ', (suc_seman+suc_synt)*1.0/(rows_seman+rows_synt)

def load_freq_words():
    read_file=open('/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/overlap_vocab_0_1_2_3_4.txt','r')
    words_list=[]
    line_control=0
    for line in read_file:
        words_list.append(line.strip())
        line_control+=1
        #if line_control==20000:
        #    break
    read_file.close()
    print 'Freq words loaded over.'
    return words_list

def trancate_by_overlap_vocab():
    freq_vocab=load_freq_words()
    files_datasets=['hlbl-embeddings-original.EMBEDDING_SIZE=100.txt', 'Huang_embeddings.txt','glove.42B.300d_L2norm_per_row.txt','CW_L2norm_per_row.txt', 'word2vec_words_300d.txt']
    files_O2M_plus=['O2M_plus_bias_d200_0_1_2_3_4.txt', 'O2M_plus_bias_d200_1_2_3_4.txt',
           'O2M_plus_bias_d200_0_2_3_4.txt', 'O2M_plus_bias_d200_0_1_3_4.txt',
           'O2M_plus_bias_d200_0_1_2_4.txt', 'O2M_plus_bias_d200_0_1_2_3.txt']
    '''
    for filename in files_datasets:
        fileinput='/mounts/data/proj/wenpeng/Dataset/'+filename
        embeddingDict=loadEmbeddingFile(fileinput)
        
        writefile=open('/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/'+filename+'_overlap.txt', 'w')
        for word in freq_vocab:
            writefile.write(word+'\t'+' '.join(map(str, embeddingDict.get(word)))+'\n')
        writefile.close()
    '''
    for filename in files_O2M_plus:
        fileinput='/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/'+filename
        embeddingDict=loadEmbeddingFile(fileinput)
        
        writefile=open('/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/'+filename+'_overlap.txt', 'w')
        for word in freq_vocab:
            writefile.write(word+'\t'+' '.join(map(str, embeddingDict.get(word)))+'\n')
        writefile.close()     
    print 'trancate over'   

def run(arg):    
    #trancate_by_overlap_vocab()
    files=['hlbl-embeddings-original.EMBEDDING_SIZE=100.txt', 'Huang_embeddings.txt','glove.42B.300d_L2norm_per_row.txt','CW_L2norm_per_row.txt', 'word2vec_words_300d.txt']
#     files=['conc_hs_wp_0_1_2_3_4_L2norm_per_row.txt', 'conc_hs_wp_1_2_3_4.txt', 'conc_hs_wp_0_2_3_4_overlap.txt', 'conc_hs_wp_0_1_3_4_overlap.txt', 'conc_hs_wp_0_1_2_4.txt', 'conc_hs_wp_0_1_2_3_overlap.txt']
#     files=['svd_True_d200_0_1_2_3_4.txt', 'svd_True_d200_1_2_3_4.txt','svd_True_d200_0_2_3_4_overlap.txt', 'svd_True_d200_0_1_3_4_overlap.txt', 'svd_True_d200_0_1_2_4.txt', 'svd_True_d200_0_1_2_3_overlap.txt']
#     files=['O2M_bias_d200_0_1_2_3_4.txt', 'O2M_bias_d200_1_2_3_4.txt', 'O2M_bias_d200_0_2_3_4_overlap.txt', 'O2M_bias_d200_0_1_3_4_overlap.txt', 'O2M_bias_d200_0_1_2_4.txt', 'O2M_bias_d200_0_1_2_3_overlap.txt']
#     files=['O2M_plus_bias_d200_0_1_2_3_4.txt_overlap.txt', 'O2M_plus_bias_d200_1_2_3_4.txt_overlap.txt', 'O2M_plus_bias_d200_0_2_3_4.txt_overlap.txt',
#            'O2M_plus_bias_d200_0_1_3_4.txt_overlap.txt', 'O2M_plus_bias_d200_0_1_2_4.txt_overlap.txt', 'O2M_plus_bias_d200_0_1_2_3.txt_overlap.txt']#'conc_hs_wp_0_1_2_3_4.txt', 'svd_True_d200_0_1_2_3_4.txt', 'O2M_bias_d200_0_1_2_3_4.txt', ]

#     files=[
#             'hlbl-embeddings-original.EMBEDDING_SIZE=100.txt_overlap.txt',
#             'Huang_embeddings.txt_overlap.txt',
#             'glove.42B.300d_L2norm_per_row.txt_overlap.txt',
#             'CW_L2norm_per_row.txt_overlap.txt',
#             'word2vec_words_300d.txt_overlap.txt',
#              
# #            'O2M_bias_d200_0_1_2_3_4.txt',
# #            'O2M_bias_d200_1_2_3_4.txt',
# #            'O2M_bias_d200_0_2_3_4.txt',
# #            'O2M_bias_d200_0_1_3_4.txt',
# #            'O2M_bias_d200_0_1_2_4.txt',
# #            'O2M_bias_d200_0_1_2_3.txt'
#            ]
#     files=['/mounts/data/proj/lamb/data/embeddings/de/cow/8.4g/form-cbow_400_2_50_1-maximilian']
    
    #files=['O2M_plus_bias_d200_1_2_3_4.txt']
    for filename in files:
        fileinput='/mounts/data/proj/wenpeng/Dataset/'+filename  #Emb_Extend/results_L2norm_forall/
#         fileinput=filename
#         filename='/mounts/data/proj/wenpeng/Dataset/hlbl-embeddings-original.EMBEDDING_SIZE=100.txt'
#         filename='/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/conc_hs_wp_0_1_2_3_4_L2norm_per_row.txt'
#         filename_OOV='/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/O2M_bias_d200_0_1_2_3_OOV_embs.txt'
#         embeddingDict=loadEmbeddingFile_extend(filename, filename_OOV)
        embeddingDict=loadEmbeddingFile(fileinput)
#         Simlex999(embeddingDict)
#         wordsimi353(embeddingDict)
#         MC30(embeddingDict)
#         RG65(embeddingDict)
#         RW(embeddingDict)
        MEN(embeddingDict)
#         word_analogy(embeddingDict)
        
    
    '''
    dims=[50,100,150,200,250,300,350]
    version_indices=[0,1,2,3,4]
    suffix=''
    for index in version_indices:
        suffix+='_'+str(index)
    for dim in dims:
        filename='/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/O2M_bias_d'+str(dim)+suffix+'.txt'
        embeddingDict=loadEmbeddingFile(filename)
        wordsimi353(embeddingDict)
        MC30(embeddingDict)
        RG65(embeddingDict)
        SCWS(embeddingDict)
        RW(embeddingDict)
    '''
    '''   
    dims=[50,100,150,200,250,300,350]
    version_indices=[0,1,2,3,4]
    for dim in dims:

        for index in version_indices:
            suffix=''
            for need in version_indices:
                if need !=index:
                    suffix+='_'+str(need)
            
            filename='/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/O2M_bias_d'+str(dim)+suffix+'.txt'
            embeddingDict=loadEmbeddingFile(filename)
            wordsimi353(embeddingDict)
            MC30(embeddingDict)
            RG65(embeddingDict)
            SCWS(embeddingDict)
            RW(embeddingDict)
    '''
    '''
    embeddingDict=loadEmbeddingFile_extend(arg[0], arg[1])
    #embeddingDict=loadEmbeddingFile(arg[0])
    word_analogy(embeddingDict)
    '''
    #findTopTenNeighbors()
    
    
if __name__ == '__main__':
    run(sys.argv[1:])
    
'''
                  wordsimi353        MC30                RG65    SCWS    RW

#individual
glove.42B.300d_L2norm_per_row.txt         0.75376915509(18)     0.836003581       0.829124371572    0.650745143063(20)    0.486831655398(21)
hlbl_L2norm_per_row.txt                   0.356859056663    0.415442822924    0.352463053238    0.47593903826(96)     0.191344137616(892)
Huang_L2norm_per_row.txt                  0.617478186329    0.659101039905    0.629891491501    0.44665804917(20)     0.0639799118073(982)
word2vec_words_300d_L2norm_per_row.txt    0.697795566416    0.788607051121    0.760782860385    0.665443918679(21)    0.534209758232(209)
CW_L2norm_per_row.txt                     0.283592877161    0.216955946626    0.299481404207    0.398900944061(97)    0.152926141521(896)
#comb
conc_0_1_2_3_4.txt                        0.781221342846    0.821984889064    0.831175377333    0.665527770256(127)    0.579081945948(1212)
conc_0_1_2_3.txt                          0.78126039018     0.821984889064    0.831175377333    0.665548330662(126)    0.579604195289(1209)
# bias=[36.598,992.27,445.85,1808.04, 14.87]
conc_hs&wp_0_1_2_3_4.txt                  0.77123002343     0.836226099919    0.828817075755    0.675364143341(127)    0.6121430466(1212)
conc_hs&wp_0_1_2_3.txt                    0.771235979804    0.836226099919    0.828817075755    0.675422990148(126)    0.611704919892(1209)
conc_hs&wp_1_2_3_4.txt                    0.771254676197    0.836226099919    0.828817075755    0.675344353835(127)    0.612920536615(1211)
#bias=[1,4,1,4,1]
conc_hs&wp_0_1_2_3_4.txt                  0.765844634926    0.817534510672    0.813224809986    0.683415598647         0.624153944936
#bias=[exp(1),exp(4),exp(1),exp(4),exp(1)]
conc_hs&wp_0_1_2_3_4.txt                  0.76503208637 (21)     0.863373408113    0.824947629477    0.684692053276         0.628834487749
conc_hs&wp_0_1_2_3.txt                    0.765053760949    0.86515355947     0.825451344377    0.684787636591         0.628842998878
conc_hs&wp_0_1_2_4.txt                    0.694027321507    0.791277278157    0.755892895908    0.668009414433         0.615088280466
conc_hs&wp_0_1_3_4.txt                    0.765055746407    0.863373408113    0.824947629477    0.68466294758          0.602276891147(936)
conc_hs&wp_0_2_3_4.txt                    0.755155096226    0.836003581       0.823917303546    0.657477643849         0.586222519535(1189)
conc_hs&wp_1_2_3_4.txt                    0.764999988137    0.863373408113    0.824947629477    0.684665119063         0.629508029712(1211)

O2M_bias_d200_0_1_2_3                     0.764548296515(21)    0.848464640498(1)    0.833808432492    0.672325901009(126)         0.62035651444(1209)


svd_True_d200_0_1_3_4+OOV                 0.715551054872    0.854472651328    0.828709120945    0.640709484486         0.479313827945(21)





#next, use L2 norm for all versions
        before        after
glove-42B-300d     0.642731252846    0.75376915509
Huang   0.617478186329 0.61941641909
CW    0.283592877161    0.278301242662
hlbl    0.356859056663    0.367243584128
word2vec    0.697795566416    0.697844267373

#new results after all L2, dot prod for concate
conc_0_1_2_3_4    0.781221342846
conc_1_2_3_4    0.781211746467
conc_0_2_3_4    0.76514261018
conc_0_1_3_4    0.7788670864
conc_0_1_2_4    0.685920697795
conc_0_1_2_3    0.78126039018
glove.42B.300d_L2norm    0.759612839904
svd_True_d250_0_1_2_3    0.764021322955
svd_True_d350_0_1_2_3    0.750003660818
svd_True_d200_0_1_2_3    0.77233493063
svd_True_d500_0_1_2_3    0.737807821557
svd_True_d300_0_1_2_3_4    0.764037206617
svd_True_d200_0_1_2_3_4    0.772535792765
svd_True_d250_0_1_2_3_4    0.764041508442


#new results after all L2, dot prod for hs
conc_hs_0_1_2_3_4    0.769898277691    0.77123002343
conc_hs_0_1_2_3    0.772444130802    0.771235979804
conc_hs_0_1_2_4    0.70553205604    0.706357344616
conc_hs_0_1_3_4    0.770312907437    0.769820844841
conc_hs_0_2_3_4    0.753439826244    0.758243310194
conc_hs_1_2_3_4    0.753439826244    0.771254676197


'''



