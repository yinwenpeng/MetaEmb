import os
import sys
from math import exp

def load_vocab_version(filename, dim):    
    open_file=open(filename, 'r')
    vocab=set()
    line_control=0
    for line in open_file:
        tokens=line.strip().split()
        vocab.add(tokens[0])       
    print 'loading vocab of '+filename+' size '+str(len(vocab))    
    return vocab
class embedding_version(object):
    def __init__(self, filename, dim=100, overlap_vocab=set(), bias=0.001):
        self.filename=filename
        self.dim=dim
        self.embeddings={}
        self.vocab=overlap_vocab
    #def load_embedding(self):
        open_file=open(self.filename, 'r')
        print 'loading '+self.filename
        
        line_control=0
        for line in open_file:
            tokens=line.strip().split()
            if tokens[0] not in self.vocab: #the first line of word2vec is (words, dim)
                continue
            #elif line_control < 20000:
            else:
                #line_control+=1
                self.embeddings[tokens[0]]=[x*bias for x in map(float, tokens[1:])]           
        self.vocab=set(self.embeddings.keys())
        self.vocab_size=len(self.vocab)
        print '\t\t\t\t loaded over, '+str(len(self.embeddings))+' overlapping embeddings.'
        
class embedding_version_full(object):
    def __init__(self, filename, dim=100):
        self.filename=filename
        self.dim=dim
        self.embeddings={}
        #self.vocab=overlap_vocab
    #def load_embedding(self):
        open_file=open(self.filename, 'r')
        print 'loading '+self.filename
        
        line_control=0
        for line in open_file:
            tokens=line.strip().split()
            self.embeddings[tokens[0]]=map(float, tokens[1:])     
            line_control+=1     
        self.vocab=set(self.embeddings.keys())
        print '\t\t\t\t loaded over, '+str(line_control)+'  embeddings.'

class embedding_version_extended(object):
    def __init__(self, origin_filename, OOV_filename, dim=100):
        
        self.dim=dim
        self.embeddings={}
        self.vocab=set()
        for filename in [origin_filename, OOV_filename]:
            #first load original embeddings
            self.filename=filename
            open_file=open(self.filename, 'r')
            print 'loading '+self.filename
            
            line_control=0
            for line in open_file:
                tokens=line.strip().split()
                if len(tokens)!=(self.dim+1) or tokens[0].find('_')!=-1: #the first line of word2vec is (words, dim)
                    continue
                #elif line_control < 20000:
                else:
                    line_control+=1
    
                    embedding=[]
                    for j in range(1, self.dim+1):
                        embedding.append(float(tokens[j])) # str to float
                    self.embeddings[tokens[0]]=embedding      
                    #if line_control==100:   #can not only read 100 words, because different version stored in different orders. then the word_list is different
                    #    break      
            #self.vocab=set(self.embeddings.keys())
            open_file.close()
            print '\t\t\t\t'+self.filename+' loaded over, '+str(line_control)+' valid embeddings, totally:'+str(len(self.embeddings))+' embeddings.'  
        self.vocab=set(self.embeddings.keys())    

def load_versions_EE_Ensemble_OOV(target_filename, target_dim=950):
    root='/mounts/data/proj/wenpeng/Dataset/'
    lbl=embedding_version_full(root+'hlbl_L2norm_per_row.txt', 100)
    word2vec=embedding_version_full(root+'word2vec_words_300d_L2norm_per_row.txt', 300)
    Huang=embedding_version_full(root+'Huang_L2norm_per_row.txt', 50)
    glove=embedding_version_full(root+'glove.42B.300d_L2norm_per_row.txt', 300)
    collobert=embedding_version_full(root+'CW_L2norm_per_row.txt', 200)
    
    target_version=embedding_version_full(target_filename, target_dim)

    overall_vocab=set()
    overlap_vocab=set()
    overall_vocab=lbl.vocab | word2vec.vocab | Huang.vocab | glove.vocab | collobert.vocab
    overlap_vocab=lbl.vocab & word2vec.vocab & Huang.vocab & glove.vocab & collobert.vocab
    
    print 'overall_vocab: '+str(len(overall_vocab))+', overalp_vocab: '+str(len(overlap_vocab))   

    return lbl, word2vec, Huang, glove, collobert, target_version, overall_vocab, overlap_vocab

def write_version(root, input_filename, version_i):
    out_filename=root + input_filename[input_filename.rindex('/')+1:input_filename.rindex('.txt')]+'_overlap_embs.txt'
    print '...is writing into', out_filename
    write_file=open(out_filename, 'w')
    for word, emb in version_i.embeddings.iteritems():
        write_file.write(word+'\t')
        dim=len(emb)
        for i in range(dim):
            write_file.write(str(emb[i])+' ')
        write_file.write('\n')
    write_file.close()
    
def load_versions(version_indices, weight):
    root='/mounts/data/proj/wenpeng/Dataset/'
    paths=['hlbl_L2norm_per_row.txt','word2vec_words_300d_L2norm_per_row.txt','Huang_L2norm_per_row.txt','glove.42B.300d_L2norm_per_row.txt','CW_L2norm_per_row.txt']
    dims=[100, 300, 50, 300, 200]
    #bias=[36.598,992.27,445.85,1808.04, 14.87]
    #bias=[exp(1),exp(4),exp(1),exp(4),exp(1)]
    bias=[1.0,weight,1.0,weight,1.0] # this bias is for O2M, we use original normalized as targets, while put different weights for costs.

    #first find overlap vocab set
    overall_vocab=set()
    overlap_vocab=set()
    count=0
    overall_dim=0
    for index in version_indices:
        overall_dim+=dims[index]
        vocab_i=load_vocab_version(root+paths[index], dims[index])    
        if count<1:
            overall_vocab=vocab_i
            overlap_vocab=vocab_i
        else:
            overall_vocab=overall_vocab|vocab_i
            overlap_vocab=overlap_vocab&vocab_i
        count+=1    
    
    
    versions=[]
    for index in version_indices:
        version_i=embedding_version(root+paths[index], dims[index], overlap_vocab, bias[index])
        #write_version(root, paths[index], version_i)
        versions.append(version_i)
    
    print 'overall_vocab: '+str(len(overall_vocab))+', overlap_vocab: '+str(len(overlap_vocab))+' overall dim: '+str(overall_dim)
    #exit(0)
    return versions, overall_vocab, overlap_vocab, overall_dim

def load_versions_for_O2M_plus(version_indices, weight):
    root='/mounts/data/proj/wenpeng/Dataset/'
    paths=['hlbl_L2norm_per_row.txt','word2vec_words_300d_L2norm_per_row.txt','Huang_L2norm_per_row.txt','glove.42B.300d_L2norm_per_row.txt','CW_L2norm_per_row.txt']
    dims=[100, 300, 50, 300, 200]
    #bias=[36.598,992.27,445.85,1808.04, 14.87]
    #bias=[exp(1),exp(4),exp(1),exp(4),exp(1)]
    bias=[1.0,weight,1.0,weight,1.0] # this bias is for O2M, we use original normalized as targets, while put different weights for costs.

    #first find overlap vocab set
    overall_vocab=set()
    overlap_vocab=set()
    count=0
    overall_dim=0
    for index in version_indices:
        overall_dim+=dims[index]
        vocab_i=load_vocab_version(root+paths[index], dims[index])    
        if count<1:
            overall_vocab=vocab_i
            overlap_vocab=vocab_i
        else:
            overall_vocab=overall_vocab|vocab_i
            overlap_vocab=overlap_vocab&vocab_i
        count+=1    
    
    
    versions=[]
    for index in version_indices:
        version_i=embedding_version(root+paths[index], dims[index], overall_vocab, bias[index]) #overall_vocab makes sure to read all embeddings
        #write_version(root, paths[index], version_i)
        versions.append(version_i)
    
    print 'overall_vocab: '+str(len(overall_vocab))+', overlap_vocab: '+str(len(overlap_vocab))+' overall dim: '+str(overall_dim)
    #exit(0)
    return versions, overall_vocab, overlap_vocab, overall_dim

def load_versions_extended():
    root1='/mounts/data/proj/wenpeng/Dataset/'
    root2='/mounts/data/proj/wenpeng/Emb_Extend/'
    lbl=embedding_version_extended(root1+'hlbl-embeddings-original.EMBEDDING_SIZE=100.txt', root2+'0_OOV_embs.txt',100)
    word2vec=embedding_version_extended(root1+'GoogleNews-vectors-negative300.txt', root2+'1_OOV_embs.txt', 300)
    Huang=embedding_version_extended(root1+'Huang_embeddings.txt', root2+'2_OOV_embs.txt', 50)
    glove=embedding_version_extended(root1+'glove.6B.300d.txt', root2+'3_OOV_embs.txt', 300)
    collobert=embedding_version_extended(root1+'embeddings-scaled.EMBEDDING_SIZE=200.txt', root2+'4_OOV_embs.txt', 200)

    overall_vocab=lbl.vocab
    
    print 'overall_vocab: '+str(len(overall_vocab))

    return lbl, word2vec, Huang, glove, collobert, overall_vocab

    
'''    
if __name__ == '__main__':
    read_emb_versions()
'''                
        
    










