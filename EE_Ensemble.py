

import sys
sys.setrecursionlimit(6000)


import numpy
import theano
import theano.tensor as T

import time
from mlp import HiddenLayer


from word2embeddings.nn.util import random_value_normal

from cis.deep.utils.theano import debug_print

from EE_read_versions import load_versions_extended, load_versions, load_versions_for_O2M_plus
from dA_random_input import dA_random_input
from theano.tensor.shared_randomstreams import RandomStreams
from TensorBuild import cosVector
from scipy.sparse import csr_matrix
import threading
import logging
from scipy import linalg, mat, dot
from math import exp
import ExtRescal.rescal as rescal
from test_embedding_quality import Simlex999

from operator import itemgetter

#in this file, we try three emsemble methods: concatenate, autoencoder, tensor factorization
#fist we do experiments on overlap vocab


class Ensemble(object):
    def __init__(self, compose_style=1, version_indices=[0,1,2,3,4],weight=20):
        self.concatenate_embs={}
        self.version_indices=version_indices
        if compose_style==1:
            lbl, word2vec, Huang, glove, collobert, overall_vocab=load_versions_extended()
            self.lbl=lbl
            self.word2vec=word2vec
            self.Huang=Huang
            self.glove=glove
            self.collobert=collobert
            self.overall_vocab=overall_vocab
            self.word_list=list(self.overall_vocab)
            self.versions=[lbl, word2vec, Huang, glove, collobert]
            self.concatenate_embs={}
        elif compose_style==2:
            #autoencoder
            self.word_list=[]
            self.load_concatenated()
        elif compose_style==3:
            #considering different combination methods on overlap first, then learn embeddings for rare words
            
            self.versions, overall_vocab, overlap_vocab, overall_dim=load_versions(version_indices,weight)
            self.overall_vocab=overall_vocab
            self.overlap_vocab=overlap_vocab
            self.overall_dim=overall_dim
            self.word_list=list(overall_vocab)
            self.overlap_word_list=list(overlap_vocab)
            #self.versions=[lbl, word2vec, Huang, glove, collobert]
            self.concatenate_embs={}           
        elif compose_style==4:
            #autoencoder
            self.word_list=[]
            self.load_concatenated_overlap()       
        elif compose_style==5:
            #considering different combination methods on overlap first, then learn embeddings for rare words
            
            self.versions, overall_vocab, overlap_vocab, overall_dim=load_versions_for_O2M_plus(version_indices,1.0)
            self.overall_vocab=overall_vocab
            self.overlap_vocab=overlap_vocab
            self.overall_dim=overall_dim
            self.word_list=list(self.overlap_vocab)+list(self.overall_vocab - self.overlap_vocab)#list(overall_vocab)
            #self.overlap_word_list=list(overlap_vocab)
            #self.versions=[lbl, word2vec, Huang, glove, collobert]
            self.concatenate_embs={}        
    def load_concatenated(self):
        read_file=open('/mounts/data/proj/wenpeng/Emb_Extend/ensamble_concatenate.txt', 'r')
        line_count=0
        for line in read_file:
            line_count+=1
            tokens=line.strip().split()
            self.overall_dim=len(tokens)-1
            emb=[]
            for j in range(1, self.overall_dim+1):
                emb.append(float(tokens[j]))
            self.concatenate_embs[tokens[0]]=emb
            self.word_list.append(tokens[0])
            #if line_count ==10000:
            #    break
            if line_count%50000==0:
                print line_count
        print 'Concatenated Embs loaded over, totally: '+str(len(self.concatenate_embs))
        
    def load_concatenated_overlap(self):
        suffix=''
        for index in self.version_indices:
            suffix+='_'+str(index)
        read_filename='/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/conc_hs_wp'+suffix+'.txt'
        read_file=open(read_filename, 'r')
        line_count=0
        for line in read_file:
            line_count+=1
            tokens=line.strip().split()
            self.overall_dim=len(tokens)-1
            emb=[]
            for j in range(1, self.overall_dim+1):
                emb.append(float(tokens[j]))
            self.concatenate_embs[tokens[0]]=emb
            self.word_list.append(tokens[0])
            #if line_count ==700:
            #    break
            #if line_count%5000==0:
            #    print line_count
        print 'Concatenated Overlap Embs loaded over, totally: '+str(len(self.concatenate_embs))
            
    
    def concatenate(self):
        write_file=open('/mounts/data/proj/wenpeng/Emb_Extend/ensamble_concatenate.txt', 'w')
        self.overall_dim=self.lbl.dim+self.word2vec.dim+self.Huang.dim+self.glove.dim+self.collobert.dim
        overall_list=list(self.overall_vocab)
        for word in overall_list:
            word_emb=self.lbl.embeddings[word]+self.word2vec.embeddings[word]+self.Huang.embeddings[word]+self.glove.embeddings[word]+self.collobert.embeddings[word]
            self.concatenate_embs[word]=word_emb
            write_file.write(word+'\t')
            for j in range(self.overall_dim):
                write_file.write(str(word_emb[j])+' ')
            write_file.write('\n')
        write_file.close()
        print 'Concatenation finished.'

    def concatenate_overlap(self, weight):
        suffix=''
        for index in self.version_indices:
            suffix+='_'+str(index)
        write_filename='/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/conc_hs&wp_1to100_'+str(weight)+suffix+'.txt'
        write_file=open(write_filename, 'w')
        #self.overall_dim=self.lbl.dim+self.word2vec.dim+self.Huang.dim+self.glove.dim+self.collobert.dim
        #overall_list=list(self.overall_vocab)
        for word in self.overlap_word_list:
            word_emb=[]
            for version in self.versions:
                word_emb+=version.embeddings[word]
            #word_emb=self.lbl.embeddings[word]+self.word2vec.embeddings[word]+self.Huang.embeddings[word]+self.glove.embeddings[word]+self.collobert.embeddings[word]
            self.concatenate_embs[word]=word_emb
            write_file.write(word+'\t')
            for j in range(self.overall_dim):
                write_file.write(str(word_emb[j])+' ')
            write_file.write('\n')
        write_file.close()
        print 'Concatenation overlap finished.'
    
    def store_target_word_embeddings(self, word_list, best_target_embs, dim, method):
        suffix=''
        for index in self.version_indices:
            suffix+='_'+str(index)
        write_file=open('/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/'+method+'_d'+str(dim)+suffix+'.txt', 'w')
        size=len(word_list)
        length=best_target_embs.shape[1]
        for i in range(size):
            write_file.write(word_list[i]+'\t')
            for j in range(length):
                write_file.write(str(best_target_embs[i][j])+' ')
            write_file.write('\n')
        write_file.close()

    def store_target_word_embeddings_bigscale(self, word_list, best_target_embs, dim):
        suffix=''
        for index in self.version_indices:
            suffix+='_'+str(index)
        write_words=open('/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/O2M_plus_bias_d'+str(dim)+suffix+'_words_rerun.txt', 'w')
        numpy.savetxt('/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/O2M_plus_bias_d'+str(dim)+suffix+'_values_rerun.txt', best_target_embs)
        size=len(word_list)
        #length=best_target_embs.shape[1]
        for i in range(size):
            write_words.write(word_list[i]+'\n')
        write_words.close()
    
    def map2one(self, training_epochs=200, batch_size=100, learning_rate=0.001, target_embedding_size=300, cost_ratio=0.5):
        #lbl, word2vec, Huang, glove, collobert
        word_count=len(self.overlap_word_list)
        self.index_list=range(word_count)
        target_embs=numpy.zeros((word_count, target_embedding_size), dtype=theano.config.floatX)

        lbl = mat(itemgetter(*self.overlap_word_list)(self.lbl.embeddings))
        word2vec = mat(itemgetter(*self.overlap_word_list)(self.word2vec.embeddings))
        Huang = mat(itemgetter(*self.overlap_word_list)(self.Huang.embeddings))
        glove = mat(itemgetter(*self.overlap_word_list)(self.glove.embeddings))
        collobert = mat(itemgetter(*self.overlap_word_list)(self.collobert.embeddings))
        
        self.train_lbl=theano.shared(value=numpy.array(lbl, dtype=theano.config.floatX), borrow=True) 
        self.train_word2vec=theano.shared(value=numpy.array(word2vec, dtype=theano.config.floatX), borrow=True) 
        self.train_Huang=theano.shared(value=numpy.array(Huang, dtype=theano.config.floatX), borrow=True) 
        self.train_glove=theano.shared(value=numpy.array(glove, dtype=theano.config.floatX), borrow=True) 
        self.train_collobert=theano.shared(value=numpy.array(collobert, dtype=theano.config.floatX), borrow=True) 
        
        self.train_index_list=theano.shared(value=numpy.array(self.index_list, dtype=theano.config.floatX), borrow=True) 
        train_index_list=T.cast(self.train_index_list, 'int32')
        
        n_train_batches = word_count / batch_size
        if word_count % batch_size !=0:
            train_batch_start=list(numpy.arange(n_train_batches)*batch_size)+[word_count-batch_size]
            n_train_batches=n_train_batches+1
        else:
            train_batch_start=list(numpy.arange(n_train_batches)*batch_size)
        
        index = T.lscalar()    # index to a [mini]batch
        indices=T.ivector('indices')
        
        print '... building the model'
        rng = numpy.random.RandomState(23455)
        
        
        
        lbl_input=self.train_lbl[indices]
        word2vec_input=self.train_word2vec[indices]
        Huang_input=self.train_Huang[indices]
        glove_input=self.train_glove[indices]
        collobert_input=self.train_collobert[indices]
        
        layer_lbl = HiddenLayer(rng, input=lbl_input, n_in=self.lbl.dim, n_out=target_embedding_size, activation=None)
        layer_word2vec = HiddenLayer(rng, input=word2vec_input, n_in=self.word2vec.dim, n_out=target_embedding_size, activation=None)
        layer_Huang = HiddenLayer(rng, input=Huang_input, n_in=self.Huang.dim, n_out=target_embedding_size, activation=None)
        layer_glove = HiddenLayer(rng, input=glove_input, n_in=self.glove.dim, n_out=target_embedding_size, activation=None)
        layer_collobert = HiddenLayer(rng, input=collobert_input, n_in=self.collobert.dim, n_out=target_embedding_size, activation=None)
        '''
        layer_list=[layer_lbl,layer_word2vec, layer_Huang, layer_glove, layer_collobert]
        for i in range(len(layer_list)):
        '''    
        cost_0=T.mean(T.sum((layer_lbl.output - layer_word2vec.output)**2, axis=1))
        cost_1=T.mean(T.sum((layer_lbl.output - layer_Huang.output)**2, axis=1))
        cost_2=T.mean(T.sum((layer_lbl.output - layer_glove.output)**2, axis=1))
        cost_3=T.mean(T.sum((layer_lbl.output - layer_collobert.output)**2, axis=1))
        cost_4=T.mean(T.sum((layer_word2vec.output - layer_Huang.output)**2, axis=1))
        cost_5=T.mean(T.sum((layer_word2vec.output - layer_glove.output)**2, axis=1))
        cost_6=T.mean(T.sum((layer_word2vec.output - layer_collobert.output)**2, axis=1))
        cost_7=T.mean(T.sum((layer_Huang.output - layer_glove.output)**2, axis=1))
        cost_8=T.mean(T.sum((layer_Huang.output - layer_collobert.output)**2, axis=1))
        cost_9=T.mean(T.sum((layer_glove.output - layer_collobert.output)**2, axis=1))
        
        #similarity cost. We expect for give batch, the mutual similarity before projection and after projection keep stable
        cost_simi_lbl=T.mean(T.sqr(row_wise_cosine_theano(lbl_input)-row_wise_cosine_theano(layer_lbl.output)))
        cost_simi_word2vec=T.mean(T.sqr(row_wise_cosine_theano(word2vec_input)-row_wise_cosine_theano(layer_word2vec.output)))
        cost_simi_Huang=T.mean(T.sqr(row_wise_cosine_theano(Huang_input)-row_wise_cosine_theano(layer_Huang.output)))
        cost_simi_glove=T.mean(T.sqr(row_wise_cosine_theano(glove_input)-row_wise_cosine_theano(layer_glove.output)))
        cost_simi_collobert=T.mean(T.sqr(row_wise_cosine_theano(collobert_input)-row_wise_cosine_theano(layer_collobert.output)))
        

        cost=cost_ratio*(cost_0+cost_1+cost_2+cost_3+cost_4+cost_5+cost_6+cost_7+cost_8+cost_9)+(1-cost_ratio)*(cost_simi_lbl+cost_simi_word2vec+cost_simi_Huang
                                                                                                                +cost_simi_glove+cost_simi_collobert)
        
        output=(layer_lbl.output+layer_word2vec.output+layer_Huang.output+layer_glove.output+layer_collobert.output)/5.0
        
        self.params=layer_lbl.params+layer_word2vec.params+layer_Huang.params+layer_glove.params+layer_collobert.params
        accumulator=[]
        for para_i in self.params:
            eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
            accumulator.append(theano.shared(eps_p, borrow=True))
         
        # create a list of gradients for all model parameters
        grads = T.grad(cost, self.params)
        updates = []
        for param_i, grad_i, acc_i in zip(self.params, grads, accumulator):
            acc = acc_i + T.sqr(grad_i)
            updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc)))   #AdaGrad
            updates.append((acc_i, acc))  
            
        train_model = theano.function([index], cost, updates=updates,
              givens={
                indices: train_index_list[index: index + batch_size]})       
        dev_model = theano.function([index], [cost,output],
              givens={
                indices: train_index_list[index: index + batch_size]})   

        ############
        # TRAINING #
        ############
        print '... training'
        self.wait_iter=10
        epoch = 0
        vali_loss_list=[]
        lowest_vali_loss=0
        validation_frequency= n_train_batches
        while (epoch < training_epochs):
            epoch = epoch + 1
            #for minibatch_index in xrange(n_train_batches): # each batch
            minibatch_index=0
            for batch_start in train_batch_start: 
                # iter means how many batches have been runed, taking into loop
                iter = (epoch - 1) * n_train_batches + minibatch_index +1
    
                minibatch_index=minibatch_index+1
                average_cost_per_batch=train_model(batch_start)
                if iter % validation_frequency == 0:
                    print 'training @ iter = '+str(iter)+' cost: '+str(average_cost_per_batch)# +' error: '+str(error_ij)
                    validation_losses=[]            
                    for batch_start in train_batch_start:
                        vali_loss_i, output_i=dev_model(batch_start)
                        validation_losses.append(vali_loss_i)  
                        for row in range(batch_start, batch_start + batch_size):
                            target_embs[row]=output_i[row-batch_start]       
                    this_validation_loss = numpy.mean(validation_losses)
                    print('\t\tepoch %i, minibatch %i/%i, validation cost %f ' % \
                      (epoch, minibatch_index , n_train_batches, \
                       this_validation_loss))
                    
                    if this_validation_loss < (minimal_of_list(vali_loss_list)-5e-4):
                        del vali_loss_list[:]
                        vali_loss_list.append(this_validation_loss)
                        #remember the best params
                        self.best_target_embs=target_embs
                        lowest_vali_loss=this_validation_loss
                    elif len(vali_loss_list)<self.wait_iter:
                        if this_validation_loss < minimal_of_list(vali_loss_list):
                            self.best_target_embs=target_embs
                            lowest_vali_loss=this_validation_loss                            
                        vali_loss_list.append(this_validation_loss)
                        if len(vali_loss_list)==self.wait_iter:
                            self.store_target_word_embeddings(self.overlap_word_list, self.best_target_embs)
                            print 'Training over, best word target embeddings got at train_cost:'+str(lowest_vali_loss)+' map2one embs stored over.'
                            exit(0)    


    def one2multiple(self, training_epochs=200, batch_size=100, learning_rate=0.001, target_embedding_size=300, L2_weight=0.5):
        #lbl, word2vec, Huang, glove, collobert
        word_count=len(self.overlap_word_list)
        self.index_list=range(word_count)
        self.train_index_list=theano.shared(value=numpy.array(self.index_list, dtype=theano.config.floatX), borrow=True) 
        train_index_list=T.cast(self.train_index_list, 'int32')
        
        
        target_embs=random_value_normal((word_count, target_embedding_size), theano.config.floatX, numpy.random.RandomState(4321))
        self.target_embs=theano.shared(value=target_embs) 
        
        golds=[]
        for i in range(len(self.versions)):
            gold_matrix=mat(itemgetter(*self.overlap_word_list)(self.versions[i].embeddings))
            gold_theano=theano.shared(value=numpy.array(gold_matrix, dtype=theano.config.floatX), borrow=True) 
            golds.append(gold_theano)
        
        
        full_cost_bias=[exp(1),exp(4),exp(1),exp(4),exp(1)]
        cost_bias=[full_cost_bias[index] for index in self.version_indices]
        
        '''
        lbl = mat(itemgetter(*self.overlap_word_list)(self.lbl.embeddings))
        word2vec = mat(itemgetter(*self.overlap_word_list)(self.word2vec.embeddings))
        Huang = mat(itemgetter(*self.overlap_word_list)(self.Huang.embeddings))
        glove = mat(itemgetter(*self.overlap_word_list)(self.glove.embeddings))
        collobert = mat(itemgetter(*self.overlap_word_list)(self.collobert.embeddings))
        
        self.train_lbl=theano.shared(value=numpy.array(lbl, dtype=theano.config.floatX), borrow=True) 
        self.train_word2vec=theano.shared(value=numpy.array(word2vec, dtype=theano.config.floatX), borrow=True) 
        self.train_Huang=theano.shared(value=numpy.array(Huang, dtype=theano.config.floatX), borrow=True) 
        self.train_glove=theano.shared(value=numpy.array(glove, dtype=theano.config.floatX), borrow=True) 
        self.train_collobert=theano.shared(value=numpy.array(collobert, dtype=theano.config.floatX), borrow=True) 
        '''
        

        
        n_train_batches = word_count / batch_size
        if word_count % batch_size !=0:
            train_batch_start=list(numpy.arange(n_train_batches)*batch_size)+[word_count-batch_size]
            n_train_batches=n_train_batches+1
        else:
            train_batch_start=list(numpy.arange(n_train_batches)*batch_size)
        
        index = T.lscalar()    # index to a [mini]batch
        indices=T.ivector('indices')
        
        print '... building the model'
        rng = numpy.random.RandomState(23455)
        
        input_embs=self.target_embs[indices]
        cost_sum=0.0
        L2_reg=0.0
        params=[]
        for i in range(len(self.versions)):
            emb_true=golds[i][indices]
            layer_i = HiddenLayer(rng, input=input_embs, n_out=self.versions[i].dim, n_in=target_embedding_size, activation=None)
            cost_i=T.mean(T.sum((layer_i.output - emb_true)**2, axis=1))
            cost_sum+=cost_i*cost_bias[i] # we consider different weights for cost
            L2_reg+=(layer_i.W**2).sum()
            params=params+layer_i.params
        
        '''
        lbl_true=self.train_lbl[indices]
        word2vec_true=self.train_word2vec[indices]
        Huang_true=self.train_Huang[indices]
        glove_true=self.train_glove[indices]
        collobert_true=self.train_collobert[indices]
        
        input_embs=self.target_embs[indices]
        
        layer_lbl = HiddenLayer(rng, input=input_embs, n_out=self.lbl.dim, n_in=target_embedding_size, activation=None)
        layer_word2vec = HiddenLayer(rng, input=input_embs, n_out=self.word2vec.dim, n_in=target_embedding_size, activation=None)
        layer_Huang = HiddenLayer(rng, input=input_embs, n_out=self.Huang.dim, n_in=target_embedding_size, activation=None)
        layer_glove = HiddenLayer(rng, input=input_embs, n_out=self.glove.dim, n_in=target_embedding_size, activation=None)
        layer_collobert = HiddenLayer(rng, input=input_embs, n_out=self.collobert.dim, n_in=target_embedding_size, activation=None)
        '''
        #layer_list=[layer_lbl,layer_word2vec, layer_Huang, layer_glove, layer_collobert]
        #for i in range(len(layer_list)):
        '''    
        cost_0=T.mean(T.sum((layer_lbl.output - lbl_true)**2, axis=1))
        cost_1=T.mean(T.sum((layer_word2vec.output - word2vec_true)**2, axis=1))
        cost_2=T.mean(T.sum((layer_Huang.output - Huang_true)**2, axis=1))
        cost_3=T.mean(T.sum((layer_glove.output - glove_true)**2, axis=1))
        cost_4=T.mean(T.sum((layer_collobert.output - collobert_true)**2, axis=1))

        L2_reg = (layer_lbl.W**2).sum()+(layer_word2vec.W**2).sum()+(layer_Huang.W**2).sum()+( layer_glove.W**2).sum()+(layer_collobert.W**2).sum()
        cost=cost_0+cost_1+cost_2+cost_3+cost_4+L2_weight*L2_reg
        '''
        cost=cost_sum+L2_weight*L2_reg
        output=input_embs
        
        self.params=params+[self.target_embs]
        accumulator=[]
        for para_i in self.params:
            eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
            accumulator.append(theano.shared(eps_p, borrow=True))
         
        # create a list of gradients for all model parameters
        grads = T.grad(cost, self.params)
        updates = []
        for param_i, grad_i, acc_i in zip(self.params, grads, accumulator):
            acc = acc_i + T.sqr(grad_i)
            updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-10)))   #AdaGrad
            updates.append((acc_i, acc))  
            
        train_model = theano.function([index], cost, updates=updates,
              givens={
                indices: train_index_list[index: index + batch_size]})       
        dev_model = theano.function([index], [cost,output],
              givens={
                indices: train_index_list[index: index + batch_size]})   

        ############
        # TRAINING #
        ############
        print '... training'
        self.wait_iter=10
        epoch = 0
        vali_loss_list=[]
        lowest_vali_loss=0
        validation_frequency= n_train_batches
        while (epoch < training_epochs):
            epoch = epoch + 1
            #for minibatch_index in xrange(n_train_batches): # each batch
            minibatch_index=0
            for batch_start in train_batch_start: 
                # iter means how many batches have been runed, taking into loop
                iter = (epoch - 1) * n_train_batches + minibatch_index +1
    
                minibatch_index=minibatch_index+1
                average_cost_per_batch=train_model(batch_start)
                if iter % validation_frequency == 0:
                    print 'training @ iter = '+str(iter)+' cost: '+str(average_cost_per_batch)# +' error: '+str(error_ij)
                    validation_losses=[]            
                    for batch_start in train_batch_start:
                        vali_loss_i, output_i=dev_model(batch_start)
                        validation_losses.append(vali_loss_i)  
                        for row in range(batch_start, batch_start + batch_size):
                            target_embs[row]=output_i[row-batch_start]       
                    this_validation_loss = numpy.mean(validation_losses)
                    print('\t\tepoch %i, minibatch %i/%i, validation cost %f ' % \
                      (epoch, minibatch_index , n_train_batches, \
                       this_validation_loss))
                    
                    if this_validation_loss < (minimal_of_list(vali_loss_list)-1e-4):
                        del vali_loss_list[:]
                        vali_loss_list.append(this_validation_loss)
                        #remember the best params
                        self.best_target_embs=target_embs
                        self.store_target_word_embeddings(self.overlap_word_list, self.best_target_embs, target_embedding_size)
                        lowest_vali_loss=this_validation_loss
                    elif len(vali_loss_list)<self.wait_iter:
                        if this_validation_loss < minimal_of_list(vali_loss_list):
                            self.best_target_embs=target_embs
                            self.store_target_word_embeddings(self.overlap_word_list, self.best_target_embs, target_embedding_size)
                            lowest_vali_loss=this_validation_loss                            
                        vali_loss_list.append(this_validation_loss)
                        if len(vali_loss_list)==self.wait_iter:
                            #self.store_target_word_embeddings(self.overlap_word_list, self.best_target_embs, target_embedding_size)
                            print 'Training over, best word target embeddings got at train_cost:'+str(lowest_vali_loss)+' one2multiple embs stored over.'
                            exit(0)  


    def auto_multi(self, training_epochs=200, batch_size=100, learning_rate=0.001, target_embedding_size=300, L2_weight=0.5):
        #lbl, word2vec, Huang, glove, collobert
        word_count=len(self.overlap_word_list)
        print word_count
        self.index_list=range(word_count)
        self.train_index_list=theano.shared(value=numpy.array(self.index_list, dtype=theano.config.floatX), borrow=True) 
        train_index_list=T.cast(self.train_index_list, 'int32')
        
        
        target_embs=random_value_normal((word_count, target_embedding_size), theano.config.floatX, numpy.random.RandomState(4321))
        self.target_embs=theano.shared(value=target_embs) 
        
        golds=[]
        for i in range(len(self.versions)):
            gold_matrix=mat(itemgetter(*self.overlap_word_list)(self.versions[i].embeddings))
            gold_theano=theano.shared(value=numpy.array(gold_matrix, dtype=theano.config.floatX), borrow=True) 
            golds.append(gold_theano)
        
        
        full_cost_bias=[1.0,8.0,1.0,8.0,1.0]
        cost_bias=[full_cost_bias[index] for index in self.version_indices]
        
        '''
        lbl = mat(itemgetter(*self.overlap_word_list)(self.lbl.embeddings))
        word2vec = mat(itemgetter(*self.overlap_word_list)(self.word2vec.embeddings))
        Huang = mat(itemgetter(*self.overlap_word_list)(self.Huang.embeddings))
        glove = mat(itemgetter(*self.overlap_word_list)(self.glove.embeddings))
        collobert = mat(itemgetter(*self.overlap_word_list)(self.collobert.embeddings))
        
        self.train_lbl=theano.shared(value=numpy.array(lbl, dtype=theano.config.floatX), borrow=True) 
        self.train_word2vec=theano.shared(value=numpy.array(word2vec, dtype=theano.config.floatX), borrow=True) 
        self.train_Huang=theano.shared(value=numpy.array(Huang, dtype=theano.config.floatX), borrow=True) 
        self.train_glove=theano.shared(value=numpy.array(glove, dtype=theano.config.floatX), borrow=True) 
        self.train_collobert=theano.shared(value=numpy.array(collobert, dtype=theano.config.floatX), borrow=True) 
        '''
        

        
        n_train_batches = word_count / batch_size
        if word_count % batch_size !=0:
            train_batch_start=list(numpy.arange(n_train_batches)*batch_size)+[word_count-batch_size]
            n_train_batches=n_train_batches+1
        else:
            train_batch_start=list(numpy.arange(n_train_batches)*batch_size)
        
        index = T.lscalar()    # index to a [mini]batch
        indices=T.ivector('indices')
        
        print '... building the model'
        rng = numpy.random.RandomState(23455)
        
        input_embs=self.target_embs[indices]
        cost_sum=0.0
        L2_reg=0.0
        params=[]
        for i in range(len(self.versions)):
            emb_true=golds[i][indices]
            layer_i = HiddenLayer(rng, input=input_embs, n_out=self.versions[i].dim, n_in=target_embedding_size, activation=None)
            #cost_i=T.mean(T.sum((layer_i.output - emb_true)**2, axis=1))
            
            fake_input_embs=T.dot(layer_i.output, layer_i.W.T)
            cost_i=T.mean(T.sum((layer_i.output - emb_true)**2, axis=1))+T.mean(T.sum((fake_input_embs - input_embs)**2, axis=1))
            
            cost_sum+=cost_i*cost_bias[i] # we consider different weights for cost
            L2_reg+=(layer_i.W**2).sum()
            params=params+layer_i.params

        cost=cost_sum+L2_weight*L2_reg
        output=input_embs
        
        self.params=params+[self.target_embs]
        accumulator=[]
        for para_i in self.params:
            eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
            accumulator.append(theano.shared(eps_p, borrow=True))
         
        # create a list of gradients for all model parameters
        grads = T.grad(cost, self.params)
        updates = []
        for param_i, grad_i, acc_i in zip(self.params, grads, accumulator):
            acc = acc_i + T.sqr(grad_i)
            updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-10)))   #AdaGrad
            updates.append((acc_i, acc))  
            
        train_model = theano.function([index], cost, updates=updates,
              givens={
                indices: train_index_list[index: index + batch_size]})       
        dev_model = theano.function([index], [cost,output],
              givens={
                indices: train_index_list[index: index + batch_size]})   

        ############
        # TRAINING #
        ############
        print '... training'
        self.wait_iter=10
        epoch = 0
        vali_loss_list=[]
        lowest_vali_loss=0
        validation_frequency= n_train_batches
        while (epoch < training_epochs):
            epoch = epoch + 1
            #for minibatch_index in xrange(n_train_batches): # each batch
            minibatch_index=0
            for batch_start in train_batch_start: 
                # iter means how many batches have been runed, taking into loop
                iter = (epoch - 1) * n_train_batches + minibatch_index +1
    
                minibatch_index=minibatch_index+1
                average_cost_per_batch=train_model(batch_start)
                if iter % validation_frequency == 0:
                    print 'training @ iter = '+str(iter)+' cost: '+str(average_cost_per_batch)# +' error: '+str(error_ij)
                    validation_losses=[]            
                    for batch_start in train_batch_start:
                        vali_loss_i, output_i=dev_model(batch_start)
                        validation_losses.append(vali_loss_i)  
                        for row in range(batch_start, batch_start + batch_size):
                            target_embs[row]=output_i[row-batch_start]       
                    this_validation_loss = numpy.mean(validation_losses)
                    print('\t\tepoch %i, minibatch %i/%i, validation cost %f ' % \
                      (epoch, minibatch_index , n_train_batches, \
                       this_validation_loss))
                    
                    if this_validation_loss < (minimal_of_list(vali_loss_list)-5e-3):
                        del vali_loss_list[:]
                        vali_loss_list.append(this_validation_loss)
                        #remember the best params
                        self.best_target_embs=target_embs
                        self.store_target_word_embeddings(self.overlap_word_list, self.best_target_embs, target_embedding_size, 'auto_multi')
                        lowest_vali_loss=this_validation_loss
                    elif len(vali_loss_list)<self.wait_iter:
                        if this_validation_loss < minimal_of_list(vali_loss_list):
                            self.best_target_embs=target_embs
                            self.store_target_word_embeddings(self.overlap_word_list, self.best_target_embs, target_embedding_size, 'auto_multi')
                            lowest_vali_loss=this_validation_loss                            
                        vali_loss_list.append(this_validation_loss)
                        if len(vali_loss_list)==self.wait_iter:
                            #self.store_target_word_embeddings(self.overlap_word_list, self.best_target_embs, target_embedding_size, 'auto_multi')
                            print 'Training over, best word target embeddings got at train_cost:'+str(lowest_vali_loss)+' one2multiple embs stored over.'
                            exit(0)  

    def one2multiple_plus(self, training_epochs=200, batch_size_freq=100, batch_size_rare=5000, learning_rate=0.001, target_embedding_size=300, L2_weight=0.5):
        #lbl, word2vec, Huang, glove, collobert
        word_count=len(self.word_list) # note that word_list contains overlap words in the beginning
        overlap_size=len(self.overlap_vocab)
        print 'word_count:', word_count, 'overlap word_count:', overlap_size
        
        #versions_known_words=[]
        #versions_known_embs=[]
        #versions_unknown_words=[]
        #versions_words=[]
        versions_unknown_embs=[]        
        versions_embs=[]
        #versions_embs_sorted=[]
        #versions_index_sorted_list=[]
        version_index_list_theano=[]
        ini_no=0
        for version in self.versions:
            known_words=version.embeddings.keys()
            unknown_words_list=list(set(self.word_list) ^ set(known_words))
            version_words=known_words+unknown_words_list #concatenate known words and unknown words
            version_words_map=dict(zip(version_words, range(word_count)))

            #versions_words.append(version_words) 
            version_index_sorted=itemgetter(*self.word_list)(version_words_map)

            version_train_index_list=theano.shared(value=numpy.array(version_index_sorted, dtype=theano.config.floatX), borrow=True) 
            version_index_list=T.cast(version_train_index_list, 'int32')	  
            version_index_list_theano.append(version_index_list)

            known_mat=version.embeddings.values()
            known_mat_theano=theano.shared(value=numpy.array(known_mat, dtype=theano.config.floatX), borrow=True) 
            #versions_known_embs.append(known_mat_theano)
            unknown_mat=random_value_normal((len(unknown_words_list), version.dim), theano.config.floatX, numpy.random.RandomState(4321))
            unknown_mat_theano=theano.shared(value=unknown_mat)
            versions_unknown_embs.append(unknown_mat_theano) 
            #concatenate
            versions_embs.append(T.concatenate([known_mat_theano, unknown_mat_theano], axis = 0))

            ini_no+=1
            print 'ini finished... ', ini_no
   

        
        
        self.index_list=range(word_count)
#         self.train_index_list=theano.shared(value=numpy.array(self.index_list, dtype=theano.config.floatX), borrow=True) 
#         train_index_list=T.cast(self.train_index_list, 'int32')
        train_index_list=self.index_list

        
        target_embs=random_value_normal((word_count, target_embedding_size), theano.config.floatX, numpy.random.RandomState(4321))
        self.target_embs=theano.shared(value=target_embs, borrow=True) 
        '''
        golds=[]
        for i in range(len(self.versions)):
            gold_matrix=mat(itemgetter(*self.overlap_word_list)(self.versions[i].embeddings))
            gold_theano=theano.shared(value=numpy.array(gold_matrix, dtype=theano.config.floatX), borrow=True) 
            golds.append(gold_theano)
        '''
        
#         full_cost_bias=[exp(1),exp(4),exp(1),exp(4),exp(1)]
        full_cost_bias=[1,1,1,1,1]
        cost_bias=[full_cost_bias[index] for index in self.version_indices]
        
        '''
        lbl = mat(itemgetter(*self.overlap_word_list)(self.lbl.embeddings))
        word2vec = mat(itemgetter(*self.overlap_word_list)(self.word2vec.embeddings))
        Huang = mat(itemgetter(*self.overlap_word_list)(self.Huang.embeddings))
        glove = mat(itemgetter(*self.overlap_word_list)(self.glove.embeddings))
        collobert = mat(itemgetter(*self.overlap_word_list)(self.collobert.embeddings))
        
        self.train_lbl=theano.shared(value=numpy.array(lbl, dtype=theano.config.floatX), borrow=True) 
        self.train_word2vec=theano.shared(value=numpy.array(word2vec, dtype=theano.config.floatX), borrow=True) 
        self.train_Huang=theano.shared(value=numpy.array(Huang, dtype=theano.config.floatX), borrow=True) 
        self.train_glove=theano.shared(value=numpy.array(glove, dtype=theano.config.floatX), borrow=True) 
        self.train_collobert=theano.shared(value=numpy.array(collobert, dtype=theano.config.floatX), borrow=True) 
        '''
        n_train_batches_freq = overlap_size / batch_size_freq +1
        train_batch_start=list(numpy.arange(n_train_batches_freq)*batch_size_freq)
        used_size=n_train_batches_freq*batch_size_freq
        remain_size=word_count-used_size
               

        
        n_train_batches_rare = remain_size / batch_size_rare
        if remain_size % batch_size_rare !=0:
            train_batch_start_rare=list(numpy.arange(n_train_batches_rare)*batch_size_rare+used_size)+[remain_size-batch_size_rare]
            n_train_batches_rare=n_train_batches_rare+1
        else:
            train_batch_start_rare=list(numpy.arange(n_train_batches_rare)*batch_size_rare+used_size)
        
        n_train_batches=n_train_batches_freq+n_train_batches_rare
        train_batch_start+=train_batch_start_rare
        train_batch_start+=[word_count]
        if n_train_batches != len(train_batch_start)-1:
            print 'n_train_batches != len(train_batch_start)-1', n_train_batches, len(train_batch_start)-1
            exit(0)
        index = T.lscalar()    # index to a [mini]batch
        batch_size=T.iscalar()
        indices=T.ivector('indices')
        
        print '... building the model'
        rng = numpy.random.RandomState(23455)
        
        input_embs=self.target_embs[indices]
#         for i in range(len(self.versions)):
#             emb_true=versions_embs[i][version_index_list_theano[i][index: index + batch_size]]
#             layer_i = HiddenLayer(rng, input=input_embs, n_out=self.versions[i].dim, n_in=target_embedding_size, activation=None)
#             cost_i=T.mean(T.sum((layer_i.output - emb_true)**2, axis=1))
#             cost_sum+=cost_i*cost_bias[i] # we consider different weights for cost
#             L2_reg+=(layer_i.W**2).sum()
#             params=params+layer_i.params

        emb_true_0=versions_embs[0][version_index_list_theano[0][index: index + batch_size]]
        layer_0 = HiddenLayer(rng, input=input_embs, n_out=self.versions[0].dim, n_in=target_embedding_size, activation=None)
        cost_0=T.mean(T.sum((layer_0.output - emb_true_0)**2, axis=1))

        emb_true_1=versions_embs[1][version_index_list_theano[1][index: index + batch_size]]
        layer_1 = HiddenLayer(rng, input=input_embs, n_out=self.versions[1].dim, n_in=target_embedding_size, activation=None)
        cost_1=T.mean(T.sum((layer_1.output - emb_true_1)**2, axis=1))
        
        
        cost_sum=cost_0*cost_bias[0]+cost_1*cost_bias[1]
        L2_reg=(layer_0.W**2).sum()+(layer_1.W**2).sum()
        params=layer_0.params+layer_1.params
        cost=cost_sum+L2_weight*L2_reg
        output=input_embs
        
        self.params=params+[self.target_embs]+versions_unknown_embs
        accumulator=[]
        for para_i in self.params:
            eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
            accumulator.append(theano.shared(eps_p, borrow=True))
         
        # create a list of gradients for all model parameters
        grads = T.grad(cost, self.params)
        updates = []
        for param_i, grad_i, acc_i in zip(self.params, grads, accumulator):
            acc = acc_i + T.sqr(grad_i)
            updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-10)))   #AdaGrad
            updates.append((acc_i, acc))  
            
        train_model = theano.function([index, indices, batch_size], cost, updates=updates)
#                                       ,
#               givens={
#                 indices: train_index_list[index: index + batch_size]})       
        dev_model = theano.function([index, indices, batch_size], output)
#         ,
#               givens={
#                 indices: train_index_list[index: index + batch_size]})   

        ############
        # TRAINING #
        ############
        print '... training'
        self.wait_iter=10
        epoch = 0
        vali_loss_list=[]
        lowest_vali_loss=0
        validation_frequency= n_train_batches
        while (epoch < training_epochs):
            epoch = epoch + 1
            #for minibatch_index in xrange(n_train_batches): # each batch
            minibatch_index=0
            for train_i in range(len(train_batch_start)-1): # do not consider the last one
                if train_i%100==0:
                    print 'train_i:', train_i, 'len(train_batch_start):', len(train_batch_start)-1
#             for batch_start in train_batch_start: 
                # iter means how many batches have been runed, taking into loop
                iter = (epoch - 1) * n_train_batches + minibatch_index +1
                train_batch_size=train_batch_start[train_i+1]-train_batch_start[train_i]
                minibatch_index=minibatch_index+1
                average_cost_per_batch=train_model(train_batch_start[train_i], train_index_list[train_batch_start[train_i]: train_batch_start[train_i+1]], train_batch_size)
                if iter % validation_frequency == 0:
                    print 'training @ iter = '+str(iter)+' cost: '+str(average_cost_per_batch)# +' error: '+str(error_ij)
                    if epoch >=5:
#                         validation_losses=[]      
                        embeddingDict={}      
                        for train_i in range(len(train_batch_start)-1):
#                         for batch_start in train_batch_start:
                            batch_start=train_batch_start[train_i]
                            batch_end=train_batch_start[train_i+1]
                            dev_batch_size=batch_end-batch_start
                            output_i=dev_model(batch_start, train_index_list[batch_start: batch_end], dev_batch_size)
#                             validation_losses.append(vali_loss_i)  
                            for row in range(batch_start, batch_end):
                                target_embs[row]=output_i[row-batch_start]  
                                embeddingDict[self.word_list[row]]=  target_embs[row]   
#                         this_validation_loss = numpy.mean(validation_losses)
                        sp=Simlex999(embeddingDict)
                        print('\t\tepoch %i, minibatch %i/%i, SimLex999 %f ' % \
                          (epoch, minibatch_index , n_train_batches, \
                           sp))
                        
                        if sp < (minimal_of_list(vali_loss_list)-1e-4):
                            del vali_loss_list[:]
                            vali_loss_list.append(sp)
                            #remember the best params
                            self.best_target_embs=target_embs
                            self.store_target_word_embeddings_bigscale(self.word_list, self.best_target_embs, target_embedding_size) # we do not instore each time, on order to make the training faster
                            #self.store_OOV_embs_for_version()#store the OOV embs for each constitute embedding version
                            lowest_vali_loss=sp
                        elif len(vali_loss_list)<self.wait_iter:
                            if sp < minimal_of_list(vali_loss_list):
                                self.best_target_embs=target_embs
                                self.store_target_word_embeddings_bigscale(self.word_list, self.best_target_embs, target_embedding_size)
                                lowest_vali_loss=sp                            
                            vali_loss_list.append(sp)
                            if len(vali_loss_list)==self.wait_iter:
                                #self.store_target_word_embeddings_bigscale(self.word_list, self.best_target_embs, target_embedding_size)
                                print 'Training over, best word target embeddings got at SimLex999:'+str(lowest_vali_loss)+' one2multiple_plus embs stored over.'
                                exit(0)  

    def autoencoder_fake(self, training_epochs=200, batch_size=100, learning_rate=0.001, target_embedding_size=300, L2_weight=0.0005):
        #first construct the matrix data
        word_list=[]
        word_count=len(self.concatenate_embs)
        init_embs_matrix=random_value_normal((word_count, self.overall_dim), theano.config.floatX, numpy.random.RandomState(1234))

        word_index=0
        for word, emb in self.concatenate_embs.iteritems():
            word_list.append(word)
            init_embs_matrix[word_index]=numpy.array(emb)
            word_index+=1
        self.train_data=theano.shared(value=numpy.array(init_embs_matrix, dtype=theano.config.floatX), borrow=True) 
    
        # compute number of minibatches for training, validation and testing
        n_train_batches = word_count / batch_size
        if word_count % batch_size !=0:
            train_batch_start=list(numpy.arange(n_train_batches)*batch_size)+[word_count-batch_size]
            n_train_batches=n_train_batches+1
        else:
            train_batch_start=list(numpy.arange(n_train_batches)*batch_size)
        
        target_embs=random_value_normal((word_count, target_embedding_size), theano.config.floatX, numpy.random.RandomState(4321))
        self.target_embs=theano.shared(value=target_embs) 
        
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as rasterized images

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
    
        rng = numpy.random.RandomState(23455)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
        '''
        da = dA_random_input(numpy_rng=rng, theano_rng=theano_rng, input=x,
                n_visible=self.overall_dim, n_hidden=target_embedding_size)
    
        cost, updates = da.get_cost_updates(corruption_level=0.,
                                            learning_rate=learning_rate)
        '''
        input_embs=self.target_embs[index: index + batch_size]
        layer_i = HiddenLayer(rng, input=input_embs, n_out=self.overall_dim, n_in=target_embedding_size, activation=None)
            
        fake_input_embs=T.dot(layer_i.output, layer_i.W.T)
        L2_reg=(layer_i.W**2).sum()
        cost=T.mean(T.sum((layer_i.output - x)**2, axis=1))+T.mean(T.sum((fake_input_embs - input_embs)**2, axis=1))+L2_weight*L2_reg
        
        output=input_embs
        self.params=layer_i.params+[self.target_embs]        
        
        accumulator=[]
        for para_i in self.params:
            eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
            accumulator.append(theano.shared(eps_p, borrow=True))
         
        # create a list of gradients for all model parameters
        grads = T.grad(cost, self.params)
        updates = []
        for param_i, grad_i, acc_i in zip(self.params, grads, accumulator):
            acc = acc_i + T.sqr(grad_i)
            updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-10)))   #AdaGrad
            updates.append((acc_i, acc)) 
            
            
            
                    
        train_da = theano.function([index], cost, updates=updates,
              givens={
                x: self.train_data[index: index + batch_size]})
        dev_da = theano.function([index], [cost, output],
              givens={
                x: self.train_data[index: index + batch_size]})
           
                    
        ############
        # TRAINING #
        ############
        print '... training'
        self.wait_iter=10
        epoch = 0
        vali_loss_list=[]
        lowest_vali_loss=0
        validation_frequency= n_train_batches
        while (epoch < training_epochs):
            epoch = epoch + 1
            #for minibatch_index in xrange(n_train_batches): # each batch
            minibatch_index=0
            for batch_start in train_batch_start: 
                # iter means how many batches have been runed, taking into loop
                iter = (epoch - 1) * n_train_batches + minibatch_index +1
    
                minibatch_index=minibatch_index+1
                average_cost_per_batch=train_da(batch_start)
                if iter % validation_frequency == 0:
                    print 'training @ iter = '+str(iter)+' cost: '+str(average_cost_per_batch)# +' error: '+str(error_ij)
                    validation_losses=[]            
                    for batch_start in train_batch_start:
                        vali_loss_i, output_i=dev_da(batch_start)
                        validation_losses.append(vali_loss_i)  
                        for row in range(batch_start, batch_start + batch_size):
                            target_embs[row]=output_i[row-batch_start]       
                    this_validation_loss = numpy.mean(validation_losses)
                    print('\t\tepoch %i, minibatch %i/%i, validation cost %f ' % \
                      (epoch, minibatch_index , n_train_batches, \
                       this_validation_loss))
                    
                    if this_validation_loss < (minimal_of_list(vali_loss_list)-1e-4):
                        del vali_loss_list[:]
                        vali_loss_list.append(this_validation_loss)
                        #remember the best params
                        self.best_target_embs=target_embs
                        lowest_vali_loss=this_validation_loss
                        self.store_target_word_embeddings(word_list, self.best_target_embs, target_embedding_size, 'auto')
                    elif len(vali_loss_list)<self.wait_iter:
                        if this_validation_loss < minimal_of_list(vali_loss_list):
                            self.best_target_embs=target_embs
                            lowest_vali_loss=this_validation_loss                            
                        vali_loss_list.append(this_validation_loss)
                        self.store_target_word_embeddings(word_list, self.best_target_embs, target_embedding_size, 'auto')
                        if len(vali_loss_list)==self.wait_iter:
                            #self.store_target_word_embeddings(word_list, self.best_target_embs, target_embedding_size,'auto')
                            print 'Training over, best word target embeddings got at train_cost:'+str(lowest_vali_loss)+' autoencoded embs stored over.'
                            exit(0) 

    def autoencoder(self, training_epochs=200, batch_size=100, learning_rate=0.001, target_embedding_size=300, corruption_level=0.0):
        #first construct the matrix data
        word_list=[]
        word_count=len(self.concatenate_embs)
        init_embs_matrix=random_value_normal((word_count, self.overall_dim), theano.config.floatX, numpy.random.RandomState(1234))
        target_embs=numpy.zeros((word_count, target_embedding_size), dtype=theano.config.floatX)
        word_index=0
        for word, emb in self.concatenate_embs.iteritems():
            word_list.append(word)
            init_embs_matrix[word_index]=numpy.array(emb)
            word_index+=1
        self.train_data=theano.shared(value=numpy.array(init_embs_matrix, dtype=theano.config.floatX), borrow=True) 
    
        # compute number of minibatches for training, validation and testing
        n_train_batches = word_count / batch_size
        if word_count % batch_size !=0:
            train_batch_start=list(numpy.arange(n_train_batches)*batch_size)+[word_count-batch_size]
            n_train_batches=n_train_batches+1
        else:
            train_batch_start=list(numpy.arange(n_train_batches)*batch_size)
    
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as rasterized images

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
    
        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
    
        da = dA_random_input(numpy_rng=rng, theano_rng=theano_rng, input=x,
                n_visible=self.overall_dim, n_hidden=target_embedding_size)
    
        cost, updates = da.get_cost_updates(corruption_level=0.,
                                            learning_rate=learning_rate)

        train_da = theano.function([index], cost, updates=updates,
              givens={
                x: self.train_data[index: index + batch_size]})
        dev_da = theano.function([index], [cost, da.output],
              givens={
                x: self.train_data[index: index + batch_size]})
           
                    
        ############
        # TRAINING #
        ############
        print '... training'
        self.wait_iter=10
        epoch = 0
        vali_loss_list=[]
        lowest_vali_loss=0
        validation_frequency= n_train_batches
        while (epoch < training_epochs):
            epoch = epoch + 1
            #for minibatch_index in xrange(n_train_batches): # each batch
            minibatch_index=0
            for batch_start in train_batch_start: 
                # iter means how many batches have been runed, taking into loop
                iter = (epoch - 1) * n_train_batches + minibatch_index +1
    
                minibatch_index=minibatch_index+1
                average_cost_per_batch=train_da(batch_start)
                if iter % validation_frequency == 0:
                    print 'training @ iter = '+str(iter)+' cost: '+str(average_cost_per_batch)# +' error: '+str(error_ij)
                    validation_losses=[]            
                    for batch_start in train_batch_start:
                        vali_loss_i, output_i=dev_da(batch_start)
                        validation_losses.append(vali_loss_i)  
                        for row in range(batch_start, batch_start + batch_size):
                            target_embs[row]=output_i[row-batch_start]       
                    this_validation_loss = numpy.mean(validation_losses)
                    print('\t\tepoch %i, minibatch %i/%i, validation cost %f ' % \
                      (epoch, minibatch_index , n_train_batches, \
                       this_validation_loss))
                    
                    if this_validation_loss < (minimal_of_list(vali_loss_list)-5e-3):
                        del vali_loss_list[:]
                        vali_loss_list.append(this_validation_loss)
                        #remember the best params
                        self.best_target_embs=target_embs
                        lowest_vali_loss=this_validation_loss
                        self.store_target_word_embeddings(word_list, self.best_target_embs, target_embedding_size, 'auto')
                    elif len(vali_loss_list)<self.wait_iter:
                        if this_validation_loss < minimal_of_list(vali_loss_list):
                            self.best_target_embs=target_embs
                            lowest_vali_loss=this_validation_loss                            
                        vali_loss_list.append(this_validation_loss)
                        self.store_target_word_embeddings(word_list, self.best_target_embs, target_embedding_size, 'auto')
                        if len(vali_loss_list)==self.wait_iter:
                            #self.store_target_word_embeddings(word_list, self.best_target_embs, target_embedding_size,'auto')
                            print 'Training over, best word target embeddings got at train_cost:'+str(lowest_vali_loss)+' autoencoded embs stored over.'
                            exit(0) 
    
    def eachSlice(self, slice, tensor):
        print 'slice..', slice
        overall_dim=self.lbl.dim+self.word2vec.dim+self.Huang.dim+self.glove.dim+self.collobert.dim
        rows=[]
        cols=[]
        data=[]
        for row, row_word in enumerate(self.word_list):
            for col, col_word in enumerate(self.word_list):
                if row!=col:
                    rows.append(row)
                    cols.append(col)
                    data.append(cosVector(self.versions[slice].embeddings[row_word], self.versions[slice].embeddings[col_word]))
        tensor[slice]=csr_matrix((data, (rows, cols)), shape=(overall_dim, overall_dim))
                    
                    
        
    def tensor_factorization(self, dimen, word_size, top_neighbors):
        #load 5 versions
        
        #5 threads to compute cosine similarities
        tensor = [0] * 5
        
        threads = []
        for i, m in enumerate(self.versions): #index, version
    
            t = threading.Thread(target=self.eachSlice_row_wise_cosine, args=(i, tensor, word_size, top_neighbors))
            t.start()
            threads.append(t)
    
        for t in threads:
            t.join()
    
        # tensor is ready
        print 'Calling rescal...'
        logging.basicConfig(level=logging.INFO)
        #A, R, fit, itr, exectimes = rescal(tensor, 50, init='nvecs', lambda_A=10, lambda_R=10, compute_fit=True)
        A, R, fit, itr, exectimes = rescal.rescal(tensor, dimen, lmbda=0)
        printWordEmbedding(A, self.word_list[:word_size], dimen, False)
        printRelation(R, dimen, False)          

    def tensor_factorization_overlap(self, dimen, word_size, top_neighbors):
        #load 5 versions
        
        #5 threads to compute cosine similarities
        tensor = [0] * 5
        
        threads = []
        for i, m in enumerate(self.versions): #index, version
    
            t = threading.Thread(target=self.eachSlice_row_wise_cosine_overlap, args=(i, tensor, word_size, top_neighbors))
            t.start()
            threads.append(t)
    
        for t in threads:
            t.join()
    
        # tensor is ready
        print 'Calling rescal...'
        logging.basicConfig(level=logging.INFO)
        #A, R, fit, itr, exectimes = rescal(tensor, 50, init='nvecs', lambda_A=10, lambda_R=10, compute_fit=True)
        A, R, fit, itr, exectimes = rescal.rescal(tensor, dimen, lmbda=0)
        printWordEmbedding(A, self.overlap_word_list[:word_size], dimen, True)
        printRelation(R, dimen, True)  


    def eachSlice_row_wise_cosine_overlap(self, slice, tensor, word_size, top_neighbors):
        print 'slice..', slice
        embeddings_map=self.versions[slice].embeddings
        myvalues = itemgetter(*self.overlap_word_list[:word_size])(embeddings_map)
        matrix=mat(myvalues)
        norm0=numpy.apply_along_axis(numpy.linalg.norm, 1, matrix)
        words=matrix.shape[0]
        rows=[]
        cols=[]
        data=[]        
        for i in range(words):
            simi_i=dot(matrix[i], matrix.T)
            simi_i_list=numpy.array(simi_i)[0]
            top_indices=sorted(numpy.argsort(simi_i_list)[-top_neighbors:]) # only consider top 1000 similar words     
            for j in top_indices:
                if simi_i_list[j]>0.0:
                    rows.append(i)
                    cols.append(j)
                    data.append(simi_i_list[j]/(norm0[i]*norm0[j]))
        tensor[slice]=csr_matrix((data, (rows, cols)), shape=(words, words))            
    def eachSlice_row_wise_cosine(self, slice, tensor, word_size, top_neighbors):
        print 'slice..', slice
        embeddings_map=self.versions[slice].embeddings
        myvalues = itemgetter(*self.word_list[:word_size])(embeddings_map)
        matrix=mat(myvalues)
        norm0=numpy.apply_along_axis(numpy.linalg.norm, 1, matrix)
        words=matrix.shape[0]
        rows=[]
        cols=[]
        data=[]        
        for i in range(words):
            simi_i=dot(matrix[i], matrix.T)
            simi_i_list=numpy.array(simi_i)[0]
            top_indices=sorted(numpy.argsort(simi_i_list)[-top_neighbors:]) # only consider top 1000 similar words     
            for j in top_indices:
                if simi_i_list[j]>0.0:
                    rows.append(i)
                    cols.append(j)
                    data.append(simi_i_list[j]/(norm0[i]*norm0[j]))
        tensor[slice]=csr_matrix((data, (rows, cols)), shape=(words, words))
        '''
        print 'slice..', slice
        #first put versions[slice] into a matrix
        #matrix_list=[]
        embeddings_map=self.versions[slice].embeddings
        myvalues = itemgetter(*self.word_list[:100000])(embeddings_map)
        matrix=mat(myvalues)
        #next, compute row-wise cosine
        cosine_matrix=row_wise_cosine(matrix)
        numpy.fill_diagonal(cosine_matrix, 0.0)
        tensor[slice]=csr_matrix(cosine_matrix)
        '''
        '''
        rows=[]
        cols=[]
        data=[]
        for row, row_word in enumerate(self.word_list[:50]):
            for col, col_word in enumerate(self.word_list[:50]):
                if row!=col and cosine_matrix[row,col]>0.0:   # can not use cosine_matrix[row][]
                    rows.append(row)
                    cols.append(col)
                    data.append(cosine_matrix[row,col])
        tensor[slice]=csr_matrix((data, (rows, cols)), shape=(50, 50))
        '''    
    def NMF(self):
        #this uses the matlab function to do NMF, first need to output the required format
        input_file_for_matlab=open('/mounts/data/proj/wenpeng/Emb_Extend/input_file_for_matlab.txt', 'w')
        word_list_for_matlab=open('/mounts/data/proj/wenpeng/Emb_Extend/word_list_for_matlab.txt', 'w')
        for index, word in enumerate(self.word_list):
            emb=self.concatenate_embs[word]
            for j in range(self.overall_dim):
                input_file_for_matlab.write(str(index+1)+'\t'+str(j+1)+'\t'+str(emb[j])+'\n') #index starts from 1 in matlab
            word_list_for_matlab.write(word+'\n')
        print 'writed over.'
        input_file_for_matlab.close()
        word_list_for_matlab.close() 
    '''
    def SVD(self):
        #this uses the matlab function to do NMF, first need to output the required format
        input_file_for_matlab=open('/mounts/data/proj/wenpeng/Emb_Extend/input_file_for_matlab.txt', 'w')
        word_list_for_matlab=open('/mounts/data/proj/wenpeng/Emb_Extend/word_list_for_matlab.txt', 'w')
        for index, word in enumerate(self.word_list):
            emb=self.concatenate_embs[word]
            for j in range(self.overall_dim):
                input_file_for_matlab.write(str(index+1)+'\t'+str(j+1)+'\t'+str(emb[j])+'\n') #index starts from 1 in matlab
            word_list_for_matlab.write(word+'\n')
        print 'writed over.'
        input_file_for_matlab.close()
        word_list_for_matlab.close()   
    '''  
    def SVD(self, dim, only_U):
        model_options = locals().copy()
        print "model options", model_options
        suffix=''
        for index in self.version_indices:
            suffix+='_'+str(index)
            
        emb_list=[]
        for index, word in enumerate(self.word_list):
            emb=self.concatenate_embs[word]
            emb_list.append(emb)
        input_matrix=array(emb_list)
        U, S, V=linalg.svd(input_matrix)
        write_filename='/mounts/data/proj/wenpeng/Emb_Extend/results_L2norm_forall/svd_'+str(only_U)+'_d'+str(dim)+suffix+'.txt'
        write_file=open(write_filename, 'w')
        print dim, len(U[0])
        for index, word in enumerate(self.word_list):
            write_file.write(word+'\t')
            for d in range(dim):
                if only_U:
                    write_file.write(str(U[index][d])+' ')
                else:
                    write_file.write(str(U[index][d]*S[d])+' ')
            write_file.write('\n')
        write_file.close()
        print 'SVD results stored over.'
                            
def minimal_of_list(list_of_ele):
    if len(list_of_ele) ==0:
        return 1e10
    else:
        return min(list_of_ele)                            

def printRelation(R, dimen, overlap_flag):
    file_name='/mounts/data/proj/wenpeng/Emb_Extend/relations_'+str(dimen)+'.txt'
    if overlap_flag:
        file_name='/mounts/data/proj/wenpeng/Emb_Extend/relations_overlap_'+str(dimen)+'.txt'
    with file(file_name, 'w') as outfile:
        for i in xrange(len(R)):
            savetxt(outfile, R[i])
    print 'Relations via tensor factorization are stored over!'

def printWordEmbedding(matrix, word_list, dimen, overlap_flag):
    file_name='/mounts/data/proj/wenpeng/Emb_Extend/ensamble_tensor_factorization_'+str(dimen)+'.txt'
    if overlap_flag:
        file_name='/mounts/data/proj/wenpeng/Emb_Extend/ensamble_tensor_factorization_overlap_'+str(dimen)+'.txt'
    output= open(file_name, 'w')
    for i, word in enumerate(word_list):
        output.write(word+'\t') #this means word printed according id
        for length in range(dimen):
            output.write(str(matrix[i][length])+' ')
        output.write('\n')
    print 'Word embeddings via tensor factorization are stored over!'    
    output.close()                        

def row_wise_cosine(x):
    #the input x is a matrix
    rows=x.shape[0]
    norm0=numpy.apply_along_axis(numpy.linalg.norm, 1, x)
    norm1=norm0.reshape(rows, 1)
    norm2=norm0.reshape(1, rows)    
    norm3=numpy.multiply(norm1, norm2)    
    
    simi=dot(x,x.T)/norm3
    simi[simi<0.0]=0.0
    return simi

def row_wise_cosine_theano(x):
    #the input x is a matrix
    rows=x.shape[0]
    sum=T.sum(T.sqr(x),axis=1)  
    norm0=T.sqrt(sum)
    norm1=debug_print(norm0.reshape((rows, 1)), 'norm1')
    norm2=debug_print(norm0.reshape((1, rows)), 'norm2')    
    norm3=debug_print(T.dot(norm1,norm2), 'norm3')
    simi=T.dot(x,x.transpose(1, 0))/norm3
    return simi     
    
    
    
def run(arg):    

    '''
    instance=Ensemble(compose_style=2)    
    #instance.concatenate()
    instance.autoencoder(training_epochs=2000, batch_size=1000, learning_rate=1e-3, target_embedding_size=300)
    '''
    '''
    #tensor factorization into 300d
    instance=Ensemble(compose_style=1)
    instance.tensor_factorization(300, 1000000, 1000) # dimen, word_size (1338447), top_neighbors  
    '''
    '''
    #NMF
    instance=Ensemble(compose_style=2)
    instance.NMF()
    '''
    '''
    instance=Ensemble(compose_style=1)    
    #instance.concatenate()
    instance.map2one(training_epochs=5000, batch_size=1000, learning_rate=1e-3, target_embedding_size=300, cost_ratio=0.5)
    '''
    '''
    #overlap
    
    instance=Ensemble(compose_style=3, version_indices=[0,1,2,3,4], weight=float(arg[0]))    
    instance.concatenate_overlap(weight=float(arg[0]))   #concatenate is only useful for "concatenated embedding", 'autoencoder', 'svd' 
    '''
    '''
    instance=Ensemble(compose_style=3)   
    instance.tensor_factorization_overlap(300, None, 1000)
    '''
    '''
    instance=Ensemble(compose_style=4, version_indices=[0,1,2,3])
    instance.SVD(dim=int(arg[0]), only_U=True)    
    '''
    '''
    instance=Ensemble(compose_style=4, version_indices=[0,1,3,4])    
    instance.autoencoder(training_epochs=2000, batch_size=200, learning_rate=1e-2, target_embedding_size=int(arg[0]), corruption_level=0.2)
    '''
    '''
    instance=Ensemble(compose_style=3)    
    instance.map2one(training_epochs=5000, batch_size=200, learning_rate=1e-3, target_embedding_size=300, cost_ratio=0.5)   
    '''
    '''
    instance=Ensemble(compose_style=3,version_indices=[0,1,2,3])    
    instance.one2multiple(training_epochs=5000, batch_size=200, learning_rate=5e-3, target_embedding_size=int(arg[0]), L2_weight=0.0005)     
    '''
    
    instance=Ensemble(compose_style=5,version_indices=[1,3])    
    instance.one2multiple_plus(training_epochs=5000, learning_rate=5e-3, target_embedding_size=200, L2_weight=0.0005)  
    
    
#     instance=Ensemble(compose_style=3,version_indices=[0,1,2,3,4])    
#     instance.auto_multi(training_epochs=5000, batch_size=200, learning_rate=1e-2, target_embedding_size=int(arg[0]), L2_weight=0.0005)   #note it needs a parameter  
#     
    
if __name__ == '__main__':
    run(sys.argv[1:])
    