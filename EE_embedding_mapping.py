
import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from WPDefined import Full_Convolayer,Conv_Fold_DynamicK_PoolLayer, dropout_from_layer, shared_dataset, load_model_for_training, SoftMaxlayer
from word2embeddings.nn.layers import BiasedHiddenLayer, SerializationLayer, \
    IndependendAttributesLoss, SquaredErrorLossLayer
from word2embeddings.nn.util import zero_value, random_value_normal, \
    random_value_GloBen10
from word2embeddings.tools.theano_extensions import MRG_RandomStreams2
from cis.deep.utils.theano import debug_print
from EE_read_versions import load_versions


class MAPPING(object):
    def __init__(self, learning_rate=0.2, n_epochs=2000, batch_size=10, useAllSamples=0, 
                    L2_weight=0.00005, vali_cost_list_length=20, source_embedding_size=48, target_embedding_size=48, version1=10, version2=10):

        self.version1=version1 #target
        self.version2=version2  #source
        self.ini_learning_rate=learning_rate
        self.n_epochs=n_epochs
        #self.nkerns=nkerns
        self.batch_size=batch_size
        self.useAllSamples=useAllSamples
        self.L2_weight=L2_weight
        self.embedding_size=0
        #self.train_scheme=train_scheme

        self.vali_cost_list_length=vali_cost_list_length
        #self.newd=newd
        train_source, train_target, dev_source, dev_target, test_source=self.creat_data()
        
        self.raw_data=[train_source, dev_source, test_source]
        
        self.train_source=theano.shared(value=numpy.array(train_source, dtype=theano.config.floatX), borrow=True) 
        self.train_target=theano.shared(value=numpy.array(train_target, dtype=theano.config.floatX), borrow=True)  
        self.dev_source=theano.shared(value=numpy.array(dev_source, dtype=theano.config.floatX), borrow=True)  
        self.dev_target=theano.shared(value=numpy.array(dev_target, dtype=theano.config.floatX), borrow=True)  
        self.test_source=theano.shared(value=numpy.array(test_source, dtype=theano.config.floatX), borrow=True)         
        
        self.source_embedding_size=source_embedding_size
        self.target_embedding_size=target_embedding_size
        
    def creat_data(self):
        overlap_vocab=list(self.version1.vocab & self.version2.vocab)
        self.OOV=list(self.version2.vocab-self.version1.vocab)  # words in version2 but not in version1
        #n_batches=len(overlap_vocab)/self.batch_size
        train_source=[]
        train_target=[]
        dev_source=[]
        dev_target=[]
        train_size=len(overlap_vocab)/6  # train:dev=5:1
        count=0
        #split overlap vocab into six parts, five for train, one for dev
        for word in  overlap_vocab:
            if count < train_size:
                train_source.append(self.version2.embeddings[word])
                train_target.append(self.version1.embeddings[word])
            else:
                dev_source.append(self.version2.embeddings[word])
                dev_target.append(self.version1.embeddings[word])      
            count+=1
        test_source=[]
        for word in self.OOV:
            test_source.append(self.version2.embeddings[word])    
        return  numpy.array(train_source), numpy.array(train_target), numpy.array(dev_source), numpy.array(dev_target), numpy.array(test_source)
 
 
    def evaluate_lenet5(self):
    #def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, nkerns=[6, 12], batch_size=70, useAllSamples=0, kmax=30, ktop=5, filter_size=[10,7],
    #                    L2_weight=0.000005, dropout_p=0.5, useEmb=0, task=5, corpus=1):
        rng = numpy.random.RandomState(23455)


        n_train_batches=self.raw_data[0].shape[0]/self.batch_size
        n_valid_batches=self.raw_data[1].shape[0]/self.batch_size
        n_test_batches=self.raw_data[2].shape[0]/self.batch_size

        train_batch_start=[]
        dev_batch_start=[]
        test_batch_start=[]
        if self.useAllSamples:
            train_batch_start=list(numpy.arange(n_train_batches)*self.batch_size)+[self.raw_data[0].shape[0]-self.batch_size]
            dev_batch_start=list(numpy.arange(n_valid_batches)*self.batch_size)+[self.raw_data[1].shape[0]-self.batch_size]
            test_batch_start=list(numpy.arange(n_test_batches)*self.batch_size)+[self.raw_data[2].shape[0]-self.batch_size]
            n_train_batches=n_train_batches+1
            n_valid_batches=n_valid_batches+1
            n_test_batches=n_test_batches+1
        else:
            train_batch_start=list(numpy.arange(n_train_batches)*self.batch_size)
            dev_batch_start=list(numpy.arange(n_valid_batches)*self.batch_size)
            test_batch_start=list(numpy.arange(n_valid_batches)*self.batch_size)
        '''
        indices_train_theano=theano.shared(numpy.asarray(indices_train, dtype=theano.config.floatX), borrow=True)
        indices_dev_theano=theano.shared(numpy.asarray(indices_dev, dtype=theano.config.floatX), borrow=True)
        indices_train_theano=T.cast(indices_train_theano, 'int32')
        indices_dev_theano=T.cast(indices_dev_theano, 'int32')
        '''
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        x = T.dmatrix('x')   # now, x is the index matrix, must be integer
        y = T.dmatrix('y') 

        
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
    
        inputs=debug_print(x, 'inputs')
        labels=debug_print(y, 'labels')
        layer2 = HiddenLayer(rng, input=inputs, n_in=self.source_embedding_size, n_out=self.target_embedding_size, activation=None)
        layer2_output=debug_print(layer2.output, 'layer2_output')
        #J= debug_print(- T.sum(labels * T.log(layer2_output) + (1 - labels) * T.log(1 - layer2_output), axis=1), 'J') # a vector of cross-entropy
        J=T.sum((layer2_output - labels)**2, axis=1)
        L2_reg = (layer2.W** 2).sum()
        self.cost = T.mean(J) + self.L2_weight*L2_reg
        
        validate_model = theano.function([index], self.cost,
                givens={
                    x: self.dev_source[index: index + self.batch_size],
                    y: self.dev_target[index: index + self.batch_size]})

        test_model = theano.function([index], layer2_output,
                givens={
                    x: self.test_source[index: index + self.batch_size],
                    y: self.test_source[index: index + self.batch_size]})   
        # create a list of all model parameters to be fit by gradient descent
        self.params = layer2.params
        #params = layer3.params + layer2.params + layer0.params+[embeddings]
        
        accumulator=[]
        for para_i in self.params:
            eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
            accumulator.append(theano.shared(eps_p, borrow=True))
          
        # create a list of gradients for all model parameters
        grads = T.grad(self.cost, self.params)
        updates = []
        for param_i, grad_i, acc_i in zip(self.params, grads, accumulator):
            acc = acc_i + T.sqr(grad_i)
            updates.append((param_i, param_i - self.ini_learning_rate * grad_i / T.sqrt(acc)))   #AdaGrad
            updates.append((acc_i, acc))    
           
        train_model = theano.function([index], self.cost, updates=updates,
              givens={
                x: self.train_source[index: index + self.batch_size],
                y: self.train_target[index: index + self.batch_size]})
    
        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        # early-stopping parameters
        patience = 500000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        best_params = None
        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()
    
        epoch = 0
        done_looping = False
        vali_loss_list=[]
        lowest_vali_loss=0
        OOV_embs=numpy.zeros((len(self.OOV),self.target_embedding_size), dtype=theano.config.floatX)
        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            #for minibatch_index in xrange(n_train_batches): # each batch
            minibatch_index=0
            for batch_start in train_batch_start: 
                # iter means how many batches have been runed, taking into loop
                iter = (epoch - 1) * n_train_batches + minibatch_index +1
    
                minibatch_index=minibatch_index+1
                
                cost_of_each_iteration= train_model(batch_start)
                #exit(0)
                #print 'sentence embeddings:'
                #print sentences_embs[:6,:]
                #if iter ==1:
                #    exit(0)
                if iter % validation_frequency == 0:
                    print 'training @ iter = '+str(iter)+' cost: '+str(cost_of_each_iteration)# +' error: '+str(error_ij)
                if iter % validation_frequency == 0:
                    #print '\t iter: '+str(iter)
                    # compute zero-one loss on validation set
                    #validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                    validation_losses=[]
                    for batch_start in dev_batch_start:
                        vali_loss_i=validate_model(batch_start)
                        validation_losses.append(vali_loss_i)
                    this_validation_loss = numpy.mean(validation_losses)
                    print('\t\tepoch %i, minibatch %i/%i, validation cost %f ' % \
                      (epoch, minibatch_index , n_train_batches, \
                       this_validation_loss))
                    
                    if this_validation_loss < (minimal_of_list(vali_loss_list)-1e-4): #is very small
                        #print str(minimal_of_list(vali_loss_list))+'-'+str(this_validation_loss)+'='+str(minimal_of_list(vali_loss_list)-this_validation_loss)
                        del vali_loss_list[:]
                        vali_loss_list.append(this_validation_loss)
                        lowest_vali_loss=this_validation_loss
                        #store params
                        self.best_params=self.params
                        for batch_start in test_batch_start:
                            predicted_embeddings=test_model(batch_start)
                            for row in range(batch_start, batch_start + self.batch_size):
                                OOV_embs[row]=predicted_embeddings[row-batch_start]
                        if len(vali_loss_list)==self.vali_cost_list_length: # only happen when self.vali_cost_list_length==1
                            print 'Training over, best model got at vali_cost:'+str(lowest_vali_loss)
                            return OOV_embs, self.OOV
                    elif len(vali_loss_list)<self.vali_cost_list_length:                        
                        if this_validation_loss < minimal_of_list(vali_loss_list): #if it's small, but not small enough
                            self.best_params=self.params
                            lowest_vali_loss=this_validation_loss
                            for batch_start in test_batch_start:
                                predicted_embeddings=test_model(batch_start)
                                for row in range(batch_start, batch_start + self.batch_size):
                                    OOV_embs[row]=predicted_embeddings[row-batch_start]   
                        vali_loss_list.append(this_validation_loss)                         
                        if len(vali_loss_list)==self.vali_cost_list_length:
                            print 'Training over, best model got at vali_cost:'+str(lowest_vali_loss)
                            return OOV_embs, self.OOV
                    #print vali_loss_list
    
    
                if patience <= iter:
                    done_looping = True
                    break
    
        end_time = time.clock()
        '''
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i,'\
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        '''
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))



def minimal_of_list(list_of_ele):
    if len(list_of_ele) ==0:
        return 1e10
    else:
        return min(list_of_ele)


def inter_map(average):
    lbl, word2vec, Huang, glove, collobert, overall_vocab, overlap_vocab=load_versions()#need to change
    version_list=[lbl, word2vec, Huang, glove, collobert]
    for i in range(len(version_list)):
        OOV_emb_list=[]
        for j in range(len(version_list)):
            if j!=i:                        
                # from version j project to version i    
                network=MAPPING(learning_rate=0.001, n_epochs=4000, batch_size=200, useAllSamples=1, 
                                L2_weight=0.00005, vali_cost_list_length=10, source_embedding_size=version_list[j].dim, 
                                target_embedding_size=version_list[i].dim, version1=version_list[i], version2=version_list[j])
                embeddings, words=network.evaluate_lenet5()
                #store into file
                #write_file=open('/mounts/data/proj/wenpeng/Emb_Extend/')
                #store into a map
                OOV_emb={}
                index=0
                for word in words:
                    OOV_emb[word]=embeddings[index]
                    index+=1  
                OOV_emb_list.append(OOV_emb)  
        i_overall_OOV=overall_vocab-version_list[i].vocab
        OOV_file=open('/mounts/data/proj/wenpeng/Emb_Extend/'+str(i)+'_OOV_embs.txt', 'w')
        if average:
            for word in i_overall_OOV:
                #print 'Considering word: '+word
                predicts=[]
                for embs in OOV_emb_list:
                    emb=embs.get(word)
                    if emb is not None:
                        predicts.append(emb)
                if len(predicts)==0:
                    print 'error: word '+word+' find no predicts.'
                    no_source=True
                    for j in range(len(version_list)):
                        if j!=i:
                            if version_list[j].embeddings.get(word, 0) !=0:
                                no_source=False
                                print word+' finds source embedding from '+str(j)
                    if no_source:
                        print word+' finds no source embeddings.'
                    exit(0)
                averge_emb=numpy.average(numpy.array(predicts), axis=0)   
                #print 'average_emb:'
                #print averge_emb
                OOV_file.write(word+'\t')
                for dim in range(version_list[i].dim):
                    OOV_file.write(str(averge_emb[dim])+' ')
                OOV_file.write('\n')
            OOV_file.close()
            print str(i)+' version stored all OOV embs.'
        else: #select randomly
            for word in i_overall_OOV:
                predicts=[]
                for embs in OOV_emb_list:
                    emb=embs.get(word)
                    if emb is not None:
                        predicts.append(emb)
                random_emb=predicts[numpy.random.randint(len(predicts))]
                OOV_file.write(word+'\t')
                for dim in range(version_list[i].dim):
                    OOV_file.write(str(random_emb[dim])+' ')
                OOV_file.write('\n')
            OOV_file.close()
            print str(i)+' version stored all OOV embs.'            
            
            
    

if __name__ == '__main__':
    #i guess this file does the mapping for vocabulary extensionl
    inter_map(average=True)

