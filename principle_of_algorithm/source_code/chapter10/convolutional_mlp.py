# -*- coding: utf-8 -*-

"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

""" 
���+�²����ϳ�һ����LeNetConvPoolLayer 
rng:����������������ڳ�ʼ��W 
input:4ά��������theano.tensor.dtensor4 
filter_shape:(number of filters, num input feature maps,filter height, filter width) 
image_shape:(batch size, num input feature maps,image height, image width) 
poolsize: (#rows, #cols) 
"""  
class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        #assert condition��conditionΪTrue�����������ִ�У�conditionΪFalse���жϳ���  
        #image_shape[1]��filter_shape[1]����num input feature maps�����Ǳ�����һ���ġ�  
        assert image_shape[1] == filter_shape[1]
        self.input = input    
        #ÿ��������Ԫ�������أ�����һ���������Ϊnum input feature maps * filter height * filter width��  
        #������numpy.prod(filter_shape[1:])�����
        fan_in = numpy.prod(filter_shape[1:])
        #lower layer��ÿ����Ԫ��õ��ݶ������ڣ�"num output feature maps * filter height * filter width" /pooling size  
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        #�������fan_in��fan_out �������Ǵ��빫ʽ���Դ��������ʼ��W,W�������Ծ����
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX ),
            borrow=True
        )
        #ƫ��b��һά������ÿ�����ͼ������ͼ����Ӧһ��ƫ�ã�  
        #�����������ͼ�ĸ�����filter���������������filter_shape[0]��number of filters����ʼ�� 
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        #������ͼ����filter�����conv.conv2d����  
        #�����û�м�b��ͨ��sigmoid��������һ���򻯡�
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        # maxpooling������Ӳ�������
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        #��ƫ�ã���ͨ��tanhӳ�䣬�õ����+�Ӳ�������������  
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #���+������Ĳ���  
        self.params = [self.W, self.b]
        
    # �洢ִ�в���
    def save_net(self, path):  
        import cPickle  
        write_file = open(path, 'wb')   
        cPickle.dump(self.params, write_file, -1)
        write_file.close()    

# ʵ��LeNet5 ��LeNet5����������㣬��һ���������20������ˣ��ڶ����������50�������
# learning_rate:ѧϰ���ʣ�����ݶ�ǰ��ϵ���� 
# n_epochsѵ��������ÿһ�������������batch������������ 
# batch_size,��������Ϊ500����ÿ������500���������ż����ݶȲ����²��� 
# nkerns=[20, 50],ÿһ��LeNetConvPoolLayer����˵ĸ�������һ��LeNetConvPoolLayer�� 
# 20������ˣ��ڶ�����50��  
def evaluate_lenet5(learning_rate=0.1, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[20, 50], batch_size=500):
    rng = numpy.random.RandomState(23455)
    datasets = load_data(dataset)  #��������
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    # ���弸��������index��ʾbatch�±꣬x��ʾ�����ѵ�����ݣ�y��Ӧ���ǩ  
    index = T.lscalar()  
    x = T.matrix('x')   
    y = T.ivector('y')  
    ############
    # ����ģ�� #
    ############
    print '... building the model'
    # ���Ǽ��ؽ�����batch��С��������(batch_size, 28 * 28)������LeNetConvPoolLayer����������ά�ģ�����Ҫreshape
    layer0_input = x.reshape((batch_size, 1, 28, 28))
    # layer0����һ��LeNetConvPoolLayer��  
    # ����ĵ���ͼƬ(28,28)������conv�õ�(28-5+1 , 28-5+1) = (24, 24)��  
    # ����maxpooling�õ�(24/2, 24/2) = (12, 12)  
    # ��Ϊÿ��batch��batch_size��ͼ����һ��LeNetConvPoolLayer����nkerns[0]������ˣ�  
    # ��layer0���Ϊ(batch_size, nkerns[0], 12, 12) 
    layer0 = LeNetConvPoolLayer(
        rng, input=layer0_input, 
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5), 
        poolsize=(2, 2)
    )
    # layer1���ڶ���LeNetConvPoolLayer��  
    # ������layer0�������ÿ������ͼΪ(12,12),����conv�õ�(12-5+1, 12-5+1) = (8, 8),  
    # ����maxpooling�õ�(8/2, 8/2) = (4, 4)  
    # ��Ϊÿ��batch��batch_size��ͼ������ͼ�����ڶ���LeNetConvPoolLayer����nkerns[1]�������  
    # ����layer1���Ϊ(batch_size, nkerns[1], 4, 4)  
    layer1 = LeNetConvPoolLayer(
        rng,  input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )
    #ǰ�涨���������LeNetConvPoolLayer��layer0��layer1����layer1�����layer2������һ��ȫ���Ӳ㣬�൱��MLP�����������  
    #�ʿ�����MLP�ж����HiddenLayer����ʼ��layer2��layer2�������Ƕ�ά��(batch_size, num_pixels) ��  
    #��Ҫ���ϲ���ͬһ��ͼ����ͬ����˾������������ͼ�ϲ�Ϊһά������  
    #Ҳ���ǽ�layer1�����(batch_size, nkerns[1], 4, 4)flattenΪ(batch_size, nkerns[1]*4*4)=(500��800),��Ϊlayer2�����롣  
    #(500��800)��ʾ��500��������ÿһ�д���һ��������layer2�������С��(batch_size,n_out)=(500,500) 
    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )
    # ���һ��layer3�Ƿ���㣬�õ����߼��ع��ж����LogisticRegression��  
    # layer3��������layer2�����(500,500)��layer3���������(batch_size,n_out)=(500,10) 
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    # ���ۺ���NLL  
    cost = layer3.negative_log_likelihood(y)
    # test_model���������x��y���ݸ�����index���廯��Ȼ�����layer3��  
    # layer3�ֻ����ص���layer2��layer1��layer0����test_model��ʵ��������CNN�ṹ��  
    # test_model��������x��y�������layer3.errors(y)�����������
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )    
    # ������train_model���漰���Ż��㷨��SGD����Ҫ�����ݶȡ����²���     
    params = layer3.params + layer2.params + layer1.params + layer0.params  # ������  
    grads = T.grad(cost, params) # �Ը����������ݶ�
    # ��Ϊ����̫�࣬��updates��������һ��һ�������д�����Ǻ��鷳�ģ�
    # ������������һ��for..in..,�Զ����ɹ����(param_i, param_i - learning_rate * grad_i)
    updates = [ (param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads) ]
    #train_model���������ͬtest_model��train_model���test_model��validation_model���updates����
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    ############
    # ѵ��ģ�� #
    ############
    print '... training'    
    patience = 10000 # ������ֹ����
    patience_increase = 2 
    improvement_threshold = 0.995  
    # ��������validation_frequency���Ա�֤ÿһ��epoch��������֤���ϲ��ԡ�
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if iter % 100 == 0:
            	print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_losses = [ test_model(i) for i in xrange(n_test_batches) ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, stock_rnn error of ' 'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
    layer0.save_net("layer0")
    layer1.save_net("layer1")
    layer2.save_net("layer2")
    layer3.save_net("layer3")
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, ' 'with stock_rnn performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
