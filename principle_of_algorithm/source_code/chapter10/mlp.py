# -*- coding: utf-8 -*-
"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
        ע�ͣ� 
        ���Ƕ������ز���࣬������ȷ�����ز�����뼴input����������ز����Ԫ����������������ز���ȫ���ӵġ� 
        ����������n_inά��������Ҳ����˵ʱn_in����Ԫ�������ز���n_out����Ԫ������Ϊ��ȫ���ӣ� 
        һ����n_in*n_out��Ȩ�أ���W��Сʱ(n_in,n_out),n_in��n_out�У�ÿһ�ж�Ӧ���ز��ÿһ����Ԫ������Ȩ�ء� 
        b��ƫ�ã����ز���n_out����Ԫ����bʱn_outά������ 
        rng���������������numpy.random.RandomState�����ڳ�ʼ��W�� 
        inputѵ��ģ�����õ����������룬������MLP������㣬MLP����������Ԫ����ʱn_in��������Ĳ���input��С�ǣ�n_example,n_in��,ÿһ��һ����������ÿһ����ΪMLP������㡣 
        activation:�����,���ﶨ��Ϊ����tan
        """
        self.input = input  # ��HiddenLayer��input�������ݽ�����input 

        """ 
        ע�ͣ� 
        ����Ҫ����GPU����W��b����ʹ�� dtype=theano.config.floatX,���Ҷ���Ϊtheano.shared 
        ���⣬W�ĳ�ʼ���и��������ʹ��tanh����������-sqrt(6./(n_in+n_hidden))��sqrt(6./(n_in+n_hidden))֮����� 
        ��ȡ��ֵ����ʼ��W����ʱsigmoid�������������ٳ�4���� 
        """  
        #���Wδ��ʼ�������������������ʼ����  
        #��������жϵ�ԭ���ǣ���ʱ�����ǿ�����ѵ���õĲ�������ʼ��W�����ҵ���һƪ���¡�
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        # �����涨���W��b����ʼ����HiddenLayer��W��b 
        self.W = W
        self.b = b
        # ���������� 
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # ������Ĳ��� 
        self.params = [self.W, self.b]


# 3���MLP  
class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        #��������hiddenLayer�������Ϊ�����logRegressionLayer�����룬�����Ͱ�����������  
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # �����Ѿ������MLP�Ļ����ṹ��������MLPģ�͵������������ߺ��� 
        
        # �����������L1��L2_sqr  
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # ��ʧ����Nll��Ҳ�д��ۺ����� 
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        #���
        self.errors = self.logRegressionLayer.errors

        #MLP�Ĳ���
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
# test_mlp��һ��Ӧ��ʵ�������ݶ��½����Ż�MLP�����MNIST���ݼ�
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """ 
    ע�ͣ� 
    learning_rateѧϰ���ʣ��ݶ�ǰ��ϵ���� 
    L1_reg��L2_reg��������ǰ��ϵ����Ȩ����������Nll��ı��� 
    ���ۺ���=Nll+L1_reg*L1����L2_reg*L2_sqr 
    n_epochs������������������ѵ�������������ڽ����Ż����� 
    dataset��ѵ�����ݵ�·�� 
    n_hidden:���ز���Ԫ���� 
    batch_size=20����ÿѵ����20�������ż����ݶȲ����²��� 
    """  
    # �������ݼ�������Ϊѵ��������֤�������Լ���  
    datasets = load_data(dataset)      
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # shape[0]���������һ�д���һ���������ʻ�ȡ����������������batch_size���Եõ��ж��ٸ�batch
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    ############
    # ����ģ�� #
    ############
    print '... building the model'
    
    index = T.lscalar()  # index��ʾbatch���±꣬���� 
    x = T.matrix('x')  # x��ʾ���ݼ� 
    y = T.ivector('y')  # y��ʾ���һά����
    
    rng = numpy.random.RandomState(1234)
    
    #ʵ����һ��MLP������Ϊclassifier  
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )
    
    #���ۺ������й�����  
    #��y����ʼ��������ʵ����һ�������Ĳ���x��classifier��
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    #cost�����Ը���������ƫ����ֵ�����ݶȣ�����gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    #�������¹���  
    #updates[(),(),()....],ÿ���������涼��(param, param - learning_rate * gparam)����ÿ�������Լ����ĸ��¹�ʽ
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams) ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
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

    patience = 10000  # ����������
    patience_increase = 2  # ����
    improvement_threshold = 0.995  # �൱��ĸ��Ʊ���Ϊ������
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

            minibatch_avg_cost = train_model(minibatch_index)
            # ��ǰ������
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # stock_rnn it on the stock_rnn set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, stock_rnn error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with stock_rnn performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp()
