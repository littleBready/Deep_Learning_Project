options = {
    'exp_number' : 1001,
    
    'data_seed' : 1234,
    'graph_rebuild': False,
    'graph_k' : 7,
    'graph_sigma' : 3.,
    'model_seed' : 1235,
    'model_type' : 'myModel1',
    'model_dim_hid':1000, 
    'model_n_hid':1, 
    'model_L1_reg':0.000, 
    'model_L2_reg':0.000,   
    'model_lrate_classify' : 0.001,
    'model_lrate_predictContext' : 0.001,
    'query_seed' : 1236,
    'query_class_of_interest' : 7,
    'query_num_pos' : 3,
    'query_num_neg' : 6,
    'train_seed' : 1237,
    'train_max_outer_iter' : 20,
    'train_max_inner_iter_cl' : 0,
    'train_max_inner_iter_pC' : 100,                    
    'train_batch_size_classify' : 30, 
    'train_batch_size_predictContext' : 200,
    'train_r1' : 0.5, 
    'train_r2' : 0.99,
    'train_d' : 3,
    'train_fast' : True,
    'train_sequence_mode' : False,
    'train_sequence_outer_iter' : [200, 200, 200],
    'train_sequence_lrate_cl' : [0.0, 0.0, 0.0],
    'train_sequence_lrate_pC' : [10.0, 2.0, 0.3]
}


from six.moves import cPickle
import timeit
import numpy
import scipy.io

import theano
import theano.tensor as T

from src.util import *
from src.models import *
from src.wrapper import *

pids = []

print "loading data ..."
start_time = timeit.default_timer()
data = load_data (seed=options['data_seed'], 
                  ratio_test=None, unlabeled=False)
print "finished %d minutes." % int((timeit.default_timer()-start_time) / 60)

if options['graph_rebuild']:
    print "constructing graph ..."
    start_time = timeit.default_timer()
    graph = TheanoGraph (data)
    graph.constructKNNexp (k=options['graph_k'], 
                           sigma=options['graph_sigma'])
    print "finished %d minutes." % int((timeit.default_timer()-start_time) / 60)
else:
    print "loading graph ..."
    start_time = timeit.default_timer()
    graph = TheanoGraph (data)
    graph.load_list (options['graph_k'], options['graph_sigma']) # can change it to load_mtx
    print "finished %d minutes." % int((timeit.default_timer()-start_time) / 60)
    
print "creating a model ..."
start_time = timeit.default_timer()
if options['model_type'] == 'myModel1':
    modelClass = DeepGraphModel  
    modelOptions={'seed':options['model_seed'],
                  'n_classes':2, 
                  'dim_feat':data['data_dim'], 
                  'dim_hid':options['model_dim_hid'],
                  'n_hid':options['model_n_hid'], 
                  'L1_reg':options['model_L1_reg'],
                  'L2_reg':options['model_L2_reg']
                 }
elif options['model_type'] == 'myModel2':
    modelClass = DeepGraphModel2  
    modelOptions={'seed':options['model_seed'],
                  'n_classes':2, 
                  'dim_feat':data['data_dim'], 
                  'dim_hid':options['model_dim_hid'],
                  'n_hid':options['model_n_hid'], 
                  'L1_reg':options['model_L1_reg'],
                  'L2_reg':options['model_L2_reg']
                 }
elif options['model_type'] == 'myModel3':
    modelClass = DeepGraphModel3  
    modelOptions={'seed':options['model_seed'],
                  'n_classes':2, 
                  'dim_feat':data['data_dim'], 
                  'dim_hid':options['model_dim_hid'],
                  'n_hid':options['model_n_hid'], 
                  'L1_reg':options['model_L1_reg'],
                  'L2_reg':options['model_L2_reg']
                 }
else:
    raise "Incorrect Model Type"
    
model = DeepGraphWrapper (modelClass, modelOptions, data, graph,
                          learning_rate_classify=options['model_lrate_classify'],
                          learning_rate_predictContext=options['model_lrate_predictContext']
                         )
print "finished %d minutes." % int((timeit.default_timer()-start_time) / 60)

print "creating the query ..."
querymaker = QueryMaker (data, seed=options['query_seed'])
querymaker.setClassOfInterest (options['query_class_of_interest'])
querymaker.init (num_pos=options['query_num_pos'], num_neg=options['query_num_neg'])

print "training the model ..."
start_time = timeit.default_timer()
    
trainFunc = model.train_fast if options['train_fast'] else model.train    

if options ['train_sequence_mode']:
    cost_classify, cost_predictContext, time_classify, time_predictContext, test_result, test_time = (
    trainFunc (seed=options['train_seed'], 
               max_outer_iter=None, 
               max_inner_iter_classify=options['train_max_inner_iter_cl'], 
               max_inner_iter_predictContext=options['train_max_inner_iter_pC'],                    
               batch_size_classify=options['train_batch_size_classify'], 
               batch_size_predictContext=options['train_batch_size_predictContext'], 
               r1=options['train_r1'], 
               r2=options['train_r2'], 
               d=options['train_d'], 
               verbose=True, verboseRate=100,
               max_outer_iter_sequence = options['train_sequence_outer_iter'],
               learning_rate_classify = options['train_sequence_lrate_cl'], 
               learning_rate_predictContext = options['train_sequence_lrate_pC'],
               pids = pids
              ))
else:
    cost_classify, cost_predictContext, time_classify, time_predictContext, test_result, test_time = (
    trainFunc (seed=options['train_seed'], 
               max_outer_iter=options['train_max_outer_iter'], 
               max_inner_iter_classify=options['train_max_inner_iter_cl'], 
               max_inner_iter_predictContext=options['train_max_inner_iter_pC'],                    
               batch_size_classify=options['train_batch_size_classify'], 
               batch_size_predictContext=options['train_batch_size_predictContext'], 
               r1=options['train_r1'], 
               r2=options['train_r2'], 
               d=options['train_d'], 
               verbose=True, verboseRate=100,
               pids = pids
              ))

print "finished %d minutes." % int((timeit.default_timer()-start_time) / 60)

print "testing the model ..."
start_time = timeit.default_timer()
test_result = model.test ()
print "finished %d minutes." % int((timeit.default_timer()-start_time) / 60)

scipy.io.savemat('./results/convergence%d.mat'%options['exp_number'],
                 {'cost_classify':cost_classify,
                  'cost_predictContext':cost_predictContext,
                  'time_classify':time_classify,
                  'time_predictContext':time_predictContext,
                  'test_result':test_result,
                  'test_time':test_time,
                  'options':options,
                  'test_result':test_result
                 })

