from six.moves import cPickle
import timeit
import numpy
import scipy.io

import theano
import theano.tensor as T

from src.util import *
from src.models import *
from src.wrapper import *
from src.wrapper_unsup import *
from src.baseline import *

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

    
pids = []
