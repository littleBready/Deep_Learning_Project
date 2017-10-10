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
