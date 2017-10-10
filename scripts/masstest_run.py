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

scipy.io.savemat('./results/exp%d.mat'%options['exp_number'],
                 {'cost_classify':cost_classify,
                  'cost_predictContext':cost_predictContext,
                  'time_classify':time_classify,
                  'time_predictContext':time_predictContext,
                  'test_result':test_result,
                  'test_time':test_time,
                  'options':options,
                  'test_result':test_result
                 })

