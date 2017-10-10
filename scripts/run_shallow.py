print "creating a model ..."
start_time = timeit.default_timer()
if options['model_type'] == 'shallowModel1':
    modelClass = ShallowGraphModelParametric  
    modelOptions={'seed':options['model_seed'],
                  'n_classes':2, 
                  'dim_feat':data['data_dim'], 
                  'dim_emb':options['model_dim_emb'],
                  'L1_reg':options['model_L1_reg'],
                  'L2_reg':options['model_L2_reg']
                 }
elif options['model_type'] == 'shallowModel2':
    modelClass = ShallowGraphModelNonParametric  
    modelOptions={'seed':options['model_seed'],
                  'n_classes':2, 
                  'dim_feat':data['data_dim'], 
                  'dict_size':data['data_num'], 
                  'dim_emb':options['model_dim_emb'],
                  'L1_reg':options['model_L1_reg'],
                  'L2_reg':options['model_L2_reg']
                 }
else:
    raise "Incorrect Model Type"
    
model = ShallowGraphWrapper (modelClass, modelOptions, data, graph,
                          learning_rate_predictContext=options['model_lrate_predictContext']
                         )
print "finished %d minutes." % int((timeit.default_timer()-start_time) / 60)

print "training the model ..."
start_time = timeit.default_timer()
    
trainFunc = model.train_fast    

if options ['train_sequence_mode']:
    cost_predictContext, time_predictContext= (
    trainFunc (seed=options['train_seed'], 
               max_outer_iter=None, 
               max_inner_iter_predictContext=options['train_max_inner_iter_pC'],
               batch_size_predictContext=options['train_batch_size_predictContext'], 
               r1=options['train_r1'], 
               r2=options['train_r2'], 
               d=options['train_d'], 
               verbose=True, verboseRate=100,
               max_outer_iter_sequence = options['train_sequence_outer_iter'],
               learning_rate_predictContext = options['train_sequence_lrate_pC'],
               pids = pids
              ))
else:
    cost_predictContext, time_predictContext = (
    trainFunc (seed=options['train_seed'], 
               max_outer_iter=options['train_max_outer_iter'], 
               max_inner_iter_predictContext=options['train_max_inner_iter_pC'],
               batch_size_predictContext=options['train_batch_size_predictContext'], 
               r1=options['train_r1'], 
               r2=options['train_r2'], 
               d=options['train_d'], 
               verbose=True, verboseRate=100,
               pids = pids
              ))

print "finished %d minutes." % int((timeit.default_timer()-start_time) / 60)

scipy.io.savemat('./results/exp%d.mat'%options['exp_number'],
                 {'cost_predictContext':cost_predictContext,
                  'time_predictContext':time_predictContext,
                  'options':options,
                 })

