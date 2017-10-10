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
