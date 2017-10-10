print "creating a model ..."
start_time = timeit.default_timer()
modelClass = DeepGraphModel  
modelOptions={'n_classes':2, 
              'dim_feat':data['data_dim'], 
             }
    
model = BaselineWrapper (modelOptions, data, graph,
                          learning_rate_classify=options['model_lrate_classify'],
                         )
print "finished %d minutes." % int((timeit.default_timer()-start_time) / 60)

print "creating the query ..."
querymaker = QueryMaker (data, seed=options['query_seed'])
querymaker.setClassOfInterest (options['query_class_of_interest'])
querymaker.init (num_pos=options['query_num_pos'], num_neg=options['query_num_neg'])

print "training the model ..."
start_time = timeit.default_timer()
    
trainFunc = model.train

cost_classify, test_result = ( trainFunc (
           max_iter=options['train_max_iter'], 
           batch_size_classify=options['train_batch_size_classify'], 
           verbose=True, verboseRate=100,
          ))

print "finished %d minutes." % int((timeit.default_timer()-start_time) / 60)

print "testing the model ..."
start_time = timeit.default_timer()
test_result = model.test ()
print "finished %d minutes." % int((timeit.default_timer()-start_time) / 60)

scipy.io.savemat('./results/exp%d.mat'%options['exp_number'],
                 {'cost_classify':cost_classify,
                  'test_result':test_result,
                  'options':options,
                 })

