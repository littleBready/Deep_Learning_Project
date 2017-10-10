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

