if options['model_type'] == 'shallowModel1':
    embdict1 = T.tanh (T.dot(data ['X'], model.model.emb1.W) + model.model.emb1.b)
    embdict2 = T.tanh (T.dot(data ['X'], model.model.emb2.W) + model.model.emb2.b)
    scipy.io.savemat('./model/emb_param_%d.mat'%options['exp_number'],
                     {'E1':embdict1.eval(),
                      'E2':embdict2.eval(),
                      #'W1':model.model.emb1.W.get_value(),
                      #'b1':model.model.emb1.b.get_value(),
                      #'W2':model.model.emb2.W.get_value(),
                      #'b2':model.model.emb2.b.get_value(),
                      'options':options,
                      'random_permutation':data['randperm'],
                      'random_query':list(data['assumed_labeled_set'])
                     })
elif options['model_type'] == 'shallowModel2':
    scipy.io.savemat('./model/emb_nonparam_%d.mat'%options['exp_number'],
                     {'E1':model.model.embdict1.get_value(),
                      'E2':model.model.embdict2.get_value(),
                      'options':options,
                      'random_permutation':data['randperm'],
                      'label':data['yval'],
                      'random_query':list(data['assumed_labeled_set'])
                     })
