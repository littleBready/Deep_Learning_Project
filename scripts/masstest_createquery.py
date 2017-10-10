print "creating the query ..."
querymaker = QueryMaker (data, seed=options['query_seed'])
querymaker.setClassOfInterest (options['query_class_of_interest'])
querymaker.init (num_pos=options['query_num_pos'], num_neg=options['query_num_neg'])
