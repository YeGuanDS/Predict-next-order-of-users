from six.moves import cPickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import gc

IDIR = 'data/'



class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose
    def __enter__(self):
        if self.verbose:
            print(self.process_name + " begin ......")
            self.begin_time = time.time()
    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            print(self.process_name + " end ......")
            print('time lapsing {0} s \n'.format(end_time - self.begin_time))
            
def ka_add_groupby_features_n_vs_1(df, group_columns_list, target_columns_list, methods_list, keep_only_stats=True, verbose=1):
    '''Create statistical columns, group by [N columns] and compute stats on [1 column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       target_columns_list: list_like
          column you want to compute stats, need to be a list with only one element
       methods_list: list_like
          methods that you want to use, all methods that supported by groupby in Pandas

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       ka_add_stats_features_n_vs_1(train, group_columns_list=['x0'], target_columns_list=['x10'])
    '''
    with tick_tock("add stats features", verbose):
        dicts = {"group_columns_list": group_columns_list , "target_columns_list": target_columns_list, "methods_list" :methods_list}

        for k, v in dicts.items():
            try:
                if type(v) == list:
                    pass
                else:
                    raise TypeError(k + "should be a list")
            except TypeError as e:
                print(e)
                raise

        grouped_name = ''.join(group_columns_list)
        target_name = ''.join(target_columns_list)
        combine_name = [[grouped_name] + [method_name] + [target_name] for method_name in methods_list]

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped[target_name].agg(methods_list).reset_index()
        the_stats.columns = [grouped_name] + \
                            ['_%s_%s_by_%s' % (grouped_name, method_name, target_name) \
                             for (grouped_name, method_name, target_name) in combine_name]
        if keep_only_stats:
            return the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')
        return df_new
    
#f = open('processed_data.save', 'rb')
#X_train = cPickle.load(f)
#y_train = cPickle.load(f)
#X_test = cPickle.load(f)
#f.close()
X_train=pd.read_hdf("X_train_w32.h5",key="X_train")
y_train=pd.read_hdf("y_train_w32.h5",key="y_train")
X_train.drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis=1, inplace=True)
train = lgb.Dataset(data=X_train, label=y_train, max_bin=127)
del X_train,y_train
gc.collect()

params = {
    'task': 'train',
    'boosting': 'gbdt',
    'application':'binary',
    'learning_rate':0.1,
    'objective': 'binary',
    'metric': {'binary_logloss','auc'},
    'num_leaves': 128,#96,128,192,256,512
    'max_depth': 12,
    'num_threads':4,
    'device':'gpu',
    'feature_fraction': 0.75,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'lambda_l1':60,
    'lambda_l2':30,
    'is_unbalance':False,
    'verbosity':-1,
    'bagging_seed':1294
}





model=lgb.train(params,train,num_boost_round=446)

# Plot importance
#plt.figure(figsize=(15,15))
#lgb.plot_importance(model)
#plt.show()

##predict
#THRESHOLD=0.21
#pred=model.predict(X_test2,num_iteration=179)
#sub=X_test['order_id'].to_frame(name='order_id')
#sub['reordered']=(pred>THRESHOLD)*1
#sub['product_id']=X_test.product_id.astype(str)
#
#submit = ka_add_groupby_features_n_vs_1(sub[sub.reordered == 1], 
#                                               group_columns_list=['order_id'],
#                                               target_columns_list= ['product_id'],
#                                               methods_list=[lambda x: ' '.join(set(x))], keep_only_stats=True)
#
#
#
#sample_submission=pd.read_csv(IDIR + 'sample_submission.csv')
#submit.columns = sample_submission.columns.tolist()
#submit_final = sample_submission[['order_id']].merge(submit, how='left').fillna('None')
#
#sub['None']=1-pred
#submit_final['None']=sub['None'].groupby(sub.order_id).agg(lambda x: np.prod(x))
#none_inx=submit_final['None']>THRESHOLD
#submit_final['products'][none_inx]='None'
#submit_final.drop('None',axis=1,inplace=True)
#
#
#submit_final.to_csv("python_test_None.csv", index=False)



#predict
X_test=pd.read_hdf("X_test_w32.h5",key="X_test")
sub=X_test['order_id'].to_frame(name='order_id')
sub['product_id']=X_test.product_id.astype(str)
X_test.drop(['reordered','eval_set', 'user_id', 'product_id', 'order_id'], axis=1,inplace=True)

pred=model.predict(X_test,num_iteration=446)
sub['reordered']=pred
sub.to_csv('sub_w32_reg_28.csv',index=False)

del X_test,sub
gc.collect()