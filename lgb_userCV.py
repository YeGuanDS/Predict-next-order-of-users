from six.moves import cPickle
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
import lightgbm as lgb


f = open('processed_data.save', 'rb')
X_train = cPickle.load(f)
y_train = cPickle.load(f)
X_test = cPickle.load(f)
f.close()
X_test.drop('reordered',axis=1,inplace=True)

#shuffle
X_train,y_train=shuffle(X_train,y_train, random_state=9412)
#add group K fold
K=5

groups=X_train.user_id
group_kfold = GroupKFold(n_splits=K)
group_kfold.get_n_splits(X_train, y_train, groups)
myGroups=list(group_kfold.split(X_train, y_train, groups))

X_train.drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis=1, inplace=True)


train = lgb.Dataset(data=X_train, label=y_train, max_bin=127)#,group=group_kfold)

params = {
    'task': 'train',
    'boosting': 'gbdt',
    'application':'binary',
    'learning_rate':0.1,
    'objective': 'binary',
    'metric': {'binary_logloss','auc'},
    'num_leaves': 128,#96,128,192,256,512
    'max_depth': 10,
    'num_threads':4,
    'device':'gpu',
    'feature_fraction': 0.75,#0.95,0.9,0.85,0.8,0.75,0.7,0.65
    'bagging_fraction': 0.95,#0.95,0.9,0.85,0.8
    'bagging_freq': 5,
    'lambda_l1':0.01,
    'lambda_l2':0.01,
    'is_unbalance':False,
    'verbosity':-1
}

#grid search
max_depth=[10]
num_leaves=[128]
feature_fraction=[0.75]
bagging_fraction=[0.8,0.7,0.6,0.5]
lambda_l1=[0.1,0.3,1,3,10,30]
lambda_l2=[0.3,1,3,10,30]
results=list()
for md in max_depth:
    params['max_depth']=md
    for nl in num_leaves:
        params['num_leaves']=nl
        for ff in feature_fraction:
            params['feature_fraction']=ff
            for bf in bagging_fraction:
                params['bagging_fraction']=bf
                for l1 in lambda_l1:
                    params['lambda_l1']=l1
                    for l2 in lambda_l2:
                        params['lambda_l2']=l2
                        param=[md,nl,ff,bf,l1,l2]
                        print param
                        bst=lgb.cv(params,train,num_boost_round=5000,metrics={'binary_logloss'},folds=myGroups,early_stopping_rounds=20,verbose_eval=True,seed=777)#
                        results.append((param,bst))
            
f = open('gridsearch_result.save', 'wb')
cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
