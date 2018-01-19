import numpy as np
import time
from scipy import stats
import pandas as pd
import gc

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
            
def ka_add_groupby_features_1_vs_n(df, group_columns_list, agg_dict, only_new_feature=True):
    '''Create statistical columns, group by [N columns] and compute stats on [N column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       agg_dict: python dictionary

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       {real_column_name: {your_specified_new_column_name : method}}
       agg_dict = {'user_id':{'prod_tot_cnts':'count'},
                   'reordered':{'reorder_tot_cnts_of_this_prod':'sum'},
                   'user_buy_product_times': {'prod_order_once':lambda x: sum(x==1),
                                              'prod_order_more_than_once':lambda x: sum(x==2)}}
       ka_add_stats_features_1_vs_n(train, ['product_id'], agg_dict)
    '''
    with tick_tock("add stats features"):
        try:
            if type(group_columns_list) == list:
                pass
            else:
                raise TypeError(k + "should be a list")
        except TypeError as e:
            print(e)
            raise

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped.agg(agg_dict)
        the_stats.columns = the_stats.columns.droplevel(0)
        the_stats.reset_index(inplace=True)
        if only_new_feature:
            df_new = the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')

    return df_new


IDIR = 'data/'


"""
load data
"""
print('loading prior')
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading train')
train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders')
orders = pd.read_csv(IDIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

print('loading products')
products = pd.read_csv(IDIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': 'category',
        'department_id': 'category'})
products.drop('product_name',axis=1,inplace=True)

print('loading aisles.csv')
aisles = pd.read_csv(IDIR + 'aisles.csv', dtype={
        'aisle_id': np.uint8,
        'aisle': 'category'})

print('loading departments.csv')
departments = pd.read_csv(IDIR + 'departments.csv', dtype={
        'department_id': np.uint8,
        'department': 'category'})

print('loading w2v')
w2v = pd.read_csv(IDIR + 'w32.csv',dtype={#w2
        'product_id': np.uint16})

products=products.merge(w2v,how='inner',on='product_id')
del w2v
gc.collect()

###
"""
product features
"""
print('computing product f')
prod_f = pd.DataFrame()
#appear in how many orders
prod_f['appearOrderCount'] = priors['order_id'].groupby(priors.product_id).count().astype(np.uint16)
#total reorder times
prod_f['reorderCount'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
#reorder ratio
prod_f['reorderRatio'] = (prod_f['reorderCount']/prod_f['appearOrderCount']).astype(np.float32)
#mean and std of add_to_cart_order
prod_f['add_to_cart_orderMean']=priors['add_to_cart_order'].groupby(priors.product_id).mean().astype(np.float32)
prod_f['add_to_cart_orderStd']=priors['add_to_cart_order'].groupby(priors.product_id).std().astype(np.float32)
#temp merge
priorsXorders=priors.merge(orders,how='inner',on='order_id')
#bought by how many users
prod_f['boughtUserCount']=priorsXorders['user_id'].groupby(priorsXorders.product_id).count().astype(np.uint16)
#bought dow
prod_f['order_dowMean']=priorsXorders['order_dow'].groupby(priorsXorders.product_id).mean().astype(np.float32)
prod_f['order_dowStd']=priorsXorders['order_dow'].groupby(priorsXorders.product_id).std().astype(np.float32)
#bought time
prod_f['order_hour_of_dayMean']=priorsXorders['order_hour_of_day'].groupby(priorsXorders.product_id).mean().astype(np.float32)
prod_f['order_hour_of_dayStd']=priorsXorders['order_hour_of_day'].groupby(priorsXorders.product_id).std().astype(np.float32)
#fillna std=na as 0-->if there is only one sample, std=na
prod_f.fillna(0,inplace=True)

#recency--This is a feature which captures if the product is generally brought more in users earlier orders or later orders
maxPriorOrder=priorsXorders['order_number'].groupby(priorsXorders.user_id).max().astype(np.float32)
maxPriorOrder=maxPriorOrder.rename('userMaxPriorOrderNumber')
maxPriorOrder=pd.DataFrame(maxPriorOrder)
priorsXorders=priorsXorders.merge(maxPriorOrder,how='left',left_on='user_id',right_index=True)
priorsXorders['recency']=(priorsXorders['order_number']/priorsXorders['userMaxPriorOrderNumber']).astype(np.float32)
prod_f['recencyMean']=priorsXorders['recency'].groupby(priorsXorders.product_id).mean().astype(np.float32)
prod_f['recencyStd']=priorsXorders['recency'].groupby(priorsXorders.product_id).std().astype(np.float32)
#user who bought this product had how many orders -- userMaxPriorOrderNumber
prod_f['userMaxPriorOrderNumberMean']=priorsXorders['userMaxPriorOrderNumber'].groupby(priorsXorders.product_id).mean().astype(np.float32)
prod_f['userMaxPriorOrderNumberStd']=priorsXorders['userMaxPriorOrderNumber'].groupby(priorsXorders.product_id).std().astype(np.float32)
#fillna std=na as 0-->if there is only one sample, std=na
prod_f.fillna(0,inplace=True)


priorsXorders['_user_buy_product_times'] = priorsXorders.groupby(['user_id', 'product_id']).cumcount() + 1
tmp_group=priorsXorders['_user_buy_product_times'].groupby(priorsXorders.product_id)
# _prod_order_once: 
prod_f['_prod_order_once']=tmp_group.aggregate(lambda x: sum(x==1))
# _prod_order_more_than_once:
prod_f['_prod_order_more_than_once']=tmp_group.aggregate(lambda x: sum(x==2))

#porduct in last order=0/last 2nd order=1/last 3rd order=2,...
priorsXorders['lastNOrder']=priorsXorders['userMaxPriorOrderNumber']-priorsXorders['order_number']
prod_f['apperInLastNOrderMean']=priorsXorders['lastNOrder'].groupby(priorsXorders.product_id).mean().astype(np.float32)
prod_f['apperInLastNOrderStd']=priorsXorders['lastNOrder'].groupby(priorsXorders.product_id).std().astype(np.float32)
prod_f['apperInLastNOrderMax']=priorsXorders['lastNOrder'].groupby(priorsXorders.product_id).max().astype(np.float32)
prod_f['apperInLastNOrderMin']=priorsXorders['lastNOrder'].groupby(priorsXorders.product_id).min().astype(np.float32)
#fillna std=na as 0-->if there is only one sample, std=na
prod_f.fillna(0,inplace=True)

prd=products.merge(prod_f,how='inner',left_on='product_id',right_index=True)
del prod_f
gc.collect()

"""
user features
"""
print('building user features...')
user_f=pd.DataFrame()
priorsXordersXproducts=priorsXorders.merge(products,how='inner',on='product_id')
# num of Aisle purchased from, num of Department purchased from
tmp_group=priorsXordersXproducts['aisle_id'].groupby(priorsXordersXproducts.user_id)
unique_aisle_ids=tmp_group.unique()
for i in xrange(len(unique_aisle_ids)):
    unique_aisle_ids.iloc[i]=len(unique_aisle_ids.iloc[i])
user_f['uniqueAisleCount']=unique_aisle_ids
user_f['uniqueAisleCount']=user_f['uniqueAisleCount'].astype(np.int16)
user_f['freqAisle']=tmp_group.agg(lambda x: stats.mode(x)[0][0])
user_f['freqAisle']=user_f['freqAisle'].astype(np.int16)

tmp_group=priorsXordersXproducts['department_id'].groupby(priorsXordersXproducts.user_id)
unique_department_ids=tmp_group.unique()
for i in xrange(len(unique_department_ids)):
    unique_department_ids.iloc[i]=len(unique_department_ids.iloc[i])
user_f['uniqueDepartmentCount']=unique_department_ids
user_f['uniqueDepartmentCount']=user_f['uniqueDepartmentCount'].astype(np.int16)
user_f['freqDepartment']=tmp_group.agg(lambda x: stats.mode(x)[0][0])
user_f['freqDepartment']=user_f['freqDepartment'].astype(np.int16)
#user total orders
user_f['_user_total_orders']=priorsXorders['order_number'].groupby(priorsXorders.user_id).max().astype(np.float32)
#_user_mean_days_since_prior_order and std
#bought time interval
user_f['days_since_prior_orderMean']=priorsXordersXproducts['days_since_prior_order'].groupby(priorsXordersXproducts.user_id).mean().astype(np.float32)
user_f['days_since_prior_orderStd']=priorsXordersXproducts['days_since_prior_order'].groupby(priorsXordersXproducts.user_id).std().astype(np.float32)
# _user_reorder_ratio
# _user_total_products
# _user_distinct_products
tmp_group=priorsXorders['product_id'].groupby(priorsXorders.user_id)
user_f['_user_total_products']=tmp_group.count()
user_f['_user_distinct_products']=tmp_group.agg(lambda x: x.nunique())

tmp_group=priorsXorders['reordered'].groupby(priorsXorders.user_id)
user_f['_user_reorder_ratio']=(priorsXorders.groupby('user_id')['reordered'].sum() /priorsXorders[priorsXorders['order_number'] > 1].groupby('user_id')['order_number'].count()).rename('_user_reorder_ratio')#tmp_group.agg(lambda x: sum(priorsXorders.loc[x.index,'reordered']==1)/sum(priorsXorders.loc[x.index,'order_number'] > 1))
#backet size
order_size=priorsXordersXproducts['product_id'].groupby(priorsXordersXproducts.order_id).count().astype(np.uint8)
order_size=order_size.rename('order_size')
order_size=pd.DataFrame(order_size)
priorsXordersXproducts=priorsXordersXproducts.merge(order_size,how='left',left_on='order_id',right_index=True)

tmp_group=priorsXordersXproducts['order_size'].groupby(priorsXordersXproducts.user_id)
user_f['order_size_mean']=tmp_group.mean()
user_f['order_size_std']=tmp_group.std()
user_f['order_size_max']=tmp_group.max()
user_f['order_size_min']=tmp_group.min()

us = orders[orders.eval_set != "prior"][['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
us.rename(index=str, columns={'days_since_prior_order': 'time_since_last_order'}, inplace=True)

user_f = user_f.merge(us, how='inner',left_index=True,right_on='user_id')

"""
user*product
"""
# _up_order_count
# _up_first_order_number
# _up_last_order_number
# _up_average_cart_position
agg_dict_4 = {'order_number':{'_up_order_count': 'count', 
                              '_up_first_order_number': 'min', 
                              '_up_last_order_number':'max',
                              '_up_order_past_appears_mean':'mean',###user purchased this product in which past orders-mean
                              '_up_order_past_appears_std':'std'}, 
              'add_to_cart_order':{'_up_average_cart_position': 'mean',
                                   '_up_std_cart_position': 'std'}}

data = ka_add_groupby_features_1_vs_n(df=priorsXorders, group_columns_list=['user_id', 'product_id'], 
                                                      agg_dict=agg_dict_4)

data['_up_order_past_appears_std'].fillna(0,inplace=True)
data['_up_std_cart_position'].fillna(0,inplace=True)#for _up_std_cart_position
data = data.merge(prd, how='inner', on='product_id').merge(user_f, how='inner', on='user_id')

data['_up_order_rate'] = data._up_order_count / data._user_total_orders
data['_up_order_since_last_order'] = data._user_total_orders - data._up_last_order_number
data['_up_order_since_last_order_normalize']=data['_up_order_since_last_order']/data['_user_total_orders']
data['_up_order_past_appears_mean_normalize']=data['_up_order_past_appears_mean']/data._up_last_order_number
data['_up_order_past_appears_std_normalize']=data['_up_order_past_appears_std']/data._up_last_order_number
data['_up_order_rate_since_first_order'] = data._up_order_count / (data._user_total_orders - data._up_first_order_number + 1)

## user purchased this product in which past orders--interval
tmp_group=priorsXorders['order_number'].groupby([priorsXorders.user_id,priorsXorders.product_id])
_up_order_past_appears_interval_mean=tmp_group.agg(lambda x: np.mean(np.diff(np.sort(x))) if len(x)>1 else -1)
_up_order_past_appears_interval_std=tmp_group.agg(lambda x: np.std(np.diff(np.sort(x))) if len(x)>1 else 0)
tmp_df=pd.DataFrame(data={'_up_order_past_appears_interval_mean':_up_order_past_appears_interval_mean,
                          '_up_order_past_appears_interval_std':_up_order_past_appears_interval_std})

data=data.merge(right=tmp_df,left_on=[data.user_id,data.product_id],how='left',right_index=True)
data['_up_order_past_appears_interval_mean'].loc[data['_up_order_past_appears_interval_mean']==-1]=data['_user_total_orders'].loc[data['_up_order_past_appears_interval_mean']==-1]
data['_up_order_past_appears_interval_mean_normalize']=data['_up_order_past_appears_interval_mean']/data['_user_total_orders']
data['_up_order_past_appears_interval_std_normalize']=data['_up_order_past_appears_interval_std']/data['_user_total_orders']
data['_up_order_expect_days_to_order']=data['_up_order_past_appears_interval_mean']-data['_up_order_since_last_order']
data['_up_order_expect_days_to_order_normalize']=data['_up_order_expect_days_to_order']/data._user_total_orders

#last order=0/last 2nd order=1/last 3rd order=2,...---cannot use. prior orders do not appear in the training data
#lastNOrder=priorsXorders['lastNOrder'].groupby(priorsXorders.order_id).min().astype(np.uint16)#they all duplicates min=max=mean
#lastNOrder=pd.DataFrame(lastNOrder)
#data1=data.merge(lastNOrder,how='left',left_on='order_id',right_index=True)

# add user_id to train set
train = train.merge(right=orders[['order_id', 'user_id']], how='left', on='order_id')
data = data.merge(train[['user_id', 'product_id', 'reordered']], on=['user_id', 'product_id'], how='left')

train = data.loc[data.eval_set == "train",:]
#train.drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis=1, inplace=True)
train.loc[:, 'reordered'] = train.reordered.fillna(0)
X_test = data.loc[data.eval_set == "test",:]
X_train=train.drop('reordered',axis=1)#X_train=train[train.columns.difference(['reordered'])]#.difference will shuffle columns
y_train=train.reordered

#del agg_dict_4,aisles,data,departments,maxPriorOrder,order_size,orders,prd,priors,priorsXorders,priorsXordersXproducts,products,tmp_df,train,unique_aisle_ids,unique_department_ids,us,user_f
#gc.collect()

#from six.moves import cPickle
#f = open('processed_data_w32.save', 'wb')
#cPickle.dump(X_train, f, protocol=cPickle.HIGHEST_PROTOCOL)
#cPickle.dump(y_train, f, protocol=cPickle.HIGHEST_PROTOCOL)
#cPickle.dump(X_test, f, protocol=cPickle.HIGHEST_PROTOCOL)
#f.close()

#X_train.to_pickle("X_train_w32.pkl")
#y_train.to_pickle("y_train_w32.pkl")
#X_test.to_pickle("X_test_w32.pkl")

X_train.to_hdf("X_train_w32.h5","X_train",mode="w",format="table")
y_train.to_hdf("y_train_w32.h5","y_train",mode="w",format="table")
X_test.to_hdf("X_test_w32.h5","X_test",mode="w",format="table")
