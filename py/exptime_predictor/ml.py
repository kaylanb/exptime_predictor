import numpy as np
import os
import pandas as pd
import sqlite3
import time
from datetime import date, timedelta
from astropy.time import Time

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelBinarizer

from obiwan.common import save_png,dobash
from exptime_predictor.data import main as data_main
# to make this notebook's output stable across runs
np.random.seed(7)


class df2array(BaseEstimator, TransformerMixin):
	"""Pipeline: DataFrame to Numpy array"""
	def __init__(self, cols):
	    self.cols = cols
	def fit(self, X, y=None):
	    return self
	def transform(self, X):
	    return X[self.cols].values

class df2binary(BaseEstimator, TransformerMixin):
	"""DataFrame of category columns and returns binarized array"""
	def __init__(self, cols):
	    self.cols = cols
	def fit(self, X, y=None):
	    return self
	def transform(self, X):
	    encoder = LabelBinarizer()
	    list_of_mat= [encoder.fit_transform(X[col].values)
	                  for col in self.cols]
	    return np.concatenate(list_of_mat,axis=1)

class MyPipelines(object):
	def num(self,num_cols):
		"""num_col: list of numerical columns"""
		return Pipeline([
	        ('df2array', df2array(num_cols)),
	        #('imputer', Imputer(strategy="median")),
	        ('stdize', StandardScaler()),
	    ])

	def cat(self,cat_cols):
		"""cat_col: list of categorical columns"""
		return Pipeline([
	        ('df2binary', df2binary(cat_cols)),
	    ])

if __name__ == '__main__':
	df_train,df_test = data_main()

	categ_cols = ['band','passnumber']
	not_num_cols= ['bad_pixcnt','readtime','expnum',
	               'id','tileid','expfactor',
	               'object','obstype','extension',
	               'filename','camera','md5sum','tneed'] + categ_cols

	num_cols = list(df_train.columns)
	for col in not_num_cols:
	    num_cols.remove(col)
	print('numerical cols= ',num_cols)
	print('categorical cols= ',categ_cols)

	num_pipeline = MyPipelines().num(num_cols) 
	cat_pipeline = MyPipelines().cat(categ_cols) 

	pipe = FeatureUnion(transformer_list=[
	    ("num_pipeline", num_pipeline),
	    ("cat_pipeline", cat_pipeline),
	])

	x_train= pipe.fit_transform(df_train)
	y_train= df_train['tneed'].copy().values
	print(x_train.shape,y_train.shape)



