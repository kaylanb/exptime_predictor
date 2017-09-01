"""
Wraps pre-processing of data, adding features, and everything else that 
  prepares for Machine Learning algorithms with Scikit-Learn duck typed 
  transformers

Scikit-Learn transformers require classes to have three methods:
  fit()
  transform()
  fit_transform()
"""

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

import exptime_predictor.data as Data


class PreProcessor(BaseEstimator, TransformerMixin):
  """Applies all preprocessing to data after read from disk
  
  Args:
    df: pandas DataFrame

  Returns:
    df: pandas DataFrame
  """
  
  def fit(self, df):
      return self
  def transform(self, df):
    df= Clean().keep_science_exposures(df)
    df= Clean().add_night_obs(df)
    df= Clean().drop_bad_transp(df, thresh=0.9)
    # TODO: remove duplicated expids b/c have > 1000 exposures on some nights
    # ALWAYS last step
    df= Clean().drop_nights_wfew_exposures(df, nexp=20)

    df= AddYlabel().use_obsdb_expfactor(df)
    df= AddYlabel().clean(df)
    return df

df_train,df_test= Split_TrainTest().random_sampling(df)

class ColumnDropper(object):
  def unneeded(self,df):
    """necessary cleaning"""
    zero_arrs= ['bad_pixcnt','readtime']
    noinfo= ['id','filename','extension',
             'camera','md5sum','obstype']
    cols= zero_arrs + noinfo
    return df.drop(cols,axis=1)

  def optional(self,df):
    """optional cleaning"""
    # correclation coeff with tneed < 0.01
    cols= ['mjd_obs','transparency','expnum']
    return df.drop(cols,axis=1)




class DataFrameSelector(BaseEstimator, TransformerMixin):
  """Select features for ML and convert DataFrame to np.array
  
  Args:
    features: list of feature names
    df: pandas DataFrame

  Returns:
    numpy array
  """
  
  def __init__(self, features):
      self.features = features
  def fit(self, df, y=None):
      return self
  def transform(self, df):
      return df[self.features].values


num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

if __name__ == "__main__":
  d= GetData(REPO_DIR)
  d.fetch()
  df = d.load()
  data = full_pipeline.fit_transform(df)
