import numpy as np
import os
import pandas as pd
import sqlite3
import time
from datetime import date, timedelta
from astropy.time import Time

from obiwan.common import save_png,dobash

# to make this notebook's output stable across runs
np.random.seed(7)

# Where to save the figures
REPO_DIR= os.path.join(os.environ['HOME'],
                       'PhdStudent/Research/desi/ml_data/')
DB_DIR= os.path.join(REPO_DIR,'obsbot/obsdb')

class SqliteConnection(object):
  def pandas_fetchall(self,sqlite_fn, exe_cmd):
    print('Reading sqlite db: %s' % sqlite_fn)
    conn = sqlite3.connect(sqlite_fn)
    c= conn.cursor() 
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")   
    print('Has tables: ',c.fetchall())        
    print('executing query: %s' % exe_cmd)
    df = pd.read_sql_query(exe_cmd, conn)
    # cleanup
    c.close()
    conn.close()
    return df

  def fetchall(self,sqlite_fn, exe_cmd):
    print('Reading sqlite db: %s' % sqlite_fn)
    conn = sqlite3.connect(sqlite_fn)
    c= conn.cursor() 
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")   
    print('Has tables: ',c.fetchall())        
    print('executing query: %s' % exe_cmd)
    c.execute(exe_cmd)
    cols= [col[0] for col in c.description] 
    print('Table has cols',cols) 
    list_of_tuples= c.fetchall()
    # cleanup
    c.close()
    conn.close()
    return list_of_tuples

class GetData(object):
  """Fetches and loads decam.sqlite3 data as Pandas df
  
  Args:
    repo_dir: path to obsbot repo, eg. $repo_dir/obsbot
    commit_id: commit id for data version in obsbot repo, 
      default: Aug 25, 2017
  """

  def __init__(self, repo_dir, commit_id='84d63bb9aa33b'):
    self.repo_dir= repo_dir
    self.commit_id= commit_id
    self.Sql= SqliteConnection()

  def fetch(self):
    curr_path = os.getcwd()
    os.chdir( os.path.join(self.repo_dir, 'obsbot'))
    dobash('git pull origin master')
    dobash('git checkout %s' % self.commit_id)
    os.chdir( curr_path)

  def load(self):
    """return Pandas DF of sqlite table"""
    fn = os.path.join(self.repo_dir,"obsbot/obsdb",
                      "decam.sqlite3")
    return self.Sql.pandas_fetchall(fn,
                                    "select * from obsdb_measuredccd")

def change_day(ymd,new_day):
  """given year month day string (ymd) returns the previous or next calendar day
  
  Args:
      ymd: '20110531'
      new_day: ['yesterday','tomorrow']
      
  Returns:
      ymd for next day: '20110601' for example above
  """
  assert(new_day in ['yesterday','tomorrow'])
  t=time.strptime(ymd,'%Y%m%d')
  dt= timedelta(days=1)
  if new_day == 'yesterday':
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday) - dt
  elif new_day == 'tomorrow':
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday) + dt
  return newdate.strftime('%Y%m%d')


class Clean(object):
  def keep_science_exposures(self,df):
    """Only exposures named 'object' and band in grz are science exposures"""
    bands= df.loc[:,'band']
    isGRZ= (bands == 'g') | (bands == 'r') | (bands == 'z') 
    isObject= df.loc[:,'obstype'] == 'object'
    isScience= df['object'].str.lower().str.contains('decals')
    longExp= df['exptime'] > 30
    print('isGRZ %d/%d' % (len(df[isGRZ]),len(df)))
    print('isObject %d/%d' % (len(df[isObject]),len(df)))
    print('isScience %d/%d' % (len(df[isScience]),len(df)))
    print('longExp %d/%d' % (len(df[longExp]),len(df)))
    allCuts= ((isGRZ) & 
              (isObject) &
              (isScience) &
              (longExp))
    print('allCuts %d/%d' % (len(df[allCuts]),len(df)))
    return df[allCuts]
   
  def add_night_obs(self,df):
    """assigns integer ymd to each instance

    Night observed begins in evening and continutes to morning of next
      day. Found that float hrminsec = 150000.0 cleanly separates morning
      from evening
    """
    t = Time(df.loc[:,'mjd_obs'], format='mjd')
    # t.iso is list of elements like '2016-02-25 05:11:31.769'
    day_str= np.array([a.split(' ')[0] for a in t.iso])
    hr_str= np.array([a.split(' ')[1] for a in t.iso])
    day= np.char.replace(day_str,'-','').astype(int) # integer ymd
    hr= np.char.replace(hr_str,':','').astype(float) # float hrminsec
    # 
    hr_thresh= 150000.
    nights= np.sort( list(set(day)))
    # store df's new night_obs colm
    night_arr= np.zeros(len(df)).astype(np.int32) - 1
    #
    for night in nights:
      # night starts at nighttime
      first_half= (day == night) & (hr > hr_thresh)
      # night continues to morning of next day
      str_next_day= get_next_day(str(night))
      sec_half= (day == int(str_next_day) ) & (hr < hr_thresh)
      keep= first_half | sec_half
      if np.where(keep)[0].size > 0:
          night_arr[keep]= night
    return df.assign(night_obs= night_arr)

  def add_night_obs(self,df):
    """assigns integer ymd to each instance

    Night observed begins in evening and continutes to morning of next
      day. Found that float hrminsec = 150000.0 cleanly separates morning
      from evening

    Returns:
      df modified to have 'night_obs' and 'hr_obs' columns
    """
    t = Time(df.loc[:,'mjd_obs'], format='mjd')
    # t.iso is list of elements like '2016-02-25 05:11:31.769'
    nightobs_str= np.array([a.split(' ')[0] for a in t.iso])
    nightobs_str= np.char.replace(nightobs_str,'-','') # ymd
    hr_str= np.array([a.split(' ')[1] for a in t.iso])
    hr= np.char.replace(hr_str,':','').astype(float) # float hrminsec
    # All mornings need to be relabeled as previous calendar day
    hr_thresh= 150000.
    isMorning= hr < hr_thresh
    for ymd in list(set(nightobs_str)):
      # relabel the mornings 
      nightobs_str[ (nightobs_str == ymd) & (isMorning) ]= change_day(ymd,'yesterday')
    df= df.assign(night_obs= nightobs_str.astype(int))
    #df= df.assign(hr_obs= hr)
    return df

  def order_by_mjd(self,df):
    return df.sort_values('mjd_obs')

  def hr_obs(self,mjd_one_night):
    """mjd_one_night: numpy array of one nights mjd_obs values"""
    mjd= mjd_one_night.copy()
    mjd -= mjd[0]
    isEvening= mjd > 1
    isMorning= isEvening == False
    if np.any(isEvening):
        mjd_evening_start= mjd[isEvening].min()
        mjd_morning_start= mjd[isEvening].max()
        mjd[isEvening] -=  mjd_evening_start
        mjd[isMorning] +=  mjd_morning_start - mjd_evening_start
    # hrs not fraction of day
    return mjd * 24.

  def add_hr_obs(self,df):
    hr_obs= np.zeros(len(df))-1
    for night in set(df['night_obs']):
        isNight= df['night_obs'] == night
        hr_obs[isNight]= self.hr_obs(df[isNight]['mjd_obs'].values)
    assert(not 'hr_obs' in df.columns)
    assert(np.all(hr_obs >= 0))
    df['hr_obs']= hr_obs
    return df

  def drop_nights_wfew_exposures(self,df, nexp=20):
    """Return df with all nights with less than nexp dropped"""
    start=len(df)
    df= df.groupby("night_obs").filter(lambda g: g.night_obs.size >= nexp)
    print('Cutting to < %d exposures: %d/%d' %
          (nexp,len(df),start))
    return df
  
  def drop_bad_transp(self,df, thresh=0.9):
    """drop low transp exposures"""
    return df[ df.loc[:,'transparency'] > thresh ]

class LegacyZptData(object):
  """Fetches and loads legacy zeropoints exposure time needed data
  """
  pass

class AddYlabel(object):
  """Adds 'tneed' to training data, the needed exposure time we are trying to predict
  """
  def add_tneed(self,df):
    """uses expfactor in obsdb db which is VERY ROUGH approx"""
    t0= dict(g=70,r=50,z=100)
    tneed_arr= np.zeros(len(df))-1 
    for band in t0.keys():
        isBand= df['band'] == band
        tneed_arr[isBand]= df['expfactor'][isBand] * t0[band]
    return df.assign(tneed= tneed_arr)

  def constrain_tneed(self,df):
    """applies cleaning to ylabel"""
    cut= ((df['tneed'] > 0) &
          (df['tneed'] < 500))
    print('Cutting to %d/%d' % (len(df[cut]),len(df)))
    return df[cut]
 

class Split_TrainTest(object):
  def random_sampling(self,df,seed=7):
    """create training and test set grouping by night observed and randomly split 50/50
    
    Dont split up data from a given night. Group by night observed and split
    based on that
    """
    np.random.seed(seed)
    all_nights= list( set(df['night_obs']) )
    i_nights= np.arange(len(all_nights))
    np.random.shuffle(i_nights)
    ihalf= len(all_nights)//2
    i_nights_train,i_nights_test= i_nights[:ihalf], i_nights[ihalf:]
    # select all instances of these nights
    keep_train,keep_test= np.zeros(len(df),bool),np.zeros(len(df),bool)
    for i in i_nights_train:
        keep_train[ df['night_obs'] == all_nights[i]]=True
    for i in i_nights_test:
        keep_test[ df['night_obs'] == all_nights[i]]=True
    return df.loc[keep_train], df.loc[keep_test] 

def main():
  d= GetData(REPO_DIR)
  d.fetch()
  df = d.load()

  df= Clean().keep_science_exposures(df)
  df= Clean().add_night_obs(df)
  df= Clean().order_by_mjd(df)
  df= Clean().add_hr_obs(df)
  #df= Clean().drop_bad_transp(df, thresh=0.9)
  df= Clean().drop_nights_wfew_exposures(df, nexp=20)
  
  # TODO: remove duplicated expids b/c have > 1000 exposures on some nights

  df= AddYlabel().add_tneed(df)
  df= AddYlabel().constrain_tneed(df)

  df_train,df_test= Split_TrainTest().random_sampling(df)
  del df
  return df_train,df_test

if __name__ == '__main__':
  df_train,df_test = main()
  

