import pandas as pd
import zipfile

import os, netCDF4

import time
import datetime as dt

def save_df(df, folderPath, name, ext):
  if not os.path.isdir(folderPath):
    os.makedirs(folderPath)
    print('Missing directory created:', folderPath)
  print('save', os.path.join(folderPath, name + '.' + ext))
  if ext == 'csv':
    df.to_csv(os.path.join(folderPath, name + '.csv'), index=False  )
  elif ext == 'hdf5':
    df.to_hdf(os.path.join(folderPath, name + '.hdf5'), key='data', mode='w', format='fixed')
  elif ext == 'zip':
    df.to_csv(os.path.join(folderPath, name + '.zip'), compression=dict(method='zip',archive_name=name + '.csv'), index=False)

def read_df(folderPath, name, ext):
  # print('read', os.path.join(folderPath, name + '.' + ext))
  if ext == 'csv':
    return pd.read_csv(os.path.join(folderPath, name + '.csv'))
  elif ext == 'hdf5':
    return pd.read_hdf(os.path.join(folderPath, name + '.hdf5'))
  elif ext == 'nc':
    return netCDF4.Dataset(os.path.join(folderPath, name + '.nc'))
  elif ext == 'zip':
    return pd.read_csv(zipfile.ZipFile(os.path.join(folderPath, name + '.zip')).open(name + '.csv', 'r'), index_col=False)

def is_in_dir(folderPath, name, ext):
  return name + '.' + ext in os.listdir(folderPath)

def save_plt(plt, folderPath, name, ext):
  plt.savefig(os.path.join(folderPath, name + '.' + ext), format=ext)

def utc_time_filename():
  return dt.datetime.utcnow().strftime('%Y.%m.%d-%H.%M.%S')

def timer_start():
  return time.time()
def timer_elapsed(t0):
  return time.time() - t0
def timer_restart(t0, msg):
  print(timer_elapsed(t0), msg)
  return timer_start()