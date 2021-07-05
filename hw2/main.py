
import os, time
import pathlib
from multiprocessing import Process
import papermill as pm


def run(dir, file, args=None):
  a = "python " + dir + '/' + file
  if type(args) is list:
    for arg in args:
      a += ' ' + str(arg)
  elif type(args) is str:
    a += ' ' + args
  print(a)
  os.system(a)


def run_nb(dir, file, args=None):
  a = "papermill " + dir + '/' + file
  if type(args) is list:
    for arg in args:
      a += ' ' + str(arg)
  elif type(args) is str:
    a += ' ' + args
  print(a)
  os.system(a)


def runMulti(dir, file, args=None):
  startDates = args[0]
  endDates = args[1]
  startDates = [startDates] if type(startDates) is not list else startDates
  endDates = [endDates] if type(endDates) is not list else endDates
  proc = []
  for i in range(min(len(startDates),len(endDates))):
    p = Process(target=run, args=(dir, file, [startDates[i], endDates[i]] + args[2:] ))
    proc.append(p)
  for p in proc:
    p.start()
  for p in proc:
    p.join()


def timer_start():
  return time.time()
def timer_elapsed(t0):
  return time.time() - t0
def timer_restart(t0, msg):
  print(timer_elapsed(t0), msg)
  return timer_start()


def main():
  dir = str(pathlib.Path(__file__).parent.absolute())

  # cmap options: https://matplotlib.org/stable/tutorials/colors/colormaps.html
  # startYear startMonth endYear endMonth group dataset cmap

  t0 = timer_start()
  t1 = t0



  # run(dir, 'read_acag_pm2-5.py', ['200001', '200012', 'YlOrRd', os.path.join('USA_states_counties', 'us_states'), '01-AL-Alabama.geojson', USAcounties, True])

  run(dir, 'TrainingaScorecardmodelusingAutoLoansDataset.py', [])


  t1 = timer_restart(t1, 'main total time')
  


if __name__ == "__main__":
  main()
