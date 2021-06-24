

import numpy as np
import pandas as pd

import json
# import pyahp # https://pyahp.gitbook.io/pyahp/
# from pyahp import errors, hierarchy, methods, parse, parser, utils, validate_model
# print(dir(pyahp)) # [..., errors, hierarchy, methods, parse, parser, utils, validate_model]



RIs = [0,0,0.52,0.89,1.12,1.26,1.36,1.41,1.46,1.49,1.52,1.54] # starts from order = 1
orderRI_table = dict(zip(range(1,len(RIs)+1),RIs))
# print(orderRI_table)



def construct_A_from_w(w):
  n = len(w)
  A = np.ones([n,n])
  for i in range(n):
    for j in range(n):
      A[i,j] = w[i]/w[j]
  return A



def get_largest_eigenvalue(A): # characteristic root == eigenvalue
  values, vectors = np.linalg.eig(A) # https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
  # print(values)
  # print(vectors)
  return np.max(values)



def consistency_test(w):
  n = len(w)
  A = construct_A_from_w(w)
  maxlambda = get_largest_eigenvalue(A)
  print('maxlambda:',maxlambda)
  # print('order:',n)

  ci = (maxlambda - n) / (n - 1)
  ri = orderRI_table[n]
  cr = ci/ri

  isJudgementMatrixConsistent = cr < 0.1 or ci == 0 # CR < 0.10 --> acceptable

  print('CI:',ci)
  print('RI:',ri)
  print('CR:',cr)
  print('judgment matrix is', 'consistent' if isJudgementMatrixConsistent else 'inconsistent')

  print()

  

if __name__ == "__main__":
  # data from sheet
  E_sub = np.array([8,	7.5,	3.5,	6.7,	4.7,	6.9,	2.1,	3.4,	5.5,	9])
  A_sub = np.array([5.5,	4,	5.2,	4.6])
  AE = np.array([6.5,	8.5])


  print('Personal information (A-*):',list(A_sub))
  consistency_test(A_sub)

  print('Credit information (E-*):',list(E_sub))
  consistency_test(E_sub)

  print('Major index categories (A, E):',list(AE))
  consistency_test(AE)