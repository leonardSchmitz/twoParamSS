import linear_combination.linear_combination as lc
import numpy as np
from itertools import permutations
from sympy import Rational

def totuple(a):
  try:
    return tuple(totuple(i) for i in a)
  except TypeError:
    return a

def is_composition(_a):
  if _a == ():
    return True 
  a = np.array(_a)
  m,n,d = a.shape
  for i in range(m):
    if np.count_nonzero(a[i,:,:])==0:
      return False 
  for j in range(n):
    if np.count_nonzero(a[:,j,:])==0:
      return False 
  return True 

def row_concat(_letter,_word):
  ret = []
  for monomial in _word:
    ret.append(np.concatenate((_letter,monomial),axis=0))
  return ret 

def row_shuffle(_alpha, _beta, _qsh): 
  if _alpha.shape[0]==0:
    return [_beta]
  if _beta.shape[0]==0:
    return [_alpha]
  d = len(_alpha[0,0,:])    
  a = _alpha[0,:,:].reshape(1,len(_alpha[0,:,0]),d)
  b = _beta[0,:,:].reshape(1,len(_beta[0,:,0]),d)
  w1 = _alpha[1:,:,:].reshape(len(_alpha[:,0,0])-1,len(_alpha[0,:,0]),d)
  w2 = _beta[1:,:,:].reshape(len(_beta[:,0,0])-1,len(_beta[0,:,0]),d)
  ret = row_concat(a,row_shuffle(w1,_beta,_qsh)) + row_concat(b,row_shuffle(_alpha,w2,_qsh)) 
  if _qsh:
     ret = ret + row_concat(a+b,row_shuffle(w1,w2,_qsh))
  return ret

def col_concat(_letter,_word):
  ret = []
  for monomial in _word:
    ret.append(np.concatenate((_letter,monomial),axis=1))
  return ret 

def col_shuffle(_alpha, _beta, _qsh): 
  if _alpha.shape[1]==0:
    return [_beta]
  if _beta.shape[1]==0:
    return [_alpha]
  d = len(_alpha[0,0,:])    
  a = _alpha[:,0,:].reshape(len(_alpha[:,0,0]),1,d)
  b = _beta[:,0,:].reshape(len(_beta[:,0,0]),1,d)
  w1 = _alpha[:,1:,:].reshape(len(_alpha[:,0,0]),len(_alpha[0,:,0])-1,d)
  w2 = _beta[:,1:,:].reshape(len(_beta[:,0,0]),len(_beta[0,:,0])-1,d)
  ret =  col_concat(a,col_shuffle(w1,_beta, _qsh)) + col_concat(b,col_shuffle(_alpha,w2,_qsh)) 
  if _qsh:
    ret = ret + col_concat(a+b,col_shuffle(w1,w2,_qsh))
  return ret 

def shuffle(_alpha,_beta, _qsh):
  # input: compositions _alpja,_beta 
  #        as 3-nested tuples
  # output: ret 
  #         which is list of compositions (as arrays) such that 
  #         _alpha qsh _beta = \sum_{g\in\ls} g 
  alpha = np.array(_alpha)
  beta = np.array(_beta)
  if _alpha == ():
    if _beta == ():
      return [()]
    return [beta]
  if _beta == ():
    return [alpha]
  (m,n,d) = alpha.shape
  (s,t,d) = beta.shape
  block1 = np.concatenate((alpha,np.zeros(shape=(s,n,d),dtype=int)),axis=0)
  block2 = np.concatenate((np.zeros(shape=(m,t,d),dtype=int),beta),axis=0)
  linKomb = col_shuffle(block1,block2,_qsh)
  ret = []
  for elem in linKomb:
    gamma = elem[0:m,:,:]
    delta = elem[m:,:,:]
    ret.extend(row_shuffle(gamma,delta,_qsh))
  return ret

def connected_decomposition(a):
  # inpit: a in mxnxd
  # list of b_i in u_i x v_i x d
  m,n,d = a.shape
  if (m==1 or n==1):
    return [a]
  for i in range(m-1):
    for j in range(n-1):
      a11 = a[:(i+1),:(j+1),:]
      a21 = a[(i+1):,:(j+1),:]
      a12 = a[:(i+1),(j+1):,:]
      a22 = a[(i+1):,(j+1):,:]
      if (np.count_nonzero(a21)==0 and np.count_nonzero(a12)==0):
         #return connected_decomposition(a22).insert(0,a11)
         return [a11]+connected_decomposition(a22)
  return [a]

def is_connected(a):
  return (1 == len(connected_decomposition(a)))

def block_shuffle(_alpha,_beta):
  # input: compositions _alpja,_beta 
  #        as 3-nested tuples
  # output: ret 
  #         which is list of compositions (as arrays) such that 
  #         _alpha qsh _beta = \sum_{g\in\ls} g 
  alpha = np.array(_alpha)
  beta = np.array(_beta)
  if _alpha == ():
    if _beta == ():
      return [()]
    return [beta]
  if _beta == ():
    return [alpha]
  return block_shuffle_help(connected_decomposition(alpha),
                            connected_decomposition(beta))

def block_diag(a,b):
  m,n,d = a.shape
  s,t,d = b.shape
  res = np.zeros((m+s,n+t,d),dtype=int)
  res[:m,:n,:] = a
  res[m:,n:,:] = b
  return res

def block_diag_ls(l):
  res = l[0]
  for i in range(len(l)-1):
     res = block_diag(res,l[i+1])
  return res

def block_concat(_letter,_word):
  # _letter is array
  # _word is list of arrays
  # _return is list of arrays, apply diag entrywise  
  ret = []
  for monomial in _word:
    ret.append(block_diag(_letter,monomial))
  return ret 
   
def block_shuffle_help(_alpha,_beta):
  # input: list _alpja,_beta of connected arrays
  # output: ret 
  #         which is list of compositions (as arrays) such that 
  #         _alpha qsh _beta = \sum_{g\in\ls} g 
  if _alpha==[]:
    return [block_diag_ls(_beta.copy())]
  if _beta==[]:
    return [block_diag_ls(_alpha.copy())]
  # both [] is not possible here 
  a = _alpha[0]
  b = _beta[0]
  w1 = _alpha[1:].copy()
  w2 = _beta[1:].copy()
  return block_concat(a,block_shuffle_help(w1,_beta.copy())) + block_concat(b,block_shuffle_help(_alpha.copy(),w2))


## helpers for Phi

def compositions(k, n):
    def help_comp(k, n):
        C = set()
        if k == 1:
            for i in range(n):
                C.add((i+1,))
        else:
            for i in range(n):
                i=i+1
                for j in help_comp(k=k-1, n=n):
                    C.add((i,)+(j))
        return C
    C = set()
    if k == 1:
        C.add((n,))
    else:
        for i in range(n):
            i=i+1
            for j in help_comp(k=k-1, n=n):
                if sum(list((i,)+(j))) == n:
                    C.add((i,)+(j))
    return C

def fact_of_brews(U):
  sums = np.sum(U,axis=1)
  fact = [np.math.factorial(i) for i in sums]#.astype(U.dtype)
  return np.prod(fact,axis=0)

def prod_of_brews(U):
  sums = np.sum(U,axis=1)
  #fact = [np.math.factorial(i) for i in sums]#.astype(U.dtype)
  return np.prod(sums,axis=0)

def comp_to_brew(c):
   ell = sum(c)
   v = len(c)
   ret = np.zeros((v,ell),dtype=np.int64)
   sum_cs = 0
   for i in range(v):
     c_i = c[i]
     for j in range(sum_cs,sum_cs+c_i):
       ret[i,j] = 1
     sum_cs = sum_cs + c_i
   return ret

def is_diagonal(a):
    (m,n,d) = a.shape
    if m != n:
       return False 
    for i in range(m):
       for j in range(i):
           if sum(a[i,j,:]) != 0:
               return False 
       for j in range(i+1,m):
           if sum(a[i,j,:]) != 0:
               return False 
    return True 
                
def U_a_VT(U,a,V):
    U_a = np.einsum('pi,ijk->pjk',U,a)
    return np.einsum('pj,ijk->ipk',V,U_a)

# helpers for not block shuffle 

def remove_from_list(base_arrays, test_array):
    for index in range(len(base_arrays)):
        if np.array_equal(base_arrays[index], test_array):
            base_arrays.pop(index)
            return 
    raise ValueError('remove_from_array(array, x): x not in array')

def member_in_list(base_arrays, test_array):
    for index in range(len(base_arrays)):
        if np.array_equal(base_arrays[index], test_array):
            return True
    return False

def cap_lists(ls1,ls2):
    res = []
    for el in ls1:
       if member_in_list(ls2,el):
          remove_from_list(ls2,el)
          res.append(el)
    return res

def not_block_shuffle_2(u1,u2):
    ls_sh = shuffle(u1,u2,False)
    ls_block_sh = block_shuffle(u1,u2)
    for el in ls_block_sh: 
      remove_from_list(ls_sh,el)
    #for el in ls_sh: 
    #  if not is_connected(el):
    #    remove_from_list(ls_sh,el)
    #return ls_sh
    return filter(is_connected,ls_sh)

def not_block_shuffle_3(u1,u2,u3):
    #(u1 . u2) . u3
    ls_sh = shuffle(u1,u2,False)
    ls_block_sh = block_shuffle(u1,u2)
    for el in ls_block_sh: 
      remove_from_list(ls_sh,el)
    res_u1u2u3 = []
    for el in ls_sh:
      ls_sh_2 = shuffle(totuple(el),u3,False)
      ls_block_sh_2 = block_shuffle(totuple(el),u3)
      for el2 in ls_block_sh_2: 
        remove_from_list(ls_sh_2,el2)
      res_u1u2u3 = res_u1u2u3 + ls_sh_2
    #(u2 . u3) . u1
    ls_sh = shuffle(u2,u3,False)
    ls_block_sh = block_shuffle(u2,u3)
    for el in ls_block_sh: 
      remove_from_list(ls_sh,el)
    res_u2u3u1 = []
    for el in ls_sh:
      ls_sh_2 = shuffle(totuple(el),u1,False)
      ls_block_sh_2 = block_shuffle(totuple(el),u1)
      for el2 in ls_block_sh_2: 
        remove_from_list(ls_sh_2,el2)
      res_u2u3u1 = res_u2u3u1 + ls_sh_2
    #(u3 . u1) . u2
    ls_sh = shuffle(u3,u1,False)
    ls_block_sh = block_shuffle(u3,u1)
    for el in ls_block_sh: 
      remove_from_list(ls_sh,el)
    res_u3u1u2 = []
    for el in ls_sh:
      ls_sh_2 = shuffle(totuple(el),u2,False)
      ls_block_sh_2 = block_shuffle(totuple(el),u2)
      for el2 in ls_block_sh_2: 
        remove_from_list(ls_sh_2,el2)
      res_u3u1u2 = res_u3u1u2 + ls_sh_2 
    res = cap_lists(cap_lists(res_u1u2u3, res_u2u3u1),  res_u3u1u2)
    #res = cap_lists(res_u1u2u3, res_u2u3u1)
    return filter(is_connected,res)

class comp(tuple):
    #def parser():
    #def from_str(s):
    #def from_list(ell):

    @staticmethod
    def from_array(a):
        return lc.lift(comp(totuple(a)))

   
    def __mul__(ell_1,ell_2):
    #    """Quasi-suffle product."""
         for a in shuffle(ell_1,ell_2,True):
             yield (comp(totuple(a)),Rational(1,1))

    def mul_sh(ell_1,ell_2):
    #    """Shuffle product."""
         for a in shuffle(ell_1,ell_2,False):
             yield (comp(totuple(a)),Rational(1,1))

    def mul_block_sh(ell_1,ell_2):
    #    """Shuffle product of Blocks."""
        for a in block_shuffle(ell_1,ell_2):
            yield (comp(totuple(a)),Rational(1,1))

    def mul_not_block_sh(*args):
        l = len(args) 
        if l == 2:
           for a in not_block_shuffle_2(args[0],args[1]):
              yield (comp(totuple(a)),Rational(1,1))
        elif l == 3:
           for a in not_block_shuffle_3(args[0],args[1],args[2]):
              yield (comp(totuple(a)),Rational(1,1))
        else: 
           print(" not block shuffle is not implemented")

    def coproduct(self):
    #    """Deconcatenation coproduct with respect to block_diag."""
        if self == () :
           yield (lc.Tensor( (comp(),comp()) ) , 1)
        else: 
            yield (lc.Tensor( (comp(self),comp()) ), 1)
            yield (lc.Tensor( (comp(),comp(self)) ), 1)
            ls = connected_decomposition(np.array(self))
            for i in range(len(ls)-1):
                yield (lc.Tensor( (comp(totuple(block_diag_ls(ls[:i+1]))), comp(totuple(block_diag_ls(ls[i+1:])))) ), 1)

    #def antipode(self):

    @staticmethod
    def unit(a=1):
        return lc.LinearCombination( {comp():a} )

    #def counit(self):
    #def __add__(self,other):
    #def __eq__(self, other):
    #def __hash__(self):
    #def __getitem__(self, key):
    #def __str__(self):
    #def weight(self):
    #def area(sw1,sw2):

class diagComp(tuple):

   # def parser():
   # def from_str(s):
   # def from_list(ell):

    @staticmethod
    def from_array(a):
        return lc.lift(diagComp(totuple(a)))

    def __mul__(self,other):
        """Concatenation product."""
        if (self == ()):
           if (other == ()):
              yield ((),1)
           else: 
              yield (other,1)
        else: 
           if (other == ()):
              yield (self,1)
           else: 
              yield (totuple(block_diag(np.array(self),np.array(other))),1)

    def coproduct(self):
    #     """Deshuffle coproduct."""
    # input: composition (as tuple)
    # output: list of pairs  c,d of compositions (as tuples)
    # such that shape(b),shape(c)<= shape(a) and min(b)+max(c),max(b)+min(c)<=max(a)
      if self == () :
         yield (lc.Tensor( (diagComp(),diagComp()) ) , 1)
      else: 
         yield (lc.Tensor( (diagComp(()), diagComp(self))) , 1)
         yield (lc.Tensor( (diagComp(self), diagComp(()))) , 1)
         m,n = np.array(self).shape
         for i1 in range(1,m+1):
           for i2 in range(1,m+1):         ## here possibly little speed-up (very small matrices are not possible)
             for j1 in range(1,n+1):
               for j2 in range(1,n+1):     ## here possibly little speed-up (very small...) 
                 for a in allComp(i1,j1,np.amax(np.array(self))):
                   for b in allComp(i2,j2,np.amax(np.array(self))):
                     qs = comp.from_array(a) * comp.from_array(b) 
                     try:
                        yield (lc.Tensor( (diagComp(totuple(a)), diagComp(totuple(b)))) , qs[self])
                     except:
                        yield (lc.Tensor( (diagComp(()), diagComp(totuple(())))) , 0)
    # TODO as soon as bi-algebra identiy is shown, use it for efficency

   # def antipode(self):

    @staticmethod
    def unit(a=1):
        return lc.LinearCombination( {diagComp():a} )

   # #def counit(self):
   # def counit(self):
   # def __add__(self,other):
   # def __eq__(self, other):
   # def __hash__(self):
   # def __str__(self):
   # def weight(self):
   # def lie_bracket(cw1,cw2):


