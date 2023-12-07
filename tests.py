from composition_algebra import *
from twoParameterSS import *
from PIL import Image

## ================ Tests ============================================================

# test one-to-one correspondece with totuple and array initialization
tup = (((1,2),(2,2)),((1,2),(3,1)))
assert(len(tup[0][0])==2)
assert(totuple(np.array(tup))==tup)
tup = (((1,),(2,)),((1,),(3,)))
assert(len(tup[0][0])==1)
assert(totuple(np.array(tup))==tup)
a = np.array((((1,2,2),(2,2,1)),((1,2,1),(3,1,4))))
assert(a.shape==(2,2,3))
assert(np.allclose(a,np.array(totuple(a)),atol=0.0001))
a = np.array((((1,),(2,)),((1,),(3,))))
assert(a.shape==(2,2,1))
assert(np.allclose(a,np.array(totuple(a)),atol=0.0001))

# test allArraysOfLengthMaxEnrty (for de-quasishuffle identity) 
#assert((3+1)**4==len(allArraysOfLenfthMaxEntry(4,3)))  # zeros are allowed 
#assert((3+1)**3==len(allArraysOfLenfthMaxEntry(3,3)))

# test is_composition
assert(is_composition(()))
a = np.array((((2,),(0,)),((1,),(0,))))
assert(not is_composition(totuple(a)))
a = np.array((((2,1),(0,0)),((1,1),(0,0))))
assert(not is_composition(totuple(a)))
a = np.array((((2,1),(0,1)),((1,1),(0,0))))
assert(is_composition(totuple(a)))
a = np.array((((2,),(1,)),((0,),(0,))))
assert(not is_composition(totuple(a)))
a = np.array((((2,1),(1,1)),((0,1),(0,0))))
assert(is_composition(totuple(a)))

# test diagonal concatenation and connected decompositions 
d = 1
a = np.array((((2,),(0,)),((0,),(1,))),dtype=np.int32)
assert(is_composition(totuple(a)))
cda = connected_decomposition(a)
assert(len(cda)==2)
assert(np.array_equal(cda[0],np.array((2,)).reshape(1,1,d)))
assert(np.array_equal(cda[1],np.array((1,)).reshape(1,1,d)))
assert(np.array_equal(block_diag_ls(cda),a))
#test lemma 2.33 (Chen)
S = 4
T = 3
z = evZ(np.random.rand(S,T,d))
z2 = evZ(np.random.rand(S,T,d))
zz2 = conc_diag_evZ(z,z2)
SSzz2a_chen = SS_naiv(z,a) + SS_naiv(z,cda[0])*SS_naiv(z2,cda[1]) + SS_naiv(z2,a)
assert(np.isclose(SSzz2a_chen, SS_naiv(zz2,a),atol=1e-09))
a = np.array((((0,),(2,)),((2,),(0,))))
assert(is_composition(totuple(a)))
cda = connected_decomposition(a)
assert(len(cda)==1)
assert(np.array_equal(cda[0],a))
assert(np.array_equal(block_diag_ls(cda),a))
#test lemma 2.33 (Chen)
assert(np.isclose(SS_naiv(z,a) + SS_naiv(z2,a), SS_naiv(zz2,a),atol=1e-09))
a1 = np.array(((2,0,0,0),(1,2,0,0),(0,0,5,0),(0,0,0,1))).reshape(4,4,d)
assert(is_composition(totuple(a1)))
cda1 = connected_decomposition(a1)
assert(np.array_equal(cda1[0],np.array(((2,0),(1,2))).reshape(2,2,d)))
assert(np.array_equal(cda1[2],np.array([1]).reshape(1,1,d)))
assert(np.array_equal(block_diag_ls(cda1),a1))
assert(np.array_equal(block_diag(np.array(((2,0),(1,2))).reshape(2,2,d),
                                 np.array(((5,0),(0,1))).reshape(2,2,d)),
                      a1))
assert(np.array_equal(block_diag_ls([np.array(((2,0),(1,2))).reshape(2,2,d),
                                     np.array([5]).reshape(1,1,d),
                                     np.array([1]).reshape(1,1,d)]),
                      a1))
#test lemma 2.33 (Chen)
SSzz2a_chen = SS_naiv(z,a1) + SS_naiv(z,block_diag(cda1[0],cda1[1]))*SS_naiv(z2,cda1[2]) 
SSzz2a_chen = SSzz2a_chen + SS_naiv(z,cda1[0])*SS_naiv(z2,block_diag(cda1[1],cda1[2])) 
SSzz2a_chen = SSzz2a_chen + SS_naiv(z2,a1)
assert(np.isclose(SSzz2a_chen, SS_naiv(zz2,a1),atol=1e-09))
d = 2
a = np.array((((2,1),(0,0)),((0,0),(1,1))))
cda = connected_decomposition(a)
assert(np.array_equal(cda[0],np.array((2,1)).reshape(1,1,d)))
assert(np.array_equal(block_diag_ls(cda),a))
#test lemma 2.33 (Chen)
z = evZ(np.random.rand(S,T,d))
z2 = evZ(np.random.rand(S,T,d))
zz2 = conc_diag_evZ(z,z2)
SSzz2a_chen = SS_naiv(z,a) + SS_naiv(z,cda[0])*SS_naiv(z2,cda[1]) + SS_naiv(z2,a)
assert(np.isclose(SSzz2a_chen, SS_naiv(zz2,a),atol=1e-09))
a = np.array((((2,1),(0,1)),((0,0),(1,1))))
cda = connected_decomposition(a)
assert(len(cda)==1)
assert(np.array_equal(block_diag_ls(cda),a))
#test lemma 2.33 (Chen)
assert(np.isclose(SS_naiv(z,a) + SS_naiv(z2,a), SS_naiv(zz2,a),atol=1e-09))

## ================ Tests comp =========================
# test Hopfalgebra properties (theorem 2.10)

# unit
a2 = np.array([3],dtype=np.int32).reshape(1,1,1)
a4 = np.array([5],dtype=np.int32).reshape(1,1,1)
a2tv = comp.from_array(a2)
a4tv = comp.from_array(a4)
a1 = np.array(((2,0,0,0),(1,2,0,0),(0,0,5,0),(0,0,0,1))).reshape(4,4,1)
lc_1 = comp.from_array(a1) + a2tv
lc_2 = a4tv
assert(lc_1 == comp.unit() * lc_1)
assert(lc_1 == lc_1 * comp.unit())
assert(comp.unit() == comp.unit() * comp.unit())
a2tv_d2 = comp.from_array(np.array((2,1),dtype=np.int32).reshape(1,1,2))
a4tv_d2 = comp.from_array(np.array((2,1,1,4),dtype=np.int32).reshape(2,1,2))
lc_1_d2 = a2tv_d2 + a4tv_d2
lc_2_d2 = comp.from_array(np.array((2,1,1,4,3,3,1,5),dtype=np.int32).reshape(2,2,2))
assert(lc_1_d2 == comp.unit() * lc_1_d2)
assert(lc_1_d2 == lc_1_d2 * comp.unit())



# coassociativity
assert(lc_1.apply_linear_function( comp.coproduct ).apply_linear_function( lc.Tensor.fn_otimes_linear( lc.id,
                                                                                                       comp.coproduct ) )
       == lc_1.apply_linear_function( comp.coproduct ).apply_linear_function( lc.Tensor.fn_otimes_linear( comp.coproduct, lc.id ) ))
assert(lc_1_d2.apply_linear_function( comp.coproduct ).apply_linear_function( lc.Tensor.fn_otimes_linear( lc.id,
                                                                                                       comp.coproduct ) )
       == lc_1_d2.apply_linear_function( comp.coproduct ).apply_linear_function( lc.Tensor.fn_otimes_linear( comp.coproduct, lc.id ) ))

# bialgebra relation 
# condition on product and coproduct: \Delta( \tau \tau' ) = \Delta(\tau) \Delta(\tau').
assert((lc_1 * lc_2).apply_linear_function( comp.coproduct )
            == lc_1.apply_linear_function( comp.coproduct )* lc_2.apply_linear_function( comp.coproduct ))
assert((lc_1 * lc_1).apply_linear_function( comp.coproduct )
            == lc_1.apply_linear_function( comp.coproduct )* lc_1.apply_linear_function( comp.coproduct ))
assert((lc_1_d2 * lc_2_d2).apply_linear_function( comp.coproduct )
            == lc_1_d2.apply_linear_function( comp.coproduct )* lc_2_d2.apply_linear_function( comp.coproduct ))
lc_1 = comp.from_array(np.array([1,2],dtype=np.int32).reshape(2,1,1))
lc_2 = comp.from_array(np.array(((0,2),(1,0))).reshape(2,2,1))
assert((lc_1 * lc_2).apply_linear_function( comp.coproduct )
            == lc_1.apply_linear_function( comp.coproduct )* lc_2.apply_linear_function( comp.coproduct ))

# test soundness of quasi-shuffle 
# calculation in introduction (section 1.2, d=1)
a_id = comp.from_array(np.array((((1,),(0,)),((0,),(1,))),dtype=np.int32))
a_aid = comp.from_array(np.array((((0,),(1,)),((1,),(0,))),dtype=np.int32))
a_11 = comp.from_array(np.array((1,1),dtype=np.int32).reshape(1,2,1))
a_11T = comp.from_array(np.array((1,1),dtype=np.int32).reshape(2,1,1))
a_2 = comp.from_array(np.array((2),dtype=np.int32).reshape(1,1,1))
a_1 = comp.from_array(np.array((1),dtype=np.int32).reshape(1,1,1))
assert(a_1*a_1 == 2*a_id + 2*a_aid + 2*a_11 + 2*a_11T + a_2)

# test associativery of the quasi-shuffle 
a_1 = comp.from_array(np.array((1),dtype=np.int32).reshape(1,1,1))
a_2 = comp.from_array(np.array((2),dtype=np.int32).reshape(1,1,1))
a_3 = comp.from_array(np.array((3),dtype=np.int32).reshape(1,1,1))
a_123 = a_1*(a_2*a_3)
a_123v = (a_1*a_2)*a_3
assert( a_123 == a_123v)


# example 2.7  (d=2)
a_1 = comp.from_array(np.array((1,0),dtype=np.int32).reshape(1,1,2))
a_2 = comp.from_array(np.array((0,1),dtype=np.int32).reshape(1,1,2))
a_1002 = comp.from_array(np.array((((1,0),(0,0)),((0,0),(0,1))),dtype=np.int32))
a_2001 = comp.from_array(np.array((((0,1),(0,0)),((0,0),(1,0))),dtype=np.int32))
a_0120 = comp.from_array(np.array((((0,0),(1,0)),((0,1),(0,0))),dtype=np.int32))
a_0210 = comp.from_array(np.array((((0,0),(0,1)),((1,0),(0,0))),dtype=np.int32))
a_12 = comp.from_array(np.array((((1,0),(0,1)),),dtype=np.int32))
a_21 = comp.from_array(np.array((((0,1),(1,0)),),dtype=np.int32))
a_12T = comp.from_array(np.array((((1,0),),((0,1),)),dtype=np.int32))
a_21T = comp.from_array(np.array((((0,1),),((1,0),)),dtype=np.int32))
a_1u2 = comp.from_array(np.array((1,1),dtype=np.int32).reshape(1,1,2))
assert(a_1*a_2 == a_1002 + a_2001 + a_0120 + a_0210 + a_12 + a_21+ a_12T + a_21T + a_1u2)


# example 2.13 (d=3)
a = 1
b = 1
c = 1
for [a,b,c] in [[1,1,1],[2,1,4],[2,2,3]]:
  a1 = np.array((a,0,0),dtype=np.int32).reshape(1,1,3)
  c1 = comp.from_array(a1)
  a2 = np.array((((0,b,0),),((0,0,c),)),dtype=np.int32)
  c2 = comp.from_array(a2)
  b1 = comp.from_array(block_diag(a1,a2))
  b2 = comp.from_array(np.array((((a,0,0),(0,b,0)),
                                 ((0,0,0),(0,0,c))),dtype=np.int32))
  b3 = comp.from_array(np.array((((0,0,0),(0,b,0)),
                                 ((a,0,0),(0,0,0)),
                                 ((0,0,0),(0,0,c))),dtype=np.int32))
  b4 = comp.from_array(np.array((((0,0,0),(0,b,0)),
                                 ((a,0,0),(0,0,c))),dtype=np.int32))
  b5 = comp.from_array(np.array((((0,0,0),(0,b,0)),
                                 ((0,0,0),(0,0,c)),
                                 ((a,0,0),(0,0,0))),dtype=np.int32))
  b6 = comp.from_array(np.array((((a,0,0),),
                                 ((0,b,0),),
                                 ((0,0,c),)),dtype=np.int32))
  b7 = comp.from_array(np.array((((a,b,0),),
                                 ((0,0,c),)),dtype=np.int32))
  b8 = comp.from_array(np.array((((0,b,0),),
                                 ((a,0,0),),
                                 ((0,0,c),)),dtype=np.int32))
  b9 = comp.from_array(np.array((((0,b,0),),
                                 ((a,0,c),)),dtype=np.int32))
  bA = comp.from_array(np.array((((0,b,0),),
                                 ((0,0,c),),
                                 ((a,0,0),)),dtype=np.int32))
  bB = comp.from_array(np.array((((0,0,0),(a,0,0)),
                                 ((0,b,0),(0,0,0)),
                                 ((0,0,c),(0,0,0))),dtype=np.int32))
  bC = comp.from_array(np.array((((0,b,0),(a,0,0)),
                                 ((0,0,c),(0,0,0))),dtype=np.int32))
  bD = comp.from_array(np.array((((0,b,0),(0,0,0)),
                                 ((0,0,0),(a,0,0)),
                                 ((0,0,c),(0,0,0))),dtype=np.int32))
  bE = comp.from_array(np.array((((0,b,0),(0,0,0)),
                                 ((0,0,c),(a,0,0))),dtype=np.int32))
  bF = comp.from_array(block_diag(a2,a1))
  assert(c1*c2 == b1+b2+b3+b4+b5+b6+b7+b8+b9+bA+bB+bC+bD+bE+bF)

# test soundness of coproduct (deconcatenaton based on block_diag)
#test example 2.9
a1 = np.array((((1,0,0),(0,1,0),(0,0,0),(0,0,0),(0,0,0)),
               ((0,0,0),(0,0,0),(0,0,1),(0,0,0),(0,0,0)),
               ((0,0,0),(0,0,0),(0,0,0),(0,1,0),(1,1,0)),
               ((0,0,0),(0,0,0),(0,0,0),(0,1,0),(0,1,0))))
assert(is_composition(totuple(a1)))
cda1 = connected_decomposition(a1)
assert(len(cda1)==3)
assert(np.array_equal(block_diag_ls(cda1),a1))
assert(np.array_equal(cda1[0],np.array((((1,0,0),(0,1,0)),))))
lc_coprod_a1 = lc.LinearCombination({lc.Tensor((totuple(a1),
                                                ()) ) : 1,
                                     lc.Tensor((totuple(block_diag(cda1[0],
                                                                   cda1[1])),
                                                totuple(cda1[2]))) : 1,
                                     lc.Tensor((totuple(cda1[0]),
                                                totuple(block_diag(cda1[1],
                                                                   cda1[2])))) : 1,
                                     lc.Tensor(((),
                                                totuple(a1))) : 1} )
assert(lc_coprod_a1 == comp.from_array(a1).apply_linear_function( comp.coproduct ))



##### Test two-parameter sums signature ####

# numeric tests according to Example 2.5
d = 1
#a = np.random.randint(2,size=(1,1,d),dtype = np.int8)
m = 1
n = 1
a = np.reshape(np.array([1],dtype=np.int8),[m,n,d])
S = 2
T = 2
z = np.reshape(np.array([[2.,1.],[3.,1.]]),[S,T,d])
assert(np.isclose(SS_2Param_naiv(-1,-1,S-1,T-1,z,a),
                  z[0,0,0]+z[1,0,0]+z[0,1,0]+z[1,1,0],rtol=1e-09))
# test theorem 2.18 (SS is invariant under zero insertion)
assert(np.isclose(SS_naiv(z,a),
                  SS_naiv(Zero_a0(1,z),a),rtol=1e-09))
assert(np.isclose(SS_naiv(z,a),
                  SS_naiv(Zero_a0(2,z),a),rtol=1e-09))
assert(np.isclose(SS_naiv(z,a),
                  SS_naiv(Zero_a1(1,z),a),rtol=1e-09))
# numeric tests according to Example 2.5
m = 2
n = 2
b = np.reshape(np.array([[1,2],[0,1]],dtype=np.int8),[m,n,d])
S = 3
T = 2
z = np.reshape(np.array([[2.,1.],[0.,7.],[3.,1.]]),[S,T,d])
assert(np.isclose(SS_2Param_naiv(-1,-1,S-1,T-1,z,b),
                  z[0,0,0]*z[0,1,0]**2*z[1,1,0]+z[0,0,0]*z[0,1,0]**2*z[2,1,0],
                  rtol=1e-09))
# test theorem 2.18 (SS is invariant under zero insertion)
assert(np.isclose(SS_naiv(z,b),
                  SS_naiv(Zero_a0(1,z),b),rtol=1e-09))
assert(np.isclose(SS_naiv(z,b),
                  SS_naiv(Zero_a0(2,z),b),rtol=1e-09))
assert(np.isclose(SS_naiv(z,b),
                  SS_naiv(Zero_a1(1,z),b),rtol=1e-09))

#test theorem 2.11 (quais-shuffle identity)
a_qsh_b = shuffle(a,b,True)
res = SS_2Param_naiv_mat(z,a_qsh_b[0])
for i in range(1,len(a_qsh_b)):
  res = res + SS_2Param_naiv_mat(z,a_qsh_b[i])
assert(np.allclose(res,SS_2Param_naiv_mat(z,a)*SS_2Param_naiv_mat(z,b),rtol=1e-09))
S = 5
T = 4
z = np.random.rand(S,T,d)
b_qsh_b = shuffle(b,b,True)
res = SS_2Param_naiv_mat(z,b_qsh_b[0])
for i in range(1,len(b_qsh_b)):
  res = res + SS_2Param_naiv_mat(z,b_qsh_b[i])
assert(np.allclose(res,SS_2Param_naiv_mat(z,b)*SS_2Param_naiv_mat(z,b),rtol=1e-09))
d = 3
a = np.random.randint(3, size=(2,2,d))+1
b = np.random.randint(3, size=(3,3,d))+1
a_qsh_b = shuffle(a,b,True)
T = 5
z = np.random.rand(S,T,d)
res = SS_2Param_naiv_mat(z,a_qsh_b[0])
for i in range(1,len(a_qsh_b)):
  res = res + SS_2Param_naiv_mat(z,a_qsh_b[i])
assert(np.allclose(res,SS_2Param_naiv_mat(z,a)*SS_2Param_naiv_mat(z,b),rtol=1e-09))

# test theorem 4.5 (efficient computation of SS w.r.t. chained 1x1 compositions)
d = 1
m = 2
n = 2
a_diagRep = np.reshape(np.array([1,2],dtype=np.int8),[m,d])
a = np.stack([np.diag(a_diagRep[:,j]) for j in range(d)],axis=2) # m=n=2
S = 3
T = 2
z = np.random.rand(S,T,d)
assert(np.allclose(SS_2Param_naiv_mat(z,a)[0,:,0,:],
                   SS_wDiag(z,a_diagRep),rtol=1e-09))
assert(np.allclose(SS_2Param_naiv_mat(z,a)[0,:,0,:],
                   SS_wChain(z,a_diagRep,[-1]),rtol=1e-09))
d = 2
m = 3
n = 3
a_diagRep = np.reshape(np.array([1,2,3,0,2,1],dtype=np.int8),[m,d])
a = np.stack([np.diag(a_diagRep[:,j]) for j in range(d)],axis=2) # m=n=2
S = 5
T = 6
z = np.random.rand(S,T,d)
assert(np.allclose(SS_2Param_naiv_mat(z,a)[0,:,0,:],
                   SS_wDiag(z,a_diagRep),rtol=1e-09))
assert(np.allclose(SS_2Param_naiv_mat(z,a)[0,:,0,:],
                   SS_wChain(z,a_diagRep,[-1,-1]),rtol=1e-09))
d = 2
m = 2
n = 3
l = 3
w_d0 = np.reshape(np.array([1,1,0,0,2,1],dtype=np.int8),[m,n])
w_d1 = np.reshape(np.array([0,2,0,0,4,2],dtype=np.int8),[m,n])
w = np.stack([w_d0,w_d1],axis=2)
w_diagRep = np.reshape(np.array([1,0,1,2,2,4,1,2],dtype=np.int8),[l+1,d])
assert(np.allclose(SS_2Param_naiv_mat(z,w)[0,:,0,:],
                   SS_wChain(z,w_diagRep,[1,0,1]),rtol=1e-09))

# test theorem 2.30 (SS is invariant under zero insertion)
# uses realistic (warped) pictures 
tree_init = evZ(np.array(Image.open('pictures/tree_init.bmp'))/100.)
S,T,d = tree_init.shape
tree_warped1 = evZ(np.array(Image.open('pictures/tree_warped1.bmp'))/100.)
S1,T1,d = tree_warped1.shape
tree_warped2 = evZ(np.array(Image.open('pictures/tree_warped2.bmp'))/100.)
S2,T2,d = tree_warped2.shape
l = 3
w_diagRep = np.random.randint(2, size=(l+1, d),dtype=np.int8)+1
assert(np.isclose(SS_wChain(delta(tree_init),w_diagRep,[1,0,1])[S-1,T-1],
                   SS_wChain(delta(tree_warped1),w_diagRep,[1,0,1])[S1-1,T1-1],rtol=1e-09))
assert(np.isclose(SS_wChain(delta(tree_init),w_diagRep,[1,0,1])[S-1,T-1],
                   SS_wChain(delta(tree_warped2),w_diagRep,[1,0,1])[S2-1,T2-1],rtol=1e-09))
# test lemma 2.19 (warp properties)
# part 2.
d = 3
S = 4
T = 5
z = evC(np.random.rand(S,T,d),np.random.rand(d))
for k in range(S):
  for j in range(T):
    assert(np.allclose(warp_a1(j,warp_a0(k,z)),
                       warp_a0(k,warp_a1(j,z)),
                       rtol=1e-09))
# part 3.
for j in range(S):
  for k in range(j+1,S):
    assert(np.allclose(warp_a0(k,warp_a0(j,z)),
                       warp_a0(j,warp_a0(k-1,z)),
                       rtol=1e-09))
for j in range(T):
  for k in range(j+1,T):
    assert(np.allclose(warp_a1(k,warp_a1(j,z)),
                       warp_a1(j,warp_a1(k-1,z)),
                       rtol=1e-09))
# test zero insertion 
d = 1
S = 2
T = 2
z = np.reshape(np.array([[2.,1.],[3.,1.]]),[S,T,d])
z_0rowAt_1 = np.reshape(np.array([[2.,1.],[0.,0.],[3.,1.]]),[S+1,T,d])
z_0rowAt_2 = np.reshape(np.array([[2.,1.],[3.,1.],[0.,0.]]),[S+1,T,d])
z_0colAt_1 = np.reshape(np.array([[2.,0.,1.],[3.,0.,1.]]),[S,T+1,d])
assert(np.allclose(Zero_a0(1,z),z_0rowAt_1,rtol=1e-09))
assert(np.allclose(Zero_a0(2,z),z_0rowAt_2,rtol=1e-09))
assert(np.allclose(Zero_a1(1,z),z_0colAt_1,rtol=1e-09))
# tests NF_Zero
z_init = evZ(np.random.rand(3,4,2))
z = Zero_a0(3,z_init)
z = Zero_a0(2,z)
z = Zero_a1(1,z)
assert(np.allclose(NF_Zero(z),z_init,atol= 0.0000001))
# tests NF_warp
x_init = evC(np.random.rand(3,4,2),np.random.rand(2))
x = warp_a0(2,x_init)
x = warp_a0(2,x)
x = warp_a1(1,x)
assert(np.allclose(NF_warp(x),x_init,atol= 0.0000001))
# test NF_kerDel
z_init = evC(np.random.rand(3,4,2),np.random.rand(2))
assert(np.allclose(NF_kerDel(z_init),sigma(delta(z_init)),atol = 0.0000001))
z_init = evZ(np.random.rand(3,4,2))
assert(np.allclose(NF_kerDel(z_init),sigma(delta(z_init)),atol = 0.0000001))
assert(np.allclose(NF_kerDel(z_init),delta(sigma(z_init)),atol = 0.0000001))
# test Lemma 2.15 
# part 2.
d = 3
S = 4
T = 5
z = evZ(np.random.rand(S,T,d))
for k in range(S+1):
  for j in range(T+1):
    assert(np.allclose(Zero_a1(j,Zero_a0(k,z)),
                       Zero_a0(k,Zero_a1(j,z)),
                       rtol=1e-09))
# part 3.
for j in range(S+1):
  for k in range(j+1,S+1):
    assert(np.allclose(Zero_a0(k,Zero_a0(j,z)),
                       Zero_a0(j,Zero_a0(k-1,z)),
                       rtol=1e-09))
for j in range(T+1):
  for k in range(j+1,T+1):
    assert(np.allclose(Zero_a1(k,Zero_a1(j,z)),
                       Zero_a1(j,Zero_a1(k-1,z)),
                       rtol=1e-09))
# test lemma 2.20
z = evZ(np.random.rand(S,T,d))
assert(np.allclose(sigma(delta(z)),z,rtol=1e-09))
assert(np.allclose(delta(sigma(z)),z,rtol=1e-09))
z = evC(np.random.rand(S,T,d),np.random.rand(d))
assert(np.allclose(delta(sigma(delta(z))),delta(z),rtol=1e-09))
# test lemma 2.22
# part 1.
d = 3
S = 4
T = 5
x = evC(np.random.rand(S,T,d),np.random.rand(d))
for k in range(S):
  assert(np.allclose(Zero_a0(k,delta(x)),delta(warp_a0(k,x)),rtol=1e-09))
for k in range(T):
  assert(np.allclose(Zero_a1(k,delta(x)),delta(warp_a1(k,x)),rtol=1e-09))
# part 2.
z = evZ(np.random.rand(S,T,d))
for k in range(S):
  assert(np.allclose(warp_a0(k,sigma(z)),sigma(Zero_a0(k,z)),rtol=1e-09))
for k in range(T):
  assert(np.allclose(warp_a1(k,sigma(z)),sigma(Zero_a1(k,z)),rtol=1e-09))
# part 3.       
assert(np.allclose(delta(NF_warp(x)),NF_Zero(delta(x)),atol=0.0000001))
# part 4.       
assert(np.allclose(sigma(NF_Zero(z)),NF_warp(sigma(z)),atol=0.0000001))
# part 5. 
x = evC(np.random.rand(S,T,d),np.random.rand(d))
for k in range(S):
  assert(np.allclose(warp_a0(k,NF_kerDel(x)),NF_kerDel(warp_a0(k,x)),atol=0.0000001))
for k in range(T):
  assert(np.allclose(warp_a1(k,NF_kerDel(x)),NF_kerDel(warp_a1(k,x)),atol=0.0000001))
# part 6.
x_init = evC(np.random.rand(3,4,2),np.random.rand(2))
x = warp_a0(2,x_init)
assert(np.allclose(delta(NF_warp(x)),NF_Zero(delta(x)),atol=1e-09))
x = warp_a0(2,x)
assert(np.allclose(delta(NF_warp(x)),NF_Zero(delta(x)),atol=1e-09))
x = warp_a0(2,x)
assert(np.allclose(NF_kerDel(NF_warp(x)),NF_warp(NF_kerDel(x)),atol=1e-09))
x = warp_a1(1,x)
assert(np.allclose(delta(NF_warp(x)),NF_Zero(delta(x)),atol=1e-09))
x = warp_a1(1,x)
assert(np.allclose(NF_kerDel(NF_warp(x)),NF_warp(NF_kerDel(x)),atol=1e-09))
# test example 2.32 (toy computation diagonal concatenation in evZ)
d = 1
z = evZ(np.array(((2.,7.),(2.,5.))).reshape(2,2,d))
z2 = evZ(np.array(((2.,2.),(1.,4.))).reshape(2,2,d))
res_evZ = evZ(np.array(((2.,7.,0.,0.),(2.,5.,0.,0.),(0.,0.,2.,2.),(0.,0.,1.,4.))).reshape(4,4,d))
assert(np.allclose(conc_diag_evZ(z,z2),res_evZ,atol=1e-09))
# test example 2.35 (toy computation diagonal concatenation in evC)
z = evC(np.array(((2.,7.),(2.,5.))).reshape(2,2,1),2.*np.ones(d))
z2 = evC(np.array(((2.,2.),(1.,4.))).reshape(2,2,1),0.*np.ones(d))
res_evC = evC(np.array(((2.,7.,2.,2.),(2.,5.,2.,2.),(2.,2.,2.,2.),(1.,1.,1.,4.))).reshape(4,4,d),0.*np.ones(d))
assert(np.allclose(conc_diag_evC(z,z2),res_evC,atol=1e-09))
## test lemma 5.6 (delta semi-group homomorphism) 
d = 3
z = evC(np.random.rand(3,4,d),np.random.rand(d))
z2 = evC(np.random.rand(5,2,d),np.random.rand(d))
# part 4.
assert(np.allclose(delta(conc_diag_evC(z,z2)),conc_diag_evZ(delta(z),delta(z2)),atol=1e-09))
z3 = evC(np.random.rand(2,2,d),np.random.rand(d))
# part 3.
assert(np.allclose(conc_diag_evC(conc_diag_evC(z,z2),z3),conc_diag_evC(z,conc_diag_evC(z2,z3)),atol=1e-09))
## test lemma 5.7 (sigma semi-group homomorphism) 
d = 3
z = evZ(np.random.rand(2,4,d))
z2 = evZ(np.random.rand(6,3,d))
# part 4.
assert(np.allclose(sigma(conc_diag_evZ(z,z2)),conc_diag_evC(sigma(z),sigma(z2)),atol=1e-09))
z3 = evC(np.random.rand(1,1,d),np.random.rand(d))
# part 3.
assert(np.allclose(conc_diag_evZ(conc_diag_evZ(z,z2),z3),conc_diag_evZ(z,conc_diag_evZ(z2,z3)),atol=1e-09))


