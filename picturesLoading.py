from PIL import Image
import numpy as np
from twoParameterSS import warp_a0,warp_a1

#create warped tree (version 1)
tree_init = np.array(Image.open('pictures/tree_init.bmp'))
dachshund_init = np.array(Image.open('pictures/dachshund_init.bmp'))
letter_init = np.array(Image.open('pictures/letter_init.bmp'))
#print(tree_init[:,:,0])
#print("green",tree_init[21,12,:]/255)
#print("brown",tree_init[25,12,:]/255)
#print("orange",tree_init[28,12,:]/255)
S,T,d = tree_init.shape
tree_warped1 = tree_init
for k in [29,29,29,10,6,5,4,4,3,3]:
  tree_warped1 = warp_a0(k-1,tree_warped1)
for k in [22,21,18,13,8,5,4,1,1]:
  tree_warped1 = warp_a1(k-1,tree_warped1)

#create warped tree (version 2)
tree_warped2 = tree_init
for k in [26,26,26,26,26,25,25,25,25,25,25,25,25,25,25,25,25]:
  tree_warped2 = warp_a0(k-1,tree_warped2)
for k in [21,19,17,15,13,11,9,7,5,1,1,1,1,1,1]:
  tree_warped2 = warp_a1(k-1,tree_warped2)

dachshund_warped1 = dachshund_init
for k in [18,18,18,18,18,18]:
  dachshund_warped1 = warp_a1(k,dachshund_warped1)
for k in [0,0,0,0,0,0,0]:
  dachshund_warped1 = warp_a0(k,dachshund_warped1)

letter_warped1 = letter_init
for k in reversed(range(14)):
  letter_warped1 = warp_a1(k,letter_warped1)
for k in reversed(range(17)):
  letter_warped1 = warp_a0(k,letter_warped1)

letter_warped2 = letter_init
for k in reversed(range(14)):
  letter_warped2 = warp_a1(0,letter_warped2)
for k in reversed(range(17)):
  letter_warped2 = warp_a0(0,letter_warped2)

def insert_parting_line_a0(z,k):
  S,T,d = z.shape
  res = np.empty([S+1,T,d],dtype=z.dtype) 
  res[:k,:,:] = z[:k,:,:] 
  res[k,:,:] = 255
  res[(k+1):,:,:] = z[k:,:,:]
  return res

def insert_parting_line_a1(z,k):
  S,T,d = z.shape
  res = np.empty([S,T+1,d],dtype=z.dtype) 
  res[:,:k,:] = z[:,:k,:] 
  res[:,k,:] = 255
  res[:,(k+1):,:] = z[:,k:,:]
  return res

#tree_warped1_print = insert_parting_line_a0(tree_warped1,3)

# scale in the sense that every line is warped scale times 
def scale_and_insert_parting_lines(z,scale):
  S,T,d = z.shape
  ret = z 
  for s in reversed(range(S)):
    for i in range(scale):
      ret = warp_a0(s,ret)
    if s != S:
      ret = insert_parting_line_a0(ret,s)
  for t in reversed(range(T)):
    for i in range(scale):
      ret = warp_a1(t,ret)
    if t != T:
      ret = insert_parting_line_a1(ret,t)
  return ret
       
#tree_warped1_print = scale_and_insert_parting_lines(tree_warped1,50)
#tree_warped2_print = scale_and_insert_parting_lines(tree_warped2,50)
#tree_init_print = scale_and_insert_parting_lines(tree_init,50)

#dachshund_init_print = scale_and_insert_parting_lines(dachshund_init,50)
#ret_dachshund_init_print = Image.fromarray(dachshund_init_print)
#ret_dachshund_init_print.save('pictures/dachshund_init_print.bmp')

#dachshund_warped1_print = scale_and_insert_parting_lines(dachshund_warped1,50)
#ret_dachshund_warped1_print = Image.fromarray(dachshund_warped1_print)
#ret_dachshund_warped1_print.save('pictures/dachshund_warped1_print.bmp')

letter_init_print = scale_and_insert_parting_lines(letter_init,50)
ret_letter_init_print = Image.fromarray(letter_init_print)
ret_letter_init_print.save('pictures/letter_init_print.bmp')

letter_warped1_print = scale_and_insert_parting_lines(letter_warped1,50)
ret_letter_warped1_print = Image.fromarray(letter_warped1_print)
ret_letter_warped1_print.save('pictures/letter_warped1_print.bmp')

letter_warped2_print = scale_and_insert_parting_lines(letter_warped2,50)
ret_letter_warped2_print = Image.fromarray(letter_warped2_print)
ret_letter_warped2_print.save('pictures/letter_warped2_print.bmp')

#save ceates images 
#ret_tree_warped1 = Image.fromarray(tree_warped1)
#ret_tree_warped1.save('pictures/tree_warped1.bmp')
#ret_tree_warped1_print = Image.fromarray(tree_warped1_print)
#ret_tree_warped1_print.save('pictures/tree_warped1_print.bmp')
#ret_tree_warped1_print = Image.fromarray(tree_init_print)
#ret_tree_warped1_print.save('pictures/tree_init_print.bmp')
#ret_tree_warped2 = Image.fromarray(tree_warped2)
#ret_tree_warped2.save('pictures/tree_warped2.bmp')
#ret_tree_warped2_print = Image.fromarray(tree_warped2_print)
#ret_tree_warped2_print.save('pictures/tree_init_warped2.bmp')
