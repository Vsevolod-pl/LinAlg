#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from LinAlgFuncs import *


print("задание 1")

# In[ ]:


a = tensor_from_iterable([
    [ -1,   0,   0,  -1, -13],
    [  6,   5,   2,   3,  -1],
    [ -3,  -2,  -1,  -2,  -3],
    [-19, -16,  -6,  -9,  -3],
    [  4,   4,   1,   1,   2]])


# In[3]:


print(rank(a)) # значит нужно представить в виде суммы 3х матриц ранга 1


# In[4]:


b1 = zeros(*a.shape())
b2 = zeros(*a.shape())
b3 = zeros(*a.shape())


# In[13]:


#Эти коэффициенты я подобрал вручную
b1[0,0] = -1
b1[0,3] = -1
b1[0,4] = -13

b1[3] = b1[0,:]
b1[4] = -1*b1[0,:]

print("rank b1:",rank(b1))


# In[14]:


b2[1,0] = 6
b2[1,1] = 5
b2[1,2] = 2
b2[1,3] = 3
b2[1,4] = -1

b2[3] = -4*b2[1,:]
b2[4] = 2*b2[1,:]

print("rank b2:",rank(b2))


# In[15]:


b3[2,0] = -3
b3[2,1] = -2
b3[2,2] = -1
b3[2,3] = -2
b3[2,4] = -3

b3[3] = -2*b3[2,:]
b3[4] = 3*b3[2,:]

print("rank b3:",rank(b3))


# In[17]:


print("b1:",b1,"b2:",b2,"b3:",b3,sep="\n")
print()
print("b1+b2+b3 - a:")
print(b1+b2+b3-a)


print("/nзадание 2")

# In[18]:


u1 = tensor_from_iterable((-48, -3, -1, 8))
u2 = tensor_from_iterable((40, 2, 0, -6))
u3 = tensor_from_iterable((-24, -3, -3, 6))
u4 = tensor_from_iterable((5, 1, 2, -6))
U = Tensor((u1,u2,u3,u4)).transpose2() #записать вектора по столбцам

v1 = tensor_from_iterable((-20, -4, -5, 7))
v2 = tensor_from_iterable((-7, 7, -5, 2))


# In[19]:


print(rref(U)) # => базис - u1, u2, u4, т.к. они стояли на местах главных позиций


# In[20]:


A = Tensor((u1,u2,u3,u4,v1,v2)).transpose2() #записал вектора по столбцам


# In[21]:


print(rref(A)) # => u1, u2, u4 - базис, v1 лежит в U, v2 - нет, так как v2 - главная позиция


# In[22]:


B = tensor_from_iterable((v1, [ 0, 1, 0, 0], [0, 0, 1, 0])) # дополним единичными векторами


# In[23]:


print(B)


print("/nзадание 3")

# In[24]:


u1 = tensor_from_iterable((-2, 3, 2, -10))
u2 = tensor_from_iterable((5, -2, 6, 3))
u3 = tensor_from_iterable((1, -2, -2, 7))
u4 = tensor_from_iterable((2, 0, 4, -2))
B = Tensor((u1,u2,u3,u4)) # записал ветора в матрицу по строкам


# In[25]:


A = Tensor(solve_hsle(B)) # записал решение в другую матрицу по строкам


# In[26]:


print(A)


print("/nзадание 4")

# In[27]:


a1 = (-16, 5, 3, -4)
a2 = (-19, 10, -3, -11)
a3 = (2, 1, -3, -2)
a4 = (-5, 4, -3, -5)


# In[28]:


b1 = (-4, 4, 1, -2)
b2 = (4, 1, 5, 4)
b3 = (20, -10, 7, 14)
b4 = (-12, 7, -3, -8)


# In[34]:


A = tensor_from_iterable([a1,a2,a3,a4,b1,b2,b3,b4]).transpose2()
print(rref(A)) #a1, a2, b1, базис U, т.к. они стояли на местах главных позиций
print("базис L1 + L2:", a1, a2, b1)
print("размерность:", rank(A))


# In[30]:


B1 = tensor_from_iterable([a1,a2,a3,a4]) # записал по строкам
D = Tensor(solve_hsle(B1)) # L1 = ФСР Dx=0
print("L1 = ФСР Dx=0, D=", D, sep="\n")


# In[31]:


B2 = tensor_from_iterable([b1,b2,b3,b4]) # записал по строкам
F = Tensor(solve_hsle(B2)) # L2 = ФСР Fx=0
print("L2 = ФСР Fx=0, F=", F, sep="\n")


# In[32]:


#L1 ∩ L2 = ФСР D/F x = 0
G = tensor_from_iterable((*D,*F))
print("L1 ∩ L2 = ФСР D/F x = 0", G, sep="\n")


# In[33]:


x = Tensor(solve_hsle(G))
print("базис L1 ∩ L2:", x)
print("размерность:", rank(x))


# In[ ]:




