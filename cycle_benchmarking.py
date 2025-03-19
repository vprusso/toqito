import numpy as np
import matplotlib.pyplot as ply
import random
from numpy import linalg as LA
from scipy.optimize import curve_fit

X = np.matrix('0 1; 1 0')
Z = np.matrix('1 0; 0 -1')
Y = np.matrix('0 -1j; 1j 0') 
I = np.matrix('1 0; 0 1')
cnot=np.matrix('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0')
RZ=[[np.exp(1j*22.5),0],[0, np.exp(-1j*22.5)]]
RY=[[np.cos(45),-np.sin(45)], [np.sin(45), np.cos(45)]]
H=[[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]]
a = np.matrix('1;0')
b=np.matrix('0;1')
c=np.matrix('1;0')
d = 1/np.sqrt(2)*np.matrix('1;1')
Z1=np.matrix('1 0; 0 1j')
I1=np.kron(I,I)
toffoli=np.matrix('1 0 0 0 0 0 0 0; 0 1 0 0 0 0 0 0; 0 0 1 0 0 0 0 0; 0 0 0 1 0 0 0 0; 0 0 0 0 1 0 0 0; 0 0 0 0 0 1 0 0; 0 0 0 0 0 0 0 1; 0 0 0 0 0 0 1 0')


P2=[I,X,Y,Z]
P3=[np.kron(I,I),np.kron(I,X),np.kron(I,Y),np.kron(I,Z),np.kron(X,I),np.kron(X,X),np.kron(X,Y),np.kron(X,Z),np.kron(Y,I),np.kron(Y,X),np.kron(Y,Y),np.kron(Y,Z),np.kron(Z,I),np.kron(Z,X),np.kron(Z,Y),np.kron(Z,Z)]
P4=[np.kron(I,np.kron(I,I)),np.kron(I,np.kron(I,X)),np.kron(I,np.kron(I,Y)),np.kron(I,np.kron(I,Z)),np.kron(I,np.kron(X,I)),np.kron(I,np.kron(X,X)),np.kron(I,np.kron(X,Y)),np.kron(I,np.kron(X,Z)),np.kron(I,np.kron(Y,I)),np.kron(I,np.kron(Y,X)),np.kron(I,np.kron(Y,Y)),np.kron(I,np.kron(Y,Z)),np.kron(I,np.kron(Z,I)),np.kron(I,np.kron(Z,X)),np.kron(I,np.kron(Z,Y)),np.kron(I,np.kron(Z,Z)),np.kron(X,np.kron(I,I)),np.kron(X,np.kron(I,X)),np.kron(X,np.kron(I,Y)),np.kron(X,np.kron(I,Z)),np.kron(X,np.kron(X,I)),np.kron(X,np.kron(X,X)),np.kron(X,np.kron(X,Y)),np.kron(X,np.kron(X,Z)),np.kron(X,np.kron(Y,I)),np.kron(X,np.kron(Y,X)),np.kron(X,np.kron(Y,Y)),np.kron(X,np.kron(Y,Z)),np.kron(X,np.kron(Z,I)),np.kron(X,np.kron(Z,X)),np.kron(X,np.kron(Z,Y)),np.kron(X,np.kron(Z,Z)),np.kron(Y,np.kron(I,I)),np.kron(Y,np.kron(I,X)),np.kron(Y,np.kron(I,Y)),np.kron(Y,np.kron(I,Z)),np.kron(Y,np.kron(X,I)),np.kron(Y,np.kron(X,X)),np.kron(Y,np.kron(X,Y)),np.kron(Y,np.kron(X,Z)),np.kron(Y,np.kron(Y,I)),np.kron(Y,np.kron(Y,X)),np.kron(Y,np.kron(Y,Y)),np.kron(Y,np.kron(Y,Z)),np.kron(Y,np.kron(Z,I)),np.kron(Y,np.kron(Z,X)),np.kron(Y,np.kron(Z,Y)),np.kron(Y,np.kron(Z,Z)),np.kron(Z,np.kron(I,I)),np.kron(Z,np.kron(I,X)),np.kron(Z,np.kron(I,Y)),np.kron(Z,np.kron(I,Z)),np.kron(Z,np.kron(X,I)),np.kron(Z,np.kron(X,X)),np.kron(Z,np.kron(X,Y)),np.kron(Z,np.kron(X,Z)),np.kron(Z,np.kron(Y,I)),np.kron(Z,np.kron(Y,X)),np.kron(Z,np.kron(Y,Y)),np.kron(Z,np.kron(Y,Z)),np.kron(Z,np.kron(Z,I)),np.kron(Z,np.kron(Z,X)),np.kron(Z,np.kron(Z,Y)),np.kron(Z,np.kron(Z,Z))]


def g(l,p):
    return random.choices(l,weights=((1-p)*1000,1000*p/3,1000*p/3,1000*p/3),k=1)



def g1(l,p):
    return random.choices(l,weights=((1-p)*1000,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15),k=1)

def g2(l,p):
    return random.choices(l,weights=((1-p)*1000,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63),k=1)


def sequence(n,p):##Function used to produce the sequence of Single Pauli gates and its corresponding noise case
    k=[]
    k1=[]
    for i in range(n):
        k.append(random.choice(P2))#ideal case
    for i in range(n):
        if (k[i]==I).all() == True:
            k1.append(np.matmul(I,g(P2,p)))
        elif (k[i]==X).all() == True:
            k1.append(np.matmul(X,g(P2,p)))
        elif (k[i]==Y).all() == True:
            k1.append(np.matmul(Y,g(P2,p)))
        else:
            k1.append(np.matmul(Z,g(P2,p)))
    return [k,k1]


def fc(L,m,N,p1,p2):##for CNOT, L is the number of sequences, m is the number of cycles, p1 is the error value set for single Pauli gates inside the m cycle, p2 is an error set for CNOT gate, N is number of qubits
    k4=[random.choice(P2),random.choice(P2)]##Firstly produce Random Input operator
    for i in range(1,len(k4)):
        t=np.kron(k4[0],k4[i])
    eigenvalue, eigenstate = np.linalg.eig(t)
    eigenstate=eigenstate.transpose()
    c = eigenstate[0]
    d1=np.matmul(c.conj().T,c)##Get the random input operator's density operator corresponding to +1 eigenvalue
    for h in range(L):##start of the sequence L
        [k,k1]=sequence(N,0)##The first random pauli cycle is set to be ideal
        for h1 in range(1,len(k)):
            gate1=np.kron(k[0],k[h1])##Random Pauli cycle noiseless case
        for h2 in range(1,len(k1)):
            gate2=np.kron(k1[0],k1[h2])##Random Pauli cycle noise case
        initial=np.matmul(gate1,np.matmul(d1,gate1.conj().T))
        initial1=np.matmul(gate1,(np.matmul(d1,gate1.conj().T)))##This is the application of first random pauli cycle
        for i in range(m):##start the m cycles
            initial=np.matmul(cnot,(np.matmul(initial,cnot.conj().T)))##Application of ideal CNOT gate 
            b=np.matmul(cnot,g1(P3,p2))
            initial1=np.matmul(b,np.matmul(initial1,b.conj().T))##Application of noise CNOT gate 
            [k2,k3]=sequence(N,p1)##Get the random Pauli sequences
            for h3 in range(1,len(k2)):
                gate3=np.kron(k2[0],k2[h3])
            for h4 in range(1,len(k3)):
                gate4=np.kron(k3[0],k3[h4]) 
            initial=np.matmul(gate3,np.matmul(initial,gate3.conj().T))##Application of ideal random Pauli cycle
            initial1=np.matmul(gate4,np.matmul(initial1,gate4.conj().T))##Application of noise random Pauli cycle
        return np.trace(np.matmul(initial,initial1))


def expect(L,m,N,p1,p2):
    count=0
    for i in range(1,L+1):
        count+=fc(i,m,N,p1,p2)
    return count/L




