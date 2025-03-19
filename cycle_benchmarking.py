{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as ply\n",
    "import random\n",
    "from numpy import linalg as LA\n",
    "from scipy.optimize import curve_fit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = np.matrix('0 1; 1 0')\n",
    "Z = np.matrix('1 0; 0 -1')\n",
    "Y = np.matrix('0 -1j; 1j 0') \n",
    "I = np.matrix('1 0; 0 1')\n",
    "cnot=np.matrix('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0')\n",
    "RZ=[[np.exp(1j*22.5),0],[0, np.exp(-1j*22.5)]]\n",
    "RY=[[np.cos(45),-np.sin(45)], [np.sin(45), np.cos(45)]]\n",
    "H=[[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]]\n",
    "a = np.matrix('1;0')\n",
    "b=np.matrix('0;1')\n",
    "c=np.matrix('1;0')\n",
    "d = 1/np.sqrt(2)*np.matrix('1;1')\n",
    "Z1=np.matrix('1 0; 0 1j')\n",
    "I1=np.kron(I,I)\n",
    "toffoli=np.matrix('1 0 0 0 0 0 0 0; 0 1 0 0 0 0 0 0; 0 0 1 0 0 0 0 0; 0 0 0 1 0 0 0 0; 0 0 0 0 1 0 0 0; 0 0 0 0 0 1 0 0; 0 0 0 0 0 0 0 1; 0 0 0 0 0 0 1 0')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "P2=[I,X,Y,Z]\n",
    "P3=[np.kron(I,I),np.kron(I,X),np.kron(I,Y),np.kron(I,Z),np.kron(X,I),np.kron(X,X),np.kron(X,Y),np.kron(X,Z),np.kron(Y,I),np.kron(Y,X),np.kron(Y,Y),np.kron(Y,Z),np.kron(Z,I),np.kron(Z,X),np.kron(Z,Y),np.kron(Z,Z)]\n",
    "P4=[np.kron(I,np.kron(I,I)),np.kron(I,np.kron(I,X)),np.kron(I,np.kron(I,Y)),np.kron(I,np.kron(I,Z)),np.kron(I,np.kron(X,I)),np.kron(I,np.kron(X,X)),np.kron(I,np.kron(X,Y)),np.kron(I,np.kron(X,Z)),np.kron(I,np.kron(Y,I)),np.kron(I,np.kron(Y,X)),np.kron(I,np.kron(Y,Y)),np.kron(I,np.kron(Y,Z)),np.kron(I,np.kron(Z,I)),np.kron(I,np.kron(Z,X)),np.kron(I,np.kron(Z,Y)),np.kron(I,np.kron(Z,Z)),np.kron(X,np.kron(I,I)),np.kron(X,np.kron(I,X)),np.kron(X,np.kron(I,Y)),np.kron(X,np.kron(I,Z)),np.kron(X,np.kron(X,I)),np.kron(X,np.kron(X,X)),np.kron(X,np.kron(X,Y)),np.kron(X,np.kron(X,Z)),np.kron(X,np.kron(Y,I)),np.kron(X,np.kron(Y,X)),np.kron(X,np.kron(Y,Y)),np.kron(X,np.kron(Y,Z)),np.kron(X,np.kron(Z,I)),np.kron(X,np.kron(Z,X)),np.kron(X,np.kron(Z,Y)),np.kron(X,np.kron(Z,Z)),np.kron(Y,np.kron(I,I)),np.kron(Y,np.kron(I,X)),np.kron(Y,np.kron(I,Y)),np.kron(Y,np.kron(I,Z)),np.kron(Y,np.kron(X,I)),np.kron(Y,np.kron(X,X)),np.kron(Y,np.kron(X,Y)),np.kron(Y,np.kron(X,Z)),np.kron(Y,np.kron(Y,I)),np.kron(Y,np.kron(Y,X)),np.kron(Y,np.kron(Y,Y)),np.kron(Y,np.kron(Y,Z)),np.kron(Y,np.kron(Z,I)),np.kron(Y,np.kron(Z,X)),np.kron(Y,np.kron(Z,Y)),np.kron(Y,np.kron(Z,Z)),np.kron(Z,np.kron(I,I)),np.kron(Z,np.kron(I,X)),np.kron(Z,np.kron(I,Y)),np.kron(Z,np.kron(I,Z)),np.kron(Z,np.kron(X,I)),np.kron(Z,np.kron(X,X)),np.kron(Z,np.kron(X,Y)),np.kron(Z,np.kron(X,Z)),np.kron(Z,np.kron(Y,I)),np.kron(Z,np.kron(Y,X)),np.kron(Z,np.kron(Y,Y)),np.kron(Z,np.kron(Y,Z)),np.kron(Z,np.kron(Z,I)),np.kron(Z,np.kron(Z,X)),np.kron(Z,np.kron(Z,Y)),np.kron(Z,np.kron(Z,Z))]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def g(l,p):\n",
    "    return random.choices(l,weights=((1-p)*1000,1000*p/3,1000*p/3,1000*p/3),k=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def g1(l,p):\n",
    "    return random.choices(l,weights=((1-p)*1000,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15,1000*p/15),k=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def g2(l,p):\n",
    "    return random.choices(l,weights=((1-p)*1000,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63,1000*p/63),k=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sequence(n,p):##Function used to produce the sequence of Single Pauli gates and its corresponding noise case\n",
    "    k=[]\n",
    "    k1=[]\n",
    "    for i in range(n):\n",
    "        k.append(random.choice(P2))#ideal case\n",
    "    for i in range(n):\n",
    "        if (k[i]==I).all() == True:\n",
    "            k1.append(np.matmul(I,g(P2,p)))\n",
    "        elif (k[i]==X).all() == True:\n",
    "            k1.append(np.matmul(X,g(P2,p)))\n",
    "        elif (k[i]==Y).all() == True:\n",
    "            k1.append(np.matmul(Y,g(P2,p)))\n",
    "        else:\n",
    "            k1.append(np.matmul(Z,g(P2,p)))\n",
    "    return [k,k1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fc(L,m,N,p1,p2):##for CNOT, L is the number of sequences, m is the number of cycles, p1 is the error value set for single Pauli gates inside the m cycle, p2 is an error set for CNOT gate, N is number of qubits\n",
    "    k4=[random.choice(P2),random.choice(P2)]##Firstly produce Random Input operator\n",
    "    for i in range(1,len(k4)):\n",
    "        t=np.kron(k4[0],k4[i])\n",
    "    eigenvalue, eigenstate = np.linalg.eig(t)\n",
    "    eigenstate=eigenstate.transpose()\n",
    "    c = eigenstate[0]\n",
    "    d1=np.matmul(c.conj().T,c)##Get the random input operator's density operator corresponding to +1 eigenvalue\n",
    "    for h in range(L):##start of the sequence L\n",
    "        [k,k1]=sequence(N,0)##The first random pauli cycle is set to be ideal\n",
    "        for h1 in range(1,len(k)):\n",
    "            gate1=np.kron(k[0],k[h1])##Random Pauli cycle noiseless case\n",
    "        for h2 in range(1,len(k1)):\n",
    "            gate2=np.kron(k1[0],k1[h2])##Random Pauli cycle noise case\n",
    "        initial=np.matmul(gate1,np.matmul(d1,gate1.conj().T))\n",
    "        initial1=np.matmul(gate1,(np.matmul(d1,gate1.conj().T)))##This is the application of first random pauli cycle\n",
    "        for i in range(m):##start the m cycles\n",
    "            initial=np.matmul(cnot,(np.matmul(initial,cnot.conj().T)))##Application of ideal CNOT gate \n",
    "            b=np.matmul(cnot,g1(P3,p2))\n",
    "            initial1=np.matmul(b,np.matmul(initial1,b.conj().T))##Application of noise CNOT gate \n",
    "            [k2,k3]=sequence(N,p1)##Get the random Pauli sequences\n",
    "            for h3 in range(1,len(k2)):\n",
    "                gate3=np.kron(k2[0],k2[h3])\n",
    "            for h4 in range(1,len(k3)):\n",
    "                gate4=np.kron(k3[0],k3[h4]) \n",
    "            initial=np.matmul(gate3,np.matmul(initial,gate3.conj().T))##Application of ideal random Pauli cycle\n",
    "            initial1=np.matmul(gate4,np.matmul(initial1,gate4.conj().T))##Application of noise random Pauli cycle\n",
    "        return np.trace(np.matmul(initial,initial1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.35866666666666663+0j)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def expect():\n",
    "    count=0\n",
    "    for i in range(1,6001):\n",
    "        count+=fc(i,2,2,0.3,0.2)\n",
    "    return count/6000\n",
    "expect()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
