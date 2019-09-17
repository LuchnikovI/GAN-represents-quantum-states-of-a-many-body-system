import numpy as np
import mpnum as mp
import mpnum.povm as mpp
import functools
import scipy
from mpnum.utils.array_transforms import local_to_global
import os
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_probability as tfp

#Pauli matrices
x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
z = np.array([[1, 0], [0, -1]], dtype=np.complex64)

#matrix filled by zeros
zero = np.zeros((2, 2))

#id matrix
idm = np.eye(2)

#array of Pauli matrices
sigma = np.array([x, y, z])

#tetrahedral POVM vector
s = np.array([[0, 0, 1], [2 * np.sqrt(2)/3, 0, -1/3],\
              [-np.sqrt(2)/3, np.sqrt(2/3), -1/3],\
              [-np.sqrt(2)/3, -np.sqrt(2/3), -1/3]], dtype=np.complex64)

#tetrahedral POVM
M = 0.25 * (idm + np.einsum('ij,jkl->ikl', s, sigma))

#intersection matrix
T = np.einsum('ijk,lkj->il', M, M)

#inverse intersection matrix
T_inv = np.linalg.inv(T)

#random numbers generator
rng = np.random

'''the function returns mpo format of TFI(ferromagnetic) hamiltonian
sites - number of spins
ta - the magnitude of magnetic field aimed along the x-axis
la - the magnitude of magnetic field aimed along the z-axis'''

def ham(sites=3, ta=1., la=0.):
    
    #two-site zz part
    h_mpo = mp.MPArray.from_kron([z, -z])
    #an empty list will be filled by local terms
    h = []
    
    #adding zz terms to the hamiltonian
    for startpos in range(sites - 1):
        left = [mp.MPArray.from_kron([idm] * startpos)] if startpos > 0 else []
        right = [mp.MPArray.from_kron([idm] * (sites - 2 - startpos))] \
            if sites - 2 - startpos > 0 else []
        h_at_startpos = mp.chain(left + [h_mpo] + right)
        h.append(h_at_startpos)
    
    #adding x terms to the hamiltonian
    for startpos in range(sites):
        left = [mp.MPArray.from_kron([idm] * startpos)] if startpos > 0 else []
        right = [mp.MPArray.from_kron([idm] * (sites - 1 - startpos))] \
            if sites - 1 - startpos > 0 else []
        h_at_startpos = mp.chain(left + [mp.MPArray.from_array_global(-ta * x)] + right)
        h.append(h_at_startpos)
    
    #adding z terms to the hamiltonian
    for startpos in range(sites):
        left = [mp.MPArray.from_kron([idm] * startpos)] if startpos > 0 else []
        right = [mp.MPArray.from_kron([idm] * (sites - 1 - startpos))] \
            if sites - 1 - startpos > 0 else []
        h_at_startpos = mp.chain(left + [mp.MPArray.from_array_global(-la * z)] + right)
        h.append(h_at_startpos)
        
    H = h[0]
    for local_term in h[1:]:
        H = H + local_term
    out, _ = H.compression(method='svd', relerr=1e-6)
    return out

'''the function returns the value of two-point corr. function built from samples
samples - set of samples (measurement outcomes)
site1 - the position of first point
site2 - the position of second point
a - one-site observable at site #1
b - one-site observable at site #2'''

def corr_from_samples(samples, site1, site2, a, b):

    left_term = np.einsum('ij,jkl,lk->i', T_inv, M, a)
    right_term = np.einsum('ij,jkl,lk->i', T_inv, M, b)
    mid_term = np.einsum('ij,jkl,lk->i', T_inv, M, a.dot(b))
    ind_left = samples[:, site1 - 1]
    ind_right = samples[:, site2 - 1]
    if site1 == site2:
        return np.array([mid_term[ind_left[i]] for i in range(samples.shape[0])]).sum() / samples.shape[0]
    else:
        return (np.array([left_term[ind_left[i]] for i in range(samples.shape[0])]) *\
        np.array([right_term[ind_right[i]] for i in range(samples.shape[0])])).sum() / samples.shape[0]
      
'''the function returns Renyi entropy of partial dens. matrix with alpha=2 from samples

the function requires two sets of samples because stochastic estimation
of Reniy entropy includes double averaging

samples1 - first set of samples
samples2 - second set of samples
separating_point - separating point'''

def renyi_entropy_from_samples(samples1, samples2, separating_point):
    
    local_dens = np.einsum('ij,jkl->ikl', T_inv, M)
    local_entropy = np.einsum('kij,lji->kl', local_dens, local_dens)
    entropy = local_entropy[samples1[:, 0], samples2[:, 0]]
    for i in range(1, separating_point):
        entropy = entropy * local_entropy[samples1[:, i], samples2[:, i]]
    return -2 * np.log(entropy.mean())
      
########################################################   
#the class provides tools for manipulating of TFI model#
########################################################
class ising_chain():
    
    '''the method initializes TFI model
    sites - number of spins
    ta - the magnitude of magnetic field aimed along the x-axis
    la - the magnitude of magnetic field aimed along the z-axis'''
    
    def __init__(self, sites, ta=1, la=0):
        
        #mpo format of hamiltonian
        self.h = ham(sites, ta, la)
        #mps format of ground state (need to run gs method to fill it)
        self.psi = None
        #number of sites
        self.sites = sites
        #tetrahedral POVM
        self.povm = mpp.mppovm.MPPovm.from_local_povm(M, width=sites)
        
        self.sampler = None
        #ground state energy (need to run gs method to fill it)
        self.e = None
    
    '''the method fills ground state (inplace) in mps format to the class
    sites - number of spins
    ta - the magnitude of magnetic field aimed along the x-axis
    la - the magnitude of magnetic field aimed along the z-axis'''
    
    def gs(self, num_sweeps=5, rank=25):
        
        #local eig solver
        eigs = functools.partial(scipy.sparse.linalg.eigsh, k=1, tol=1e-6, which='SA')
        #filling ground state and its energy
        self.e, self.psi = mp.linalg.eig(self.h, num_sweeps=num_sweeps, startvec_rank=rank, eigs=eigs)
        
    '''the method returns value of two points corr. func. from mps,
    site1 - the position of first point,
    site2 - the position of second point,
    a - one-site observable at site #1,
    b - one-site observable at site #2,
    the method returns a value of two point corr. func.'''
    
    def corr(self, site1, site2, a, b):
        #making two points observable in the form of mpo
        left = [mp.MPArray.from_kron([idm] * (site1 - 1))] if (site1 - 1) > 0 else []
        right = [mp.MPArray.from_kron([idm] * (self.sites - site2))] \
            if (self.sites - site2) > 0 else []
        mid = [mp.MPArray.from_kron([idm] * (site2 - site1 - 1))] \
            if (site2 - site1 - 1) > 0 else []
        
        mpo = mp.chain(left + [mp.MPArray.from_array_global(a)] + mid + \
                 [mp.MPArray.from_array_global(b)] + right) if site1 != site2 else\
                  mp.chain(left + [mp.MPArray.from_array_global(a.dot(b))]\
                           + right)
        #returning matrix element
        return mp.dot(mp.dot(self.psi.conj(), mpo), self.psi).to_array()
      
    '''the method returns one site mean value (from mps),
    site - the position of the point,
    a - one-site observable at given site,
    method returns the value of observable after averaging'''
    
    def one_site_mean(self, site, a):
        #making one point observable in the form of mpo
        left = [mp.MPArray.from_kron([idm] * (site - 1))] if (site - 1) > 0 else []
        right = [mp.MPArray.from_kron([idm] * (self.sites - site))] \
            if (self.sites - site) > 0 else []
        
        mpo = mp.chain(left + [mp.MPArray.from_array_global(a)] + right)
        #returning matrix element
        return mp.dot(mp.dot(self.psi.conj(), mpo), self.psi).to_array()
    
    '''the method calculates Renyi entropy of partial dens. matrix with alpha=2 from mps
    separating_point - separating point,
    mathod returns number - value of Renyi entropy'''
    
    def renyi_entropy(self, separating_point):
        #separating set of local tensors into two parts
        left_part = self.psi._lt._ltens[:separating_point]
        right_part = self.psi._lt._ltens[separating_point:]
        #
        in_left = left_part[0]
        in_left = np.einsum('ijk,ljm->ilkm', in_left, in_left.conj())
        for i in range(1, len(left_part)):
            update_left = left_part[i]
            update_left = np.einsum('ijk,ljm->ilkm', update_left, update_left.conj())
            in_left = np.einsum('ijkl,klmn->ijmn', in_left, update_left)
        in_left = in_left.reshape(in_left.shape[2:])
        #
        in_right = right_part[-1]
        in_right = np.einsum('ijk,ljm->ilkm', in_right, in_right.conj())
        for i in range(len(right_part)-2, -1, -1):
            update_right = right_part[i]
            update_right = np.einsum('ijk,ljm->ilkm', update_right, update_right.conj())
            in_right = np.einsum('ijkl,klmn->ijmn', update_right, in_right)
        in_right = in_right.reshape(in_right.shape[:2])
        t = in_left.dot(in_right.T)
        
        return -2 * np.log(np.trace(t.dot(t)))
      
    '''the method generates fake measurements from mps,
    n - number of samples, method returns np.array of shape
    (n, m), m is the number of spins in the model, method
    accelerated by tensor flow'''
    
    def sample(self, n):
        
        #sampling from discrete distribution
        tf.reset_default_graph()
        probs = tf.cast(tf.placeholder(shape=(None, 4, 1), dtype=tf.complex128), dtype=tf.float32)
        gumbel_dist = tfp.distributions.Gumbel(loc=0., scale=1.)
        gumbel_eps = gumbel_dist.sample((n, 4, 1))
        smpl = tf.argmax(tf.log(probs) + gumbel_eps, axis=1)
        
        #update marginal distribution
        tensor_update = tf.placeholder(shape=(None, 4, None), dtype=tf.complex64)
        input_tensor = tf.placeholder(shape=(n, None), dtype=tf.complex64)
        out_tensor = tf.einsum('jk,klm->jlm', input_tensor, tensor_update)
        
        sess = tf.Session()
        
        #marginal mass functions
        inds = np.arange(n)
        pmf = mp.prune(self.povm.pmf(self.psi), singletons=True)
        marginal_pmf = [None] * (len(pmf) + 1)
        marginal_pmf[len(pmf)] = pmf
        for n_sites in reversed(range(len(pmf))):
            p = marginal_pmf[n_sites + 1].sum([()] * (n_sites) + [(0,)])
            if n_sites > 0:
                p = mp.prune(p)
            marginal_pmf[n_sites] = p
        del marginal_pmf[0]
        
        #sampling loop
        for i in range(len(marginal_pmf) - 1):
            sub_chain = marginal_pmf[i]._lt._ltens
            next_sub_chain = marginal_pmf[i + 1]._lt._ltens
            if i == 0:
                samples = sess.run(smpl, feed_dict={probs:sub_chain[0].reshape((1, 4, 1))})
                in_tensor = next_sub_chain[0][0, samples[:, 0], :]
            else:
                
                prob = sess.run(out_tensor, feed_dict={input_tensor:in_tensor, tensor_update:sub_chain[-1]})
                samples = np.append(samples, sess.run(smpl, \
                                  feed_dict={probs:prob.reshape((n, 4, 1))}), axis=-1)
                
                in_tensor = sess.run(out_tensor, feed_dict={input_tensor:in_tensor, tensor_update:next_sub_chain[-2]})
                in_tensor = in_tensor[inds, samples[:, -1]]
                
        sub_chain = marginal_pmf[-1]._lt._ltens
        
        prob = sess.run(out_tensor, feed_dict={input_tensor:in_tensor, tensor_update:sub_chain[-1]})
        samples = np.append(samples, sess.run(smpl, \
                                  feed_dict={probs:prob.reshape((n, 4, 1))}), axis=-1)
        
        tf.reset_default_graph()
        
        return samples
    
    '''the method returns value of povm induced mass function for a given index,
    indices - np. array of indices with shape (n, sites), where n is number of indices,
    sites is number of spins'''
    def prob(self, indices):
        
        # update of local tensor
        tensor_update = tf.placeholder(shape=(None, indices.shape[0], None), dtype=tf.complex64)
        input_tensor = tf.placeholder(shape=(None, indices.shape[0], None), dtype=tf.complex64)
        out_tensor = tf.einsum('ijk,kjm->ijm', input_tensor, tensor_update)
        sess = tf.Session()
        
        # mass function
        pmf = mp.prune(self.povm.pmf(self.psi), singletons=True)._lt._ltens
        
        # initial local tensor
        in_tensor = pmf[0][:, indices[:, 0], :]
        
        #loop over a chain
        for i in range(1, len(pmf)):
            
            update = pmf[i][:, indices[:, i], :]
            in_tensor = sess.run(out_tensor, feed_dict={input_tensor:in_tensor, tensor_update:update})
            
        return in_tensor.reshape((in_tensor.shape[1],))