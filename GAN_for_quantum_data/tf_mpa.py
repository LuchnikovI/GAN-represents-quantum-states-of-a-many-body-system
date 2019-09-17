import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

#Pauli matrices
x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
z = np.array([[1, 0], [0, -1]], dtype=np.complex64)

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

#Here is the class describing mpa decomposition of multy-dimensional tensor, powered by tensorflow
class mpa():
    
    '''the method initializes of mpa object,          
    set_of_tensors - list of local tensors'''
    def __init__(self, set_of_tensors):
        
        #if set_of_tensors is none one creates a class with empty properties
        if set_of_tensors == None:
            
            self.shape_of_site = None
            self.bond_dims = None
            self.set_of_tensors = None
            
        else:
            
            #checking of site shapes
            shape_of_site = set_of_tensors[0].shape[1:-1]
            flag = 1
            for i in range(1, len(set_of_tensors)):
                if set_of_tensors[i].shape[1:-1] == shape_of_site:
                    pass
                else:
                    flag = 0
            assert flag == 1, 'Incorect site shapes'

            #shape of site in the body of the object
            self.shape_of_site = shape_of_site

            #checking bond dimension
            flag = 1
            for i in range(len(set_of_tensors) - 1):
                if set_of_tensors[i].shape[-1] == set_of_tensors[i+1].shape[0]:
                    pass
                else:
                    flag = 0
            assert flag == 1, 'Incorect bond dimension'

            #all bond dimensions in the body of the object
            self.bond_dims = []
            self.bond_dims.append(set_of_tensors[0].shape[0])
            for tensor in set_of_tensors:
                self.bond_dims.append(tensor.shape[-1])

            #tf local tensors in the body of the obeject
            self.set_of_tensors = [tf.constant(tensor, dtype=tf.complex64) for tensor in set_of_tensors]
    
    '''the method performs site reshape (works inplace),
    shape - new site shape '''     
    def reshape(self, shape):
        
        #reshaping of each site
        for i in range(len(self.set_of_tensors)):
            self.set_of_tensors[i] = tf.reshape(self.set_of_tensors[i],\
            (tf.shape(self.set_of_tensors[i])[0],) + shape + (tf.shape(self.set_of_tensors[i])[-1],))
            
        #filling the new shape into the body of the object
        self.shape_of_site = shape
        
    '''the method performs site transpose (works inplace),
    order - new order of site indecies'''
    def transpose(self, order):
        
        #corrected order tuple
        order_corrected = tuple([i + 1 for i in order])
        
        #rank of local tensor
        rank = len(self.shape_of_site) + 1
        
        #transpose of each site
        for i in range(len(self.set_of_tensors)):
            self.set_of_tensors[i] = tf.transpose(self.set_of_tensors[i],\
            perm=(0,) + order_corrected + (rank,))
        
        #filling the new shape into the body of the object
        self.shape_of_site  = tuple(np.array(self.shape_of_site)[list(order)])
    
    '''the method performs complex conj. (works inplace)'''
    def conj(self):
        #conj. of each local tensor
        for i in range(len(self.set_of_tensors)):
            self.set_of_tensors[i] = tf.conj(self.set_of_tensors[i])

    '''the method performs site einsum,
    string - einsum specification,
    mpa1 - first mp array,
    mpa2 - second mp array,

    returns resulting mp array'''
    @staticmethod
    def einsum(string, mpa1, mpa2):
        
        #preparing einsum string for local tensors
        inp, out = string.split('->')
        inp_a, inp_b = inp.split(',')
        corr_inp_a = 'w' + inp_a + 'x'
        corr_inp_b = 'y' + inp_b + 'z'
        corr_out = 'wy' + out + 'xz'
        corr_string = corr_inp_a + ',' + corr_inp_b + '->' + corr_out
        
        #are the lengths of mp arrays equal?
        assert len(mpa1.set_of_tensors) == len(mpa2.set_of_tensors), 'mp arrays have different lenght'
        
        #einsum of each site
        new_mpa = mpa(None)
        new_set_of_tensors = []
        new_bond_dims = []
        for i in range(len(mpa1.set_of_tensors)):
            new_set_of_tensors.append(tf.einsum(corr_string, mpa1.set_of_tensors[i], mpa2.set_of_tensors[i]))
            shape = tuple(map(int, new_set_of_tensors[-1].shape))
            new_set_of_tensors[-1] = tf.reshape(new_set_of_tensors[-1],\
            (-1,) + shape[2:-2] + (shape[-2] * shape[-1],))
            
            #correct site shape and bond dims in the body of the class
            if i == 0:
                new_mpa.shape_of_site = shape[2:-2]
                new_bond_dims.append(shape[0] * shape[1])
                new_bond_dims.append(shape[-2] * shape[-1])
            new_bond_dims.append(shape[-2] * shape[-1])
        new_mpa.set_of_tensors = new_set_of_tensors
        new_mpa.bond_dims = new_bond_dims
        return new_mpa
    
    '''the method returns local tensors in np format,
    sess - tf session'''
    def eval_local_tensors(self, sess):
        return sess.run(self.set_of_tensors)
    
    '''the method evaluates fully convolved network,
    sess - tf session,

    returns scalar - value of convolved network'''
    def eval_network(self, sess):
        
        #if network can be evaluated in scalar?
        assert np.array(self.set_of_tensors[0].shape[1:-1]).prod() == 1, 'network cannot be evaluated in scalar'
        
        mpa_copy = []
        for tensor in self.set_of_tensors:
            shape = tuple(map(int, tensor.shape))
            mpa_copy.append(tf.reshape(tensor, (shape[0],) + (shape[-1],)))
        in_tensor = mpa_copy[0]
        for i in range(1, len(mpa_copy)):
            in_tensor = tf.matmul(in_tensor, mpa_copy[i])
        return sess.run(in_tensor)
    
    
    '''the method makes a copy of the object, 

    rerurns copy of the object'''
    def copy(self):
        
        new_mpa = mpa(None)
        new_set_of_tensors = [tf.identity(tensor) for tensor in self.set_of_tensors]
        new_mpa.set_of_tensors = new_set_of_tensors
        new_mpa.bond_dims = self.bond_dims
        new_mpa.shape_of_site = self.shape_of_site
        
        return new_mpa
    
    '''the method returns mass functin induced 
    by the tetrahedral POVM in the form of mpa'''
    def prob(self):
        
        assert self.shape_of_site[0] == self.shape_of_site[1] and len(self.shape_of_site) == 2, 'mpa has incorrect shape'
        
        len_of_povm = len(self.set_of_tensors)
        mpa_povm = mpa([M.reshape((1,) + M.shape + (1,))] * len_of_povm)
        return mpa.einsum('ij,kji->k', self, mpa_povm)
    
    '''the method returns values of tensor for the given set of indices,
    sess - tensor flow session,
    indices - set of indices (np matrix)'''
    def value(self, sess, indices):
        
        local_tensors = self.eval_local_tensors(sess)
        local_tensors_per_index = [local_tensors[i][:, indices[:, i], :] for i in range(len(local_tensors))]
        tf_local_tensors_per_index = [tf.constant(tensor, dtype=tf.complex64) for tensor in local_tensors_per_index]
        in_tensor = tf_local_tensors_per_index[0]
        
        for i in range(1, len(tf_local_tensors_per_index)):
            in_tensor = tf.einsum('ijk,kjl->ijl', in_tensor, tf_local_tensors_per_index[i])
        
        return sess.run(in_tensor)
    
    '''the function returns set of measurement outcomes 
    (np array of shape (n, number_of_spins)),
    n - number of samples,
    sess - tf session'''
    def sample(self, n, sess):
        
        #the list will be filled by marginal distributions
        marginals = []
        
        #the auxiliary np array filled by indices from zero to (n-1)
        idx = np.arange(n)
        
        #the mass function and its np form
        mf = self.prob()
        np_mf = mf.eval_local_tensors(sess)
        
        #the tf graph performs sampling from discrete distribution
        probs = tf.placeholder(shape=(1, 4, None), dtype=tf.float32)
        gumbel_dist = tfp.distributions.Gumbel(loc=0., scale=1.)
        gumbel_eps = gumbel_dist.sample((1, 4, n))
        smpl = tf.argmax(tf.log(probs) + gumbel_eps, axis=1)
        
        #the tf graph calculates a distribution for given history of measurements
        s_in_tensor = tf.placeholder(shape=(None, n), dtype=tf.complex64)
        s_update_tensor = tf.placeholder(shape=(None, 4, None), dtype=tf.complex64)
        s_new_tensor = tf.einsum('ijk,kl->ijl', s_update_tensor, s_in_tensor)
        
        #the loop fills marginals by marginal distributions
        marginals.append(np_mf)
        for _ in range(1, len(np_mf)):
            new_marginal = marginals[-1][2:]
            marginals.append([np.einsum('ik,kjl->ijl', marginals[-1][0].sum(1), marginals[-1][1])] + new_marginal) 
        
        #changing of the order in the list of marginal distributions
        marginals = marginals[::-1]
        
        #the first sample in the chain
        samples = sess.run(smpl, feed_dict={probs:marginals[0][0]}).reshape((-1, 1))
        in_tensor = marginals[1][1][:, samples[:, 0], :][:, :, 0]
        
        #the loop generates the rest of the samples except the last
        for i in range(1, len(marginals)-1):
            
            new_prob = sess.run(s_new_tensor, feed_dict={s_update_tensor:marginals[i][0], s_in_tensor:in_tensor})
            sample = sess.run(smpl, feed_dict={probs:new_prob}).reshape((-1, 1))
            samples = np.append(sample, samples, axis=1)
            in_tensor = sess.run(s_new_tensor, feed_dict={s_update_tensor:marginals[i+1][1], s_in_tensor:in_tensor})
            in_tensor = in_tensor[:, sample[:, 0], idx]
        
        #the last sample
        new_prob = sess.run(s_new_tensor, feed_dict={s_update_tensor:marginals[-1][0], s_in_tensor:in_tensor})
        sample = sess.run(smpl, feed_dict={probs:new_prob}).reshape((-1, 1))
        samples = np.append(sample, samples, axis=1)
            
        return samples