import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_probability as tfp

'''The function returns keras model of discriminator. As an input it takes
num_of_in_n - number of input nurons, num_of_h_n - number of hidden nurons and
name.'''
def Discriminator(num_of_in_n=32, num_of_h_n=256, name='D'):
    with tf.name_scope(name):
        inp = K.Input(shape=(num_of_in_n, 4), dtype=tf.float32)
        h0 = K.layers.Reshape((num_of_in_n * 4,))(inp)
        h1 = K.layers.Dense(num_of_h_n, activation=None,\
                            kernel_initializer='he_normal')(h0)
        h1 = K.layers.ELU()(h1)
        out = K.layers.Dense(1, activation=None, kernel_initializer='he_normal')(h1)
        return K.Model(inputs=inp, outputs=out)

'''The function returns keras model of generator. As an input it takes
num_of_in_n - number of input nurons, num_of_h_n - number of hidden nurons,
num_of_out_n - number of output nerons and name.'''
def Generator(num_of_in_n=32, num_of_h_n=256, num_of_out_n=32, name='G'):
    with tf.name_scope(name):
        inp = K.Input(shape=(num_of_in_n,), dtype=tf.float32)
        h1 = K.layers.Dense(num_of_h_n, activation=None,\
                            kernel_initializer='he_normal')(inp)
        h1 = K.layers.ELU()(h1)
        h2 = K.layers.Dense(num_of_out_n * 4, activation=None,\
                            kernel_initializer='he_normal')(h1)
        out = K.layers.Reshape((num_of_out_n, 4))(h2)

        return K.Model(inputs=inp, outputs=out)

'''The function returns tf node of gumbel samples. As an input it takes shape
and name.'''
def Gumbel_samples(shape, name='gumbel_samples'):
    with tf.name_scope(name):
        gumbel_dist = tfp.distributions.Gumbel(loc=0., scale=1.)
        return gumbel_dist.sample(shape)

'''The function returns tf node of normal samples. As an input it takes shape
and name.'''
def Normal_samples(shape, name='normal_dist'):
    with tf.name_scope(name):
        normal_dist = tfp.distributions.normal.Normal(loc=0., scale=1.)
        return normal_dist.sample(shape)

'''The function returns tf tensor of shape (batch_size, number_of_spins, 4) - 
smoothed version of a one-hot representation of samples. As an input it takes 
x - samples of shape (batch_size, number_of_spins), normal_noise - normal noise
of shape (batch_size, number_of_spins, 4), eta - smoothing coefficient and name.'''      
def Smoothing(x, normal_noise, eta, name='smoothing'):
    with tf.name_scope(name):
        softmax_noise = tf.nn.softmax(normal_noise, axis=-1)
        one_hot_x = tf.cast(tf.one_hot(x, depth=4, axis=-1), dtype=tf.float32)
        return softmax_noise * eta + one_hot_x * (1 - eta)

'''The function returns soft samples (gubel softmax trick) from logits. As an
input it takes G_output - output of gan network (logits), gumbel_samples - 
gumbel samples, T - temperature of softmax and name.'''
def Soft_samples(G_output, gumbel_samples, T, name='soft_samples_from_gen'):
    with tf.name_scope(name):
        logits = tf.nn.log_softmax(G_output, axis=-1)
        return tf.nn.softmax((logits + gumbel_samples) / T, axis=-1)

'''The function returns samples from logits. As an input it takes G_output - 
output of gan network (logits), gumbel_samples - 
gumbel samples and name.'''
def Hard_samples(G_output, gumbel_samples, name='hard_samples_from_gen'):
    with tf.name_scope(name):
        logits = tf.nn.log_softmax(G_output, axis=-1)
        return tf.argmax(logits + gumbel_samples, axis=-1)

'''The function returns values of D and G loss functions. As an input it takes
d_true_samples - D(original samles), d_fake_samples - D(fake samples) and name.'''
def Loss_function(d_true_samples, d_fake_samples, name='loss'):
    with tf.name_scope(name):
        loss_d = -tf.reduce_mean(tf.log_sigmoid(d_true_samples) + tf.log_sigmoid(-d_fake_samples))
        loss_g = -tf.reduce_mean(tf.log_sigmoid(d_fake_samples))
    return loss_g, loss_d

'''The function splits data on batches. As an input it takes data and batch_size.'''
def batcher(data, batch_size):
    np.random.shuffle(data)
    return data.reshape((-1, batch_size) + (data.shape[-1],))

'''The class gives tools for operating with GAN.'''
class gan():
    
    '''The method initializes GAN. As an input it takes dimension of input(output)
    and dimension of hidden layer.'''
    def __init__(self, dim=32, dim_hidden=256):
        
        # discriminator and generator networks
        self.D = Discriminator(dim, dim_hidden)
        self.G = Generator(dim, dim_hidden, dim)
        
        # softmax temperature placeholder
        self.T = tf.placeholder(shape=(), dtype=tf.float32, name='temp')
        
        # smoothing coefficient placeholder
        self.eta = tf.placeholder(shape=(), dtype=tf.float32, name='eta')
        
        # data placeholder
        self.x_in = tf.placeholder(shape=(None, dim), dtype=tf.int32, name='true_samples')
        
        # discriminator's learning rate placeholder
        self.lr_d = tf.placeholder(shape=(), dtype=tf.float32, name='lr_d')
        
        # generator's learning rate placeholder
        self.lr_g = tf.placeholder(shape=(), dtype=tf.float32, name='lr_g')
        
        # size of batch
        self.b_size = tf.placeholder(shape=(), dtype=tf.int32, name='batch_size')
        
        # normal noise for smoothing of input data
        noise_for_x = Normal_samples(shape=(self.b_size, dim, 4), name='noise_for_x')
        
        # normal noise will be passed as an input to the generator
        g_input = Normal_samples(shape=(self.b_size, dim), name='g_input')
        
        # gumbel noise will be used for sampling
        gumbel_noise = Gumbel_samples(shape=(self.b_size, dim, 4))
        
        # smoothed version of data
        smooth_x = Smoothing(self.x_in, noise_for_x, self.eta)
        
        # soft samples from the GAN
        fake_x = Soft_samples(self.G(g_input), gumbel_noise, self.T)
        
        # scores of samples (fake and true) assigned by the discriminator
        d_fake_x = self.D(fake_x)
        d_smooth_x = self.D(smooth_x)
        
        # loss functions (for generator and discriminator)
        self.loss_g, self.loss_d = Loss_function(d_smooth_x, d_fake_x)
        
        # optimization part
        with tf.name_scope('optimizators'):
    
            self.train_d = tf.train.AdamOptimizer(self.lr_d).minimize(self.loss_d,\
                                                        var_list=self.D.weights)
            self.train_g = tf.train.AdamOptimizer(self.lr_g).minimize(self.loss_g,\
                                                        var_list=self.G.weights)
        
        # hard samples will be used at the stage of inference
        self.hard_samples = Hard_samples(self.G(g_input), gumbel_noise)
    
    '''The method return samples from gan. As an input it takes number of samples,
    and tf session.'''
    def sample(self, n, sess):
        
        return sess.run(self.hard_samples, feed_dict={self.b_size:n})
    
    '''the method provides one epoch of GAN training. As an input it takes
    sess - tf session, lr_g - learning rate of generator,
    lr_d - learning rate of discriminator,
    T - temperature of softmax, eta - smoothing coefficient, b_size - size of
    batchm data - data. It returns values of g and d loss.'''
    def train_epoch(self, sess, lr_g, lr_d, T, eta, b_size, data):
        
        # batched data
        batched_data = batcher(data, b_size)
        
        # values of d loss and g loss, will be filled during an epoch
        epoch_d_loss = 0.
        epoch_g_loss = 0.
        
        # number of iterations
        iter_num = batched_data.shape[0]
        
        # epoch loop
        for i in range(iter_num):
            
            # training step of discriminator
            l_d, _ = sess.run([self.loss_d, self.train_d],\
            feed_dict={self.x_in:batched_data[i], self.b_size:b_size,\
            self.lr_d:lr_d, self.lr_g:lr_g, self.T:T, self.eta:eta})
            
            epoch_d_loss = epoch_d_loss + l_d

            # training step of generator
            l_g, _ = sess.run([self.loss_g, self.train_g],\
            feed_dict={self.x_in:batched_data[i], self.b_size:b_size,\
            self.lr_d:lr_d, self.lr_g:lr_g, self.T:T, self.eta:eta})
            
            epoch_g_loss = epoch_g_loss + l_g
        
        return epoch_g_loss / iter_num, epoch_d_loss / iter_num
    
    ''' the method returns weights of generator and discriminator.'''
    def get_weights(self):
        
        return self.G.get_weights(), self.D.get_weights()
    
    ''' the method sets weights of generator and discriminator.'''
    def set_weights(self, g_weights, d_weights):
        
        self.D.set_weights(d_weights)
        self.G.set_weights(g_weights)