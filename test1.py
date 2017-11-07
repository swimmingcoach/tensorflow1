import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.7 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
Biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + Biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
init = tf.global_variables_initializer()  # 替换成这样就好

sess = tf.Session()
sess.run(init)          # Very important

for step in range(100):
    sess.run(train)
    _W = sess.run(Weights)
    _B = sess.run(Biases)
    print(step, _W, _B)