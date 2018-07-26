import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import pandas as pd 

mnist = input_data.read_data_sets('digits/',one_hot = True)



hm_epochs = 10
rnn_size = 128
n_chunks = 28
chunk_size = 28
batch_size = 128
n_classes = 10

x = tf.placeholder('float',[None,n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_nn(x):
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
	         'biases':tf.Variable(tf.random_normal([n_classes]))}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(x, n_chunks, 0)

	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

	output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

	return output

def train_network(x):
	prediction = recurrent_nn(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c

			print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

                
train_network(x)

"""got an accuracy of 98.43 pecent on the test set  
FUTURE WORK:
1.try improving the architrcture
2.play with the hyper parameters
3.submit on kaggle"""      






