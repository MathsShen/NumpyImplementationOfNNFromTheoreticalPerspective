import numpy

batch_size, n_x, n_h, n_y = 64, 1000, 100, 10

x = np.random.randn(batch_size, n_x)
t = np.random.randn(batch_size, n_y)

W = np.random.randn(n_x, n_h)
S = np.random.randn(n_h, n_y)

learning_rate = 1e-6
for i in range(500):
	# Forward pass: compute predicted y
	u = x.dot(W)
	h = np.maximum(u, 0)
	y = h.dot(S)

	# Compute and print loss
	loss = np.square(y-t).sum()
	print("Iter: {}\tLoss: {}".format(i, loss))

	# Backprop to compute gradients of W and S with respect to loss
	grad_y = 2.0 * (y-t)
	grad_S = h.T.dot(grad_y)

	grad_h = grad_y.dot(S.T)
	grad_u = grad_h.copy()
	grad_u[u<0] = 0
	grad_W = x.T.dot(grad_u)

	# Update weights by gradient descent
	W -= learning_rate * grad_W
	S -= learning_rate * grad_S