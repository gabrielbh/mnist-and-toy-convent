from scipy import signal
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
# from keras.datasets import mnist

# define the functions we would like to predict:
from scipy.misc import logsumexp


num_of_functions = 3
size = 4
W = 4 * (np.random.random((size, size)) - 0.5)
y = {
    0: lambda x: np.sum(np.dot(x, W), axis=1),
    1: lambda x: np.max(x, axis=1),
    2: lambda x: np.log(np.sum(np.exp(np.dot(x, W)), axis=1))
}


def learn_linear(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a linear model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (w, training_loss, test_loss):
            w: the weights of the linear model
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    w = {func_id: np.zeros(size) for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):
        for _ in range(iterations):

            # draw a random batch:
            idx_train = np.random.choice(len(Y[func_id]['train']), batch_size)
            x_train, y_train = X['train'][idx_train,:], Y[func_id]['train'][idx_train]

            idx_test = np.random.choice(len(Y[func_id]['test']), batch_size)
            x_test, y_test = X['test'][idx_test, :], Y[func_id]['test'][idx_test]

            # calculate the loss and derivatives:
            p = np.dot(x_train, w[func_id])
            test_p = np.dot(x_test, w[func_id])
            loss = np.mean((p - y_train) ** 2 + 0.5 * lamb * np.power(np.linalg.norm(w[func_id], 2), 2))
            iteration_test_loss = np.mean((test_p - y_test) ** 2 + 0.5 * lamb * np.power(np.linalg.norm(w[func_id], 2), 2))
            dl_dw = np.mean(2 * ((p - y_train).reshape(p.shape))[:, None] * x_train + 0.5 * lamb * np.power(np.linalg.norm(w[func_id], 2), 2))

            # update the model and record the loss:
            w[func_id] -= learning_rate * dl_dw
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)

    return w, training_loss, test_loss

def forward(cnn_model, x):
    """
    Given the CNN model, fill up a dictionary with the forward pass values.
    :param cnn_model: the model
    :param x: the input of the CNN
    :return: a dictionary with the forward pass values
    """
    fwd = {}
    fwd['x'] = x
    fwd['o1'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, [np.array(cnn_model['w1'])], mode='same')).astype(np.float64)
    fwd['o2'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, [cnn_model['w2']], mode='same')).astype(np.float64)
    fwd['m'] = np.array([np.maximum(fwd['o1'][:, 0], fwd['o1'][:, 1]), np.maximum(fwd['o1'][:, 2], fwd['o1'][:, 3]),
                          np.maximum(fwd['o2'][:, 0], fwd['o2'][:, 1]), np.maximum(fwd['o2'][:, 2], fwd['o2'][:, 3])]).astype(np.float64)
    fwd['m_argmax'] = np.argmax(fwd['m'])
    fwd['p'] = np.dot(fwd['m'].T, cnn_model['u'])

    return fwd


def find_dl_der_du(p, y, m):
    return 2 * (p - y)[:, None] * m.T


def find_dl_dev_dw(model, y, fwd, batch_size, find_for_w1):
    dl_der_dp = 2 * (fwd['p'] - y)
    dp_der_m1 = np.asarray([u.T for u in model['u']])
    dl_der_dw = []

    for j in range(batch_size):
        dm_der_do = np.zeros(np.shape(fwd['o1'][j, :]))
        if find_for_w1:
            o_bigger_ind = np.where(fwd['o1'][j, :] >= fwd['o2'][j, :])
        else:
            o_bigger_ind = np.where(fwd['o2'][j] >= fwd['o1'][j])
        dm_der_do[o_bigger_ind] = 1

        x_array = np.asarray([[0, fwd['x'][j][0], fwd['x'][j][1], fwd['x'][j][2]],
                              [fwd['x'][j][0], fwd['x'][j][1], fwd['x'][j][2], fwd['x'][j][3]],
                              [fwd['x'][j][1], fwd['x'][j][2], fwd['x'][j][3], 0]])

        if find_for_w1:
            ind = np.where(fwd['o1'][j, :] > 0)
        else:
            ind = np.where(fwd['o2'][j, :] > 0)

        ReLU = np.zeros(np.shape(fwd['o1'][j, :]))
        ReLU[ind] = 1
        do1_der_dw_j = x_array * ReLU
        dl_dw_j = np.dot(dl_der_dp[j] * dp_der_m1 * dm_der_do, do1_der_dw_j.T).astype(np.float64)
        dl_der_dw.append(dl_dw_j)

    return dl_der_dw


def backprop(model, y, fwd, batch_size):
    """
    given the forward pass values and the labels, calculate the derivatives
    using the back propagation algorithm.
    :param model: the model
    :param y: the labels
    :param fwd: the forward pass values
    :param batch_size: the batch size
    :return: a tuple of (dl_dw1, dl_dw2, dl_du)
            dl_dw1: the derivative of the w1 vector
            dl_dw2: the derivative of the w2 vector
            dl_du: the derivative of the u vector
    """
    dl_dw1 = find_dl_dev_dw(model, y, fwd, batch_size, find_for_w1=True)
    dl_dw2 = find_dl_dev_dw(model, y, fwd, batch_size, find_for_w1=False)
    dl_du = find_dl_der_du(fwd['p'], y, fwd['m'])

    return (dl_dw1, dl_dw2, dl_du)


def learn_cnn(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a cnn model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (models, training_loss, test_loss):
            models: a model for every function (a dictionary for the parameters)
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    models = {func_id: {} for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):

        # initialize the model:
        models[func_id]['w1'] = np.random.rand(num_of_functions)
        models[func_id]['w2'] = np.random.rand(num_of_functions)
        models[func_id]['u'] = np.random.rand(size)

        # train the network:
        for _ in range(iterations):

            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx, :], Y[func_id]['train'][idx]

            # calculate the loss and derivatives using back propagation:
            fwd = forward(models[func_id], x)
            loss = np.mean(np.power((fwd['p'] - y), 2) + 0.5 * lamb * (np.power(np.linalg.norm(models[func_id]['w1']), 2)
                                                       + np.power(np.linalg.norm(models[func_id]['w2']), 2)
                                                       + np.power(np.linalg.norm(models[func_id]['u']), 2))).astype(np.float64)
            dl_dw1, dl_dw2, dl_du = backprop(models[func_id], y, fwd, batch_size)

            # record the test loss before updating the model:
            t_i = np.random.choice(len(Y[func_id]['test']), batch_size)
            x_t, y_t = X['test'][t_i, :], Y[func_id]['test'][t_i]
            test_fwd = forward(models[func_id], x_t)
            iteration_test_loss = np.mean((test_fwd['p'] - y) ** 2 + 0.5 * lamb * (np.power(np.linalg.norm(models[func_id]['w1']), 2)
                                                                           + np.power(np.linalg.norm(models[func_id]['w2']), 2)
                                                                           + np.power(np.linalg.norm(models[func_id]['u']), 2))).astype(np.float64)

            # update the model using the derivatives and record the loss:
            models[func_id]['w1'] -= learning_rate * np.mean(dl_dw1, axis=0)
            models[func_id]['w2'] -= learning_rate * np.mean(dl_dw2, axis=0)
            models[func_id]['u'] -= learning_rate * np.mean(dl_du, axis=0)
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)

    return models, training_loss, test_loss


def print_loss(training_loss, test_loss, title):
    plt.figure()
    plt.plot(training_loss, 'b', label="training_loss")
    plt.plot(test_loss, 'r', label="test_loss")
    plt.legend()
    plt.xlabel('Number of iteration')
    plt.ylabel('Func loss')
    plt.title(title)
    plt.show()


def displayResultsOfToyConvent(X, Y):
    batch_size = 100
    learning_rate = 0.001
    iter = 15000
    lamb = 0.001
    cnnLearn = learn_cnn(X, Y, batch_size, lamb, iter, learning_rate)

    print_loss(cnnLearn[1][0], cnnLearn[2][0], "Toy Convent for y1, batch size = 100")
    print_loss(cnnLearn[1][1], cnnLearn[2][1], "Toy Convent for y2, batch size = 100")
    print_loss(cnnLearn[1][2], cnnLearn[2][2], "Toy Convent for y3, batch size = 100")

    cnnLearn = learn_linear(X, Y,batch_size, lamb, iter, learning_rate)

    print_loss(cnnLearn[1][0], cnnLearn[2][0], "Linear model for y1, batch size = 100")
    print_loss(cnnLearn[1][1], cnnLearn[2][1], "Linear model for func y2, batch size = 100")
    print_loss(cnnLearn[1][2], cnnLearn[2][2], "Linear model for func y3, batch size = 100")

    cnnLearn = learn_cnn(X, Y, batch_size, lamb, iter, learning_rate)

    print_loss(cnnLearn[1][0], cnnLearn[2][0], "Toy Convent for y1, batch size = 500")
    print_loss(cnnLearn[1][1], cnnLearn[2][1], "Toy Convent for y2, batch size = 500")
    print_loss(cnnLearn[1][2], cnnLearn[2][2], "Toy Convent for y3, batch size = 500")

    cnnLearn = learn_linear(X, Y, batch_size, lamb, iter, learning_rate)

    print_loss(cnnLearn[1][0], cnnLearn[2][0], "Linear model for y1, batch size = 500")
    print_loss(cnnLearn[1][1], cnnLearn[2][1], "Linear model for y2, batch size = 500")
    print_loss(cnnLearn[1][2], cnnLearn[2][2], "Linear model for y3, batch size = 500")

if __name__ == '__main__':

    # generate the training and test data, adding some noise:
    X = dict(train=5 * (np.random.random((1000, size)) - .5),
             test=5 * (np.random.random((200, size)) - .5))
    Y = {i: {
        'train': y[i](X['train']) * (
        1 + np.random.randn(X['train'].shape[0]) * .01),
        'test': y[i](X['test']) * (
        1 + np.random.randn(X['test'].shape[0]) * .01)}
         for i in range(len(y))}
    displayResultsOfToyConvent(X, Y)
