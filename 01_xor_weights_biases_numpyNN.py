import matplotlib
matplotlib.use('WebAgg')
#['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

import matplotlib.pyplot as plt  # import matplotlib for plotting and visualization
import numpy as np  # import numpy library

class SigmoidLayer:
    """
    This class implements the Sigmoid Layer

    Args:
        shape: shape of input to the layer

    Methods:
        forward(Z)
        backward(upstream_grad)

    """

    def __init__(self, shape):
        """
        The consturctor of the sigmoid/logistic activation layer takes in the following arguments

        Args:
            shape: shape of input to the layer
        """
        self.A = np.zeros(shape)  # create space for the resultant activations

    def forward(self, Z):
        """
        This function performs the forwards propagation step through the activation function

        Args:
            Z: input from previous (linear) layer
        """
        self.A = 1 / (1 + np.exp(-Z))  # compute activations

    def backward(self, upstream_grad):
        """
        This function performs the  back propagation step through the activation function
        Local gradient => derivative of sigmoid => A*(1-A)

        Args:
            upstream_grad: gradient coming into this layer from the layer above

        """
        # couple upstream gradient with local gradient, the result will be sent back to the Linear layer
        self.dZ = upstream_grad * self.A*(1-self.A)

class LinearLayer:
    """
        This Class implements all functions to be executed by a linear layer
        in a computational graph

        Args:
            input_shape: input shape of Data/Activations
            n_out: number of neurons in layer
            ini_type: initialization type for weight parameters, default is "plain"
                      Opitons are: plain, xavier and he

        Methods:
            forward(A_prev)
            backward(upstream_grad)
            update_params(learning_rate)

    """

    def __init__(self, input_shape, n_out, ini_type="plain"):
        """
        The constructor of the LinearLayer takes the following parameters

        Args:
            input_shape: input shape of Data/Activations
            n_out: number of neurons in layer
            ini_type: initialization type for weight parameters, default is "plain"
        """

        self.m = input_shape[1]  # number of examples in training data
        # `params` store weights and bias in a python dictionary
        self.params = initialize_parameters(input_shape[0], n_out, ini_type)  # initialize weights and bias
        self.Z = np.zeros((self.params['W'].shape[0], input_shape[1]))  # create space for resultant Z output

    def forward(self, A_prev):
        """
        This function performs the forwards propagation using activations from previous layer

        Args:
            A_prev:  Activations/Input Data coming into the layer from previous layer
        """

        self.A_prev = A_prev  # store the Activations/Training Data coming in
        self.Z = np.dot(self.params['W'], self.A_prev) + self.params['b']  # compute the linear function

    def backward(self, upstream_grad):
        """
        This function performs the back propagation using upstream gradients

        Args:
            upstream_grad: gradient coming in from the upper layer to couple with local gradient
        """

        # derivative of Cost w.r.t W
        self.dW = np.dot(upstream_grad, self.A_prev.T)

        # derivative of Cost w.r.t b, sum across rows
        self.db = np.sum(upstream_grad, axis=1, keepdims=True)

        # derivative of Cost w.r.t A_prev
        self.dA_prev = np.dot(self.params['W'].T, upstream_grad)

    def update_params(self, learning_rate=0.1):
        """
        This function performs the gradient descent update

        Args:
            learning_rate: learning rate hyper-param for gradient descent, default 0.1
        """

        self.params['W'] = self.params['W'] - learning_rate * self.dW  # update weights
        self.params['b'] = self.params['b'] - learning_rate * self.db  # update bias(es)

def initialize_parameters(n_in, n_out, ini_type='plain'):
    """
    Helper function to initialize some form of random weights and Zero biases
    Args:
        n_in: size of input layer
        n_out: size of output/number of neurons
        ini_type: set initialization type for weights

    Returns:
        params: a dictionary containing W and b
    """

    params = dict()  # initialize empty dictionary of neural net parameters W and b

    if ini_type == 'plain':
        params['W'] = np.random.randn(n_out, n_in) *0.01  # set weights 'W' to small random gaussian
    elif ini_type == 'xavier':
        params['W'] = np.random.randn(n_out, n_in) / (np.sqrt(n_in))  # set variance of W to 1/n
    elif ini_type == 'he':
        # Good when ReLU used in hidden layers
        # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        # Kaiming He et al. (https://arxiv.org/abs/1502.01852)
        # http: // cs231n.github.io / neural - networks - 2 /  # init
        params['W'] = np.random.randn(n_out, n_in) * np.sqrt(2/n_in)  # set variance of W to 2/n

    params['b'] = np.zeros((n_out, 1))    # set bias 'b' to zeros

    return params

def compute_cost(Y, Y_hat):
    """
    This function computes and returns the Cost and its derivative.
    The is function uses the Squared Error Cost function -> (1/2m)*sum(Y - Y_hat)^.2

    Args:
        Y: labels of data
        Y_hat: Predictions(activations) from a last layer, the output layer

    Returns:
        cost: The Squared Error Cost result
        dY_hat: gradient of Cost w.r.t the Y_hat

    """
    m = Y.shape[1]

    cost = (1 / (2 * m)) * np.sum(np.square(Y - Y_hat))
    cost = np.squeeze(cost)  # remove extraneous dimensions to give just a scalar

    dY_hat = -1 / m * (Y - Y_hat)  # derivative of the squared error cost function

    return cost, dY_hat

def predict(X, Y, Zs, As):
    """
    helper function to predict on data using a neural net model layers

    Args:
        X: Data in shape (features x num_of_examples)
        Y: labels in shape ( label x num_of_examples)
        Zs: All linear layers in form of a list e.g [Z1,Z2,...,Zn]
        As: All Activation layers in form of a list e.g [A1,A2,...,An]
    Returns::
        p: predicted labels
        probas : raw probabilities
        accuracy: the number of correct predictions from total predictions
    """
    m = X.shape[1]
    n = len(Zs)  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    Zs[0].forward(X)
    As[0].forward(Zs[0].Z)
    for i in range(1, n):
        Zs[i].forward(As[i-1].A)
        As[i].forward(Zs[i].Z)
    probas = As[n-1].A

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:  # 0.5 is threshold
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    accuracy = np.sum((p == Y) / m)

    return p, probas, accuracy*100

def plot_learning_curve(costs, learning_rate, total_epochs, save=False):
    """
    This function plots the Learning Curve of the model

    Args:
        costs: list of costs recorded during training
        learning_rate: the learning rate during training
        total_epochs: number of epochs the model was trained for
        save: bool flag to save the image or not. Default False
    """
    # plot the cost
    plt.figure()

    steps = int(total_epochs / len(costs))  # the steps at with costs were recorded
    plt.ylabel('Cost')
    plt.xlabel('Iterations ')
    plt.title("Learning rate =" + str(learning_rate))
    plt.plot(np.squeeze(costs))
    locs, labels = plt.xticks()
    plt.xticks(locs[1:-1], tuple(np.array(locs[1:-1], dtype='int')*steps))  # change x labels of the plot
    plt.xticks()
    if save:
        plt.savefig('Cost_Curve.png', bbox_inches='tight')
    plt.show()

def predict_dec(Zs, As, X):
    """
    Used for plotting decision boundary.

    Args:
        Zs: All linear layers in form of a list e.g [Z1,Z2,...,Zn]
        As: All Activation layers in form of a list e.g [A1,A2,...,An]
        X: Data in shape (features x num_of_examples) i.e (K x m), where 'm'=> number of examples
           and "K"=> number of features

    Returns:
        predictions: vector of predictions of our model (red: 0 / green: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    m = X.shape[1]
    n = len(Zs)  # number of layers in the neural network

    # Forward propagation
    Zs[0].forward(X)
    As[0].forward(Zs[0].Z)
    for i in range(1, n):
        Zs[i].forward(As[i - 1].A)
        As[i].forward(Zs[i].Z)
    probas = As[n - 1].A   # output probabilities

    predictions = (probas > 0.5)  # if probability of example > 0.5 => output 1, vice versa
    return predictions

def plot_decision_boundary(model, X, Y, feat_crosses=None, save=False):
    """
    Plots decision boundary

    Args:
        model: neural network layer and activations in lambda function
        X: Data in shape (num_of_examples x features)
        feat_crosses: list of tuples showing which features to cross
        save: flag to save plot image
    """
    # Generate a grid of points between -0.5 and 1.5 with 1000 points in between
    xs = np.linspace(-0.5, 1.5, 1000)
    ys = np.linspace(1.5, -0.5, 1000)
    xx, yy = np.meshgrid(xs, ys) # create data points
    # Predict the function value for the whole grid

    # Z = model(np.c_[xx.ravel(), yy.ravel()])  # => add this for feature cross eg "xx.ravel()*yy.ravel()"

    prediction_data = np.c_[xx.ravel(), yy.ravel()]
    # add feat_crosses if provided
    if feat_crosses:
        for feature in feat_crosses:
            prediction_data = np.c_[prediction_data, prediction_data[:, feature[0]]*prediction_data[:, feature[1]]]

    Z = model(prediction_data)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.style.use('seaborn-whitegrid')
    plt.contour(xx, yy, Z, cmap='Blues')  # draw a blue colored decision boundary
    plt.title('Decision boundary', size=18)
    plt.xlabel('$x_1$', size=20)
    plt.ylabel('$x_2$', size=20)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    # color map 'cmap' maps 0 labeled data points to red and 1 labeled points to green
    cmap = matplotlib.colors.ListedColormap(["red", "green"], name='from_list', N=None)
    plt.scatter(X[:, 0], X[:, 1], s=200, c=np.squeeze(Y), marker='x', cmap=cmap)  # s-> size of marker

    if save:
        plt.savefig('decision_boundary.png', bbox_inches='tight')

    plt.show()

def plot_decision_boundary_shaded(model, X, Y, feat_crosses=None, save=False):
    """
        Plots shaded decision boundary

        Args:
            model: neural network layer and activations in lambda function
            X: Data in shape (num_of_examples x features)
            feat_crosses: list of tuples showing which features to cross
            save: flag to save plot image
    """

    # Generate a grid of points between -0.5 and 1.5 with 1000 points in between
    xs = np.linspace(-0.5, 1.5, 1000)
    ys = np.linspace(1.5, -0.5, 1000)
    xx, yy = np.meshgrid(xs, ys)
    # Predict the function value for the whole grid
    # Z = model(np.c_[xx.ravel(), yy.ravel()]) # => add this for feature cross eg "xx.ravel()*yy.ravel()"

    prediction_data = np.c_[xx.ravel(), yy.ravel()]
    # add feat_crosses if provided
    if feat_crosses:
        for feature in feat_crosses:
            prediction_data = np.c_[prediction_data, prediction_data[:, feature[0]] * prediction_data[:, feature[1]]]

    Z = model(prediction_data)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    cmap = matplotlib.colors.ListedColormap(["red","green"], name='from_list', N=None)
    plt.style.use('seaborn-whitegrid')

    # 'contourf'-> filled contours (red('#EABDBD'): 0 / green('#C8EDD6'): 1)
    plt.contourf(xx, yy, Z, cmap=matplotlib.colors.ListedColormap(['#EABDBD', '#C8EDD6'], name='from_list', N=None))
    plt.title('Decision boundary', size=18)
    plt.xlabel('$x_1$', size=20)
    plt.ylabel('$x_2$', size=20)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.scatter(X[:, 0], X[:, 1], s=200, c=np.squeeze(Y), marker='x', cmap=cmap)  # s-> size of marker

    if save:
        plt.savefig('decision_boundary_shaded.png', bbox_inches='tight')

    plt.show()


def main():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    operator = 'xor'
    if operator == 'and':
        Y = np.array([
            [0],
            [0],
            [0],
            [1]
        ])
    elif operator == 'or':
        Y = np.array([
            [0],
            [1],
            [1],
            [1]
        ])
    elif operator == 'xor':
        Y = np.array([
            [0],
            [1],
            [1],
            [0]
        ])

    X_train = X.T
    Y_train = Y.T

    # define training constants
    learning_rate = 1
    number_of_epochs = 5000

    np.random.seed(48)  # set seed value so that the results are reproduceable

    # Our network architecture has the shape: (input)--> [Linear->Sigmoid] -> [Linear->Sigmoid] -->(output)

    # ------ LAYER-1 ----- define hidden layer that takes in training data
    Z1 = LinearLayer(input_shape=X_train.shape, n_out=3, ini_type='xavier')
    A1 = SigmoidLayer(Z1.Z.shape)

    # ------ LAYER-2 ----- define output layer that take is values from hidden layer
    Z2 = LinearLayer(input_shape=A1.A.shape, n_out=1, ini_type='xavier')
    A2 = SigmoidLayer(Z2.Z.shape)

    # see what random weights and bias were selected and their shape
    print(Z1.params)
    print(Z2.params)

    costs = []  # initially empty list, this will store all the costs after a certian number of epochs

    # Start training
    for epoch in range(number_of_epochs):

        # ------------------------- forward-prop -------------------------
        Z1.forward(X_train)
        A1.forward(Z1.Z)

        Z2.forward(A1.A)
        A2.forward(Z2.Z)

        # ---------------------- Compute Cost ----------------------------
        cost, dA2 = compute_cost(Y=Y_train, Y_hat=A2.A)

        # print and store Costs every 100 iterations.
        if (epoch % 100) == 0:
            print("Cost at epoch#{}: {}".format(epoch, cost))
            costs.append(cost)

        # ------------------------- back-prop ----------------------------
        A2.backward(dA2)
        Z2.backward(A2.dZ)

        A1.backward(Z2.dA_prev)
        Z1.backward(A1.dZ)

        # ----------------------- Update weights and bias ----------------
        Z2.update_params(learning_rate=learning_rate)
        Z1.update_params(learning_rate=learning_rate)

    # See what the final weights and bias are training
    print(Z2.params)
    print(Z2.params)

    # see the ouptput predictions
    predicted_outputs, prediction_orig, accuracy = predict(X=X_train, Y=Y_train, Zs=[Z1, Z2], As=[A1, A2])

    print("The predicted outputs:\n {}".format(prediction_orig))
    # print("The accuracy of the model is: {}%".format(accuracy))

    # # The learning curve
    # plot_learning_curve(costs=costs, learning_rate=learning_rate, total_epochs=number_of_epochs)
    #
    # # The decision boundary
    # plot_decision_boundary(lambda x: predict_dec(Zs=[Z1, Z2], As=[A1, A2], X=x.T), X_train.T, Y_train.T)

    # The shaded decision boundary
    plot_decision_boundary_shaded(lambda x: predict_dec(Zs=[Z1, Z2], As=[A1, A2], X=x.T), X_train.T, Y_train.T)


    print('DONE')

if __name__ == '__main__':
    main()
