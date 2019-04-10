# [CHAPTER 1](http://neuralnetworksanddeeplearning.com/chap1.html)

# [Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)

- #### Goals

  我们这一章的目的是只用74行纯代码来写出一个神经网络模型用来识别手写数字，通过这一章你将体会到神经网络与深度学习的魅力

- #### Problems

  计算机要能识别数字并不像你我一样那么简单，比如你尝试想一下如何写一个程序让计算机来识别图片中的手写数字？我们可以换一个思路，神经网络是怎么处理的呢？它是通过输入大量的training examples,然后开发出一个能从那些training examples中学习的系统，即通过实例来自动参考和形成识别数字的规则

- #### Why

  我们专注于handwriting recognition是因为它是一个很好的学习neural networks的典型问题，可以为以后深入deep learning和其他应用打下良好基础

- ### 让我们赶快开始吧,,

### [Perceptrons](http://neuralnetworksanddeeplearning.com/chap1.html#perceptrons)

这是一种artificial neuron，虽然今天常用的是sigmoid neuron,但通过perceptrons我们才可以理解sigmoid neuron为什么这么定义。

- perceptrons输入几个二进制数x1,x2,...，产生一个二进制输出

  ![img](http://neuralnetworksanddeeplearning.com/images/tikz0.png)

- Rosenblatt提出了一种计算输出的方法。即引入权重w1,w2,…实数表示相应输入对输出的重要性。神经元输出0或1根据加权和![img](file:///C:/Users/win10/AppData/Local/Temp/msohtmlclip1/01/clip_image001.png)是小于或大于一个threshold value.这个阈值也是一个实数，它是这个神经元的参数。数学表达式如下：

  ![1554863669057](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554863669057.png)

- 这就是最基础的数学模型。你可以把神经元当作一个权衡各种依据来做决定的设备。通过调整weights和threshold value，我们可以得到不同的决策模型

- 通过增加神经元个数与层数，我们可以做出很复杂的决策模型

  ![1554863828419](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554863828419.png)

- 上图中第一层为input layer,最后一层为output layer,其余的均为hidden layer,可以想象，随着层数的增加，我们可以拟合出很复杂的模型，因为每一层都是基于上一层所作的更复杂的决定

- 我们可以简化一下perceptrons的描述方式:

  ![1554864066587](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554864066587.png)

  上图中b=-threshold,只是做了一个vectorization和移项，b即bias，描述了神经元激发的难易程度，b越大，越容易激发

- perceptrons可用来进行逻辑运算

  ![1554864291945](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554864291945.png)

  可自行带入x1,x2计算得知为一个NAND gate.这里要注意一点，通过这个NAND gate我们可以通过增加层数来计算任意逻辑函数

  ![1554864444929](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554864444929.png)

  我们可以用perceptrons来代替上述逻辑结构，把每个与非门转换为一个perceptrons,weights为-2，bias为3

  ![img](http://neuralnetworksanddeeplearning.com/images/tikz4.png)

  注意到上图中最下一层的perceptrons得到了来自同一个perceptron的2个输入，weights均为-2，可简化以一个输入

  ![img](http://neuralnetworksanddeeplearning.com/images/tikz5.png)

  目前为止我们还把x1,x2当作浮点型变量，实际上更常用的是输入层也换为perceptrons,但这个perceptron没有输入，单纯的输出它的activation值，即圆圈内的值

  ![img](http://neuralnetworksanddeeplearning.com/images/tikz6.png)

### [Sigmoid neurons](http://neuralnetworksanddeeplearning.com/chap1.html#sigmoid_neurons)

为了引入sigmoid neurons,我们先想象一下我们有一个perceptrons neural network，输入为raw pixel data from a scanned, handwritten image of a digit，我们想让神经网络来学习weights和 biases来让神经网络的输出正确分类识别数字。为了看这个学习过程是怎么工作的，我们希望我们对weights(bias)做一个微小的改变，相应的输出也只有微小的改变。下图即为我们想要的效果

![img](http://neuralnetworksanddeeplearning.com/images/tikz8.png)

如果能达到这种效果，如果我们的网络误判9为8，我们就可以通过修改weights和bias来使得结果不断地一点点靠近9，不断的重复来获得更好的输出，那么就完成了学习过程

但实际上，因为perceptron的原因，我们对输入的微小改变可能导致输出flip,即从0变到1.这样的话，如果我们确实通过修改后可以识别9，我们这个网络对于其它所有的图像的表现可能都会有很大的改变，这是很不好的，这个问题可被sigmoid neuron解决

下面来看什么是sigmoid neuron

- 与perceptron相比其他的基本一样，差别在于以下几点：

  - 输入输出值可取0~1内任何值比如0.638

  - 输入不再是wx+b，而是![1554865975692](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554865975692.png)

  - 其中sigmoid function定义如下

    ![1554866031675](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554866031675.png)

    带入z后可以得到具体输出为

    ![1554866064536](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554866064536.png)

    神经元的结构还是一样

    ![img](http://neuralnetworksanddeeplearning.com/images/tikz9.png)

  - 实际上，2者很相似。我们假设
    $$
    若z≡w⋅x+b>>0,则e^{-z}\approx0，\sigma(z)\approx1
    $$

    $$
    若z≡w⋅x+b<<0,则e^{-z}\rightarrow\infty，\sigma(z)\approx0
    $$

    你会发现这实际上和perceptrons做的事一样

- 接下来我们看看sigmoid function的图像

![1554867255547](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554867255547.png)

可以看作是一个阶跃函数的smooth版本

![1554867304979](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554867304979.png)

事实上perceptron的功能就和阶跃函数一个道理.

而光滑意味着对于weights和bias小的改变会导致输出的小的改变
$$
而实际上，微积分告诉我们\Delta output可以被很好的如下估计：
$$

$$
Δoutput≈∑\frac{\partial output}{\partial wj}Δwj+\frac{\partial output}{\partial b}Δb
$$

上述的式子告诉我们Δoutput是ΔWj和Δb的线性函数，这使得我们可以很容易地选择小的改变来达成输出上所想要的改变。因为输出是0~1内任意数值，所以我们可以用它来表示一些连续变化的东西，但如果我们想通过输出来进行一个二元判断，我们就也可以选一个值比如0.5来做分界.

### [The architecture of neural networks](http://neuralnetworksanddeeplearning.com/chap1.html#the_architecture_of_neural_networks)

- 接下来我们来看一些neural networks的术语

  ![img](http://neuralnetworksanddeeplearning.com/images/tikz11.png)

  - 有一点要注意，出于一些历史原因，上图类似的多层网络有时也被叫做*multilayer perceptrons* or *MLPs* ，尽管它并不是由perceptron构成，这点知道即可
  - 有关输入与输出层的设计很直接，依据具体问题来判断。而对于hidden layer的设计，就很有艺术和技巧了，我们后面会谈到
  - 我们目前为止谈到的网络都是一层的输出用作下一层的输入，这被称为*feedforward* neural networks.即网络中没有loops.

### [A simple network to classify handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html#a_simple_network_to_classify_handwritten_digits)

让我们回到我们一开始的目标，我们可以把识别手写数字分为2布。第一步是找到一种可以把图片中的数字分割开的方法，即把下图中的数字拆开

![1554870990338](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554870990338.png)

![1554870998809](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554870998809.png)

一旦拆开之后，我们的任务就变成了识别单个数字，我们的目标就是写一个程序来解决这个问题，因为事实上第一个任务并不是很难，可以参考相关资料，比如用小的filter取判断，这里不涉及了。而为了解决第二个任务，我们用如下3层神经网络去做

![img](http://neuralnetworksanddeeplearning.com/images/tikz12.png)

其中，输入层的神经元用于处理输入的像素点(our training data for the network will consist of many 28 by 28 pixel images of scanned handwritten digits, and so the input layer contains 784=28×28 neurons.)而且我们输入的像素点的灰度值，0表示纯白，1表示纯黑。

隐藏层我们用了n=15个neurons,这里只是举个例子，我们也可以试验其他的n,最后是输出层。输出层有10个neuron,If the first neuron fires, i.e., has an output ≈1, then that will indicate that the network thinks the digit is a 0.其余的以此类推。

可能有人会想输出用4位2进制数表示不也行吗？但这样很难如10位一样有具体的与数字的组成部分的对应关系，这里我不细讲了

### [Learning with gradient descent](http://neuralnetworksanddeeplearning.com/chap1.html#learning_with_gradient_descent)

接下来讲讲我们用来优化weights和bias的算法，即梯度下降。为了方便，我们用x来表示输入，它是一个28×28=783维的vector,相应的我们想要的输出(即正确的数字)表示为y=y(x)，y为一个10维vector。

- 首先我们定义一个cost function用于表征数字识别的好坏
  $$
  C(w,b)≡\frac{1 }{2n}\sum_{x}^{}\left \|y(x)−a\right \|^2
  $$
  上式中，w为weights,b为biases,n为训练样本数，a为根据网络计算出的预测数字，为一个10维vector，注意这个求和是针对所有训练输入样本的。如果C(w,b)≈0，那么我们的算法就做的很好，那么如何做到呢，通过梯度下降

- 具体还是用一个我们熟悉的例子，要让一个小球在山谷中尽快下落并达到全局最低，我们要做的就是计算出那一点的梯度，回顾一下数学中的梯度定义，它表征了最快增长的方向，我们只需计算出梯度，那么其相反方向即为最快下降方向，我们就是通过这个来进行更新w和b,公式推导很简单，我就省略了:
  $$
  wk→w′k=wk−\eta \frac{\partial C}{\partial wk}
  $$

$$
bl→b′l=bl−\eta \frac{\partial C}{\partial bl}
$$

​	通过不断调用上述方法就可以不断的优化，最终找到一个cost function的全局最小

- 我们这里提一下梯度下降的一个问题，因为求C时我们是对所有训练样本求和，如果x非常大，那么光进行一步更新就要花大量的时间，我们很难较快的看到优化效果。所以，我们有了随机梯度下降法，可用来加速学习，我们随机选择输入样本中的一部分进行梯度下降，使得

  ![1554873916957](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554873916957.png)

  那么相应的有

  ![1554873947222](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554873947222.png)

  证明我们可以这么做，之后应用到我们的梯度下降上

  ![1554874037352](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554874037352.png)

  这样我们就每次取m个样本训练，知道耗尽训练集，这相当于完成了一轮训练，之后我们再不断一轮轮重复

- 还有一点，我们虽然是以二维为例子来研究，但这是通用的，可以拓展到高维

### [Implementing our network to classify digits](http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits)

现在开始看我们的代码，注意我们预先把60000个训练集分为2个，1个有50000个样本，叫做training set,另一组10000个作为交叉验证集，下面看代码，我们定义了一个Network类用于描述我们的神经网络

```python
""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
```

具体训练时，我们按照以下步骤：

```python
import mnist_loader
training_data, validation_data, test_data = \
... mnist_loader.load_data_wrapper()
import network
net = network.Network([784, 30, 10]) //这个表示结构为3层，神经元个数为784，30，10
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)//调用随机梯度下降训练

//输出样例如下
Epoch 0: 9129 / 10000
Epoch 1: 9295 / 10000
Epoch 2: 9348 / 10000
...
Epoch 27: 9528 / 10000
Epoch 28: 9542 / 10000
Epoch 29: 9534 / 10000
```

我们省略了一个load数据的模块如下

```python
"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
```

### [Toward deep learning](http://neuralnetworksanddeeplearning.com/chap1.html#toward_deep_learning)

这一段原文说的很好，我就附上原文了，大意是，通过不断分解复杂的问题比如面部识别，为如何识别眼睛，鼻子，嘴这种子任务，再把子任务不断分解下去，直到每个子任务一个neural即可解决，那么我们就可以用神经网络解决非常复杂的问题。这也被叫做deep neural network,其中hidden layers可多达几十层

While our neural network gives impressive performance, that performance is somewhat mysterious. The weights and biases in the network were discovered automatically. And that means we don't immediately have an explanation of how the network does what it does. Can we find some way to understand the principles by which our network is classifying handwritten digits? And, given such principles, can we do better?

To put these questions more starkly, suppose that a few decades hence neural networks lead to artificial intelligence (AI). Will we understand how such intelligent networks work? Perhaps the networks will be opaque to us, with weights and biases we don't understand, because they've been learned automatically. In the early days of AI research people hoped that the effort to build an AI would also help us understand the principles behind intelligence and, maybe, the functioning of the human brain. But perhaps the outcome will be that we end up understanding neither the brain nor how artificial intelligence works!

To address these questions, let's think back to the interpretation of artificial neurons that I gave at the start of the chapter, as a means of weighing evidence. Suppose we want to determine whether an image shows a human face or not:

![1554875618347](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1554875618347.png)

We could attack this problem the same way we attacked handwriting recognition - by using the pixels in the image as input to a neural network, with the output from the network a single neuron indicating either "Yes, it's a face" or "No, it's not a face".

Let's suppose we do this, but that we're not using a learning algorithm. Instead, we're going to try to design a network by hand, choosing appropriate weights and biases. How might we go about it? Forgetting neural networks entirely for the moment, a heuristic we could use is to decompose the problem into sub-problems: does the image have an eye in the top left? Does it have an eye in the top right? Does it have a nose in the middle? Does it have a mouth in the bottom middle? Is there hair on top? And so on.

If the answers to several of these questions are "yes", or even just "probably yes", then we'd conclude that the image is likely to be a face. Conversely, if the answers to most of the questions are "no", then the image probably isn't a face.

Of course, this is just a rough heuristic, and it suffers from many deficiencies. Maybe the person is bald, so they have no hair. Maybe we can only see part of the face, or the face is at an angle, so some of the facial features are obscured. Still, the heuristic suggests that if we can solve the sub-problems using neural networks, then perhaps we can build a neural network for face-detection, by combining the networks for the sub-problems. Here's a possible architecture, with rectangles denoting the sub-networks. Note that this isn't intended as a realistic approach to solving the face-detection problem; rather, it's to help us build intuition about how networks function. Here's the architecture:



![img](http://neuralnetworksanddeeplearning.com/images/tikz14.png)



It's also plausible that the sub-networks can be decomposed. Suppose we're considering the question: "Is there an eye in the top left?" This can be decomposed into questions such as: "Is there an eyebrow?"; "Are there eyelashes?"; "Is there an iris?"; and so on. Of course, these questions should really include positional information, as well - "Is the eyebrow in the top left, and above the iris?", that kind of thing - but let's keep it simple. The network to answer the question "Is there an eye in the top left?" can now be decomposed:



![img](http://neuralnetworksanddeeplearning.com/images/tikz15.png)



Those questions too can be broken down, further and further through multiple layers. Ultimately, we'll be working with sub-networks that answer questions so simple they can easily be answered at the level of single pixels. Those questions might, for example, be about the presence or absence of very simple shapes at particular points in the image. Such questions can be answered by single neurons connected to the raw pixels in the image.

The end result is a network which breaks down a very complicated question - does this image show a face or not - into very simple questions answerable at the level of single pixels. It does this through a series of many layers, with early layers answering very simple and specific questions about the input image, and later layers building up a hierarchy of ever more complex and abstract concepts. Networks with this kind of many-layer structure - two or more hidden layers - are called *deep neural networks*.

Of course, I haven't said how to do this recursive decomposition into sub-networks. It certainly isn't practical to hand-design the weights and biases in the network. Instead, we'd like to use learning algorithms so that the network can automatically learn the weights and biases - and thus, the hierarchy of concepts - from training data. Researchers in the 1980s and 1990s tried using stochastic gradient descent and backpropagation to train deep networks. Unfortunately, except for a few special architectures, they didn't have much luck. The networks would learn, but very slowly, and in practice often too slowly to be useful.

Since 2006, a set of techniques has been developed that enable learning in deep neural nets. These deep learning techniques are based on stochastic gradient descent and backpropagation, but also introduce new ideas. These techniques have enabled much deeper (and larger) networks to be trained - people now routinely train networks with 5 to 10 hidden layers. And, it turns out that these perform far better on many problems than shallow neural networks, i.e., networks with just a single hidden layer. The reason, of course, is the ability of deep nets to build up a complex hierarchy of concepts. It's a bit like the way conventional programming languages use modular design and ideas about abstraction to enable the creation of complex computer programs. Comparing a deep network to a shallow network is a bit like comparing a programming language with the ability to make function calls to a stripped down language with no ability to make such calls. Abstraction takes a different form in neural networks than it does in conventional programming, but it's just as important.