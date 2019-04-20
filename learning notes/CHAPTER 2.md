# [CHAPTER 2](http://neuralnetworksanddeeplearning.com/chap2.html)

# [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)

上一章中我们知道了神经网络是如何通过梯度下降算法来学习相应的Weights和biases.但我们却没有解释梯度到底是怎么算的，这一章我们就介绍反向传播算法来计算梯度.

这一章会涉及比较多的算术运算，我们学习它主要是因为它可以让我们更加深入地了解weights和biases是如何改变整个网络的行为的。

### [Warm up: a fast matrix-based approach to computing the output from a neural network](http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network)

![1555651393391](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555651393391.png)

![img](http://neuralnetworksanddeeplearning.com/images/tikz16.png)

![1555651686795](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555651686795.png)

![img](http://neuralnetworksanddeeplearning.com/images/tikz17.png)

我们有了以上几种表示方式后，我们就可以通过以下公式将它们联系起来
$$
a_{j}^{l}=\sigma\left(\sum_{k} w_{j k}^{l} a_{k}^{l-1}+b_{j}^{l}\right)
$$
我们用矩阵的形式重写上述式子，![1555652071489](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555652071489.png)

![1555652089587](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555652089587.png)

还有一点要注意的就是vectorization，我们对一个矩阵用某个函数，实际上是把这个函数作用在矩阵的每个元素上，现在，我们重写的式子如下
$$
a^{l}=\sigma\left(w^{l} a^{l-1}+b^{l}\right)
$$
![1555652468306](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555652468306.png)

### [The two assumptions we need about the cost function](http://neuralnetworksanddeeplearning.com/chap2.html#the_two_assumptions_we_need_about_the_cost_function)

我们用反向传播算法的目的其实是计算2个对于Cost的偏导数，对于网络中的任意w和b求导.我们还是以平方误差函数为例来讲解
$$
C=\frac{1}{2 n} \sum_{x}\left\|y(x)-a^{L}(x)\right\|^{2}
$$
上式中，n是训练样本的总量，求和是针对所有训练样本求和，y(x)是相应样本的y，L为网络层数

- 现在我们来谈谈有关cost function的2个假设

  - 第一个假设是cost function能被写为对于每一个训练样本的cost的和的平均，即
    $$
    C=\frac{1}{n} \sum_{x} C_{x}
    $$

  这个假设对于quadratic cost function成立，对于这本书之后会谈到的其他cost function也成立；那么我们为什么需要这个假设呢？原因很简单，因为反向传播算法只允许我们对单个训练样本求相应partial derivative,只有第一个假设满足我们才能对每一个样本求偏导后平均

  - 第二个假设即cost能被写为是神经网络输出的一个函数

    ![img](http://neuralnetworksanddeeplearning.com/images/tikz18.png)

    我们的quadratic cost显然满足，对于每一个训练样本
    $$
    C=\frac{1}{2}\left\|y-a^{L}\right\|^{2}=\frac{1}{2} \sum_{j}\left(y_{j}-a_{j}^{L}\right)^{2}
    $$

### [The Hadamard product, s⊙t](http://neuralnetworksanddeeplearning.com/chap2.html#the_hadamard_product_$s_\odot_t$)

我们用s⊙t表示2个相同维度向量的对应元素乘积
$$
(s \odot t)_{j}=s_{j} t_{j}
$$
举个例子
$$
\left[ \begin{array}{l}{1} \\ {2}\end{array}\right] \odot \left[ \begin{array}{l}{3} \\ {4}\end{array}\right]=\left[ \begin{array}{l}{1 * 3} \\ {2 * 4}\end{array}\right]=\left[ \begin{array}{l}{3} \\ {8}\end{array}\right]
$$
这个运算有时也被称为*Hadamard product* or *Schur product*。

### [The four fundamental equations behind backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html#the_four_fundamental_equations_behind_backpropagation)

![1555663160059](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555663160059.png)

要理解上文所说的error是如何被定义的，假设我们的神经网络中有一个恶魔

![img](http://neuralnetworksanddeeplearning.com/images/tikz19.png)

![1555663396668](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555663396668.png)

由上述故事启发，我们可以定义error为
$$
\delta_{j}^{l} \equiv \frac{\partial C}{\partial z_{j}^{l}}
$$
接下来我们给出4个基本等式

- **An equation for the error in the output layer**：
  $$
  \delta_{j}^{L}=\frac{\partial C}{\partial a_{j}^{L}} \sigma^{\prime}\left(z_{j}^{L}\right)
  $$
  这个等式稍微解释一下，如果C并不是很依靠某个输出神经元j,那么对应的误差会很小，这正是我们想要的，而右边的那个sigmoid函数的导数，衡量在相应点处activation function即sigmoid的变化快慢

  这个式子的vectorization形式为：
  $$
  \delta^{L}=\nabla_{a} C \odot \sigma^{\prime}\left(z^{L}\right)
  $$
  ![1555664777156](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555664777156.png)

  对于quadratic function因为它的
  $$
  \nabla_{a} C=\left(a^{L}-y\right)
  $$
  所以
  $$
  \delta^{L}=\left(a^{L}-y\right) \odot \sigma^{\prime}\left(z^{L}\right)
  $$
  ![1555665011280](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555665011280.png)
  $$
  \delta^{l}=\left(\left(w^{l+1}\right)^{T} \delta^{l+1}\right) \odot \sigma^{\prime}\left(z^{l}\right)
  $$
  ![1555665145766](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555665145766.png)

  ![1555665229774](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555665229774.png)

  ![1555665283687](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555665283687.png)
  $$
  \frac{\partial C}{\partial b_{j}^{l}}=\delta_{j}^{l}
  $$
  ![1555665336372](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555665336372.png)
  $$
  \frac{\partial C}{\partial b}=\delta
  $$
  ![1555665413762](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555665413762.png)
  $$
  \frac{\partial C}{\partial w_{j k}^{l}}=a_{k}^{l-1} \delta_{j}^{l}
  $$
  上述等式可以重写为索引更少的
  $$
  \frac{\partial C}{\partial w}=a_{\mathrm{in}} \delta_{\mathrm{out}}
  $$
  ![img](http://neuralnetworksanddeeplearning.com/images/tikz20.png)

  从上述等式中我们可以看出C对于w的偏导若a_in很小，则偏导也会很小，即w学的很慢

![1555665746205](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555665746205.png)

![1555665858687](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555665858687.png)

总结一下，我们已经明白了，weight将会学的很慢如果输入神经元是low_activation或是输出神经元饱和(收敛)，下面这张图即最重要的4个等式

![img](http://neuralnetworksanddeeplearning.com/images/tikz21.png)

#### [Problem](http://neuralnetworksanddeeplearning.com/chap2.html#problem_563815)

![1555666337443](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555666337443.png)

![1555666348972](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555666348972.png)

### [Proof of the four fundamental equations (optional)](http://neuralnetworksanddeeplearning.com/chap2.html#proof_of_the_four_fundamental_equations_(optional))

1. 让我们首先证明等式1

$$
\delta_{j}^{L}=\frac{\partial C}{\partial z_{j}^{L}}
$$

运用链式法则，我们可以把它和输出的activations结合起来
$$
\delta_{j}^{L}=\sum_{k} \frac{\partial C}{\partial a_{k}^{L}} \frac{\partial a_{k}^{L}}{\partial z_{j}^{L}}
$$
这里有一点要注意，输出层的各个神经元是无关的，某一个输出神经元的activation只与它自己的z有关，这个别搞混了

![1555678341253](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555678341253.png)
$$
\delta_{j}^{L}=\frac{\partial C}{\partial a_{j}^{L}} \frac{\partial a_{j}^{L}}{\partial z_{j}^{L}}
$$
![1555678405402](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555678405402.png)
$$
\delta_{j}^{L}=\frac{\partial C}{\partial a_{j}^{L}} \sigma^{\prime}\left(z_{j}^{L}\right)
$$
由此等式1得证

2.接下来我们来证等式2

![1555678589381](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555678589381.png)

同样是用到链式法则
$$
\begin{aligned} \delta_{j}^{l} &=\frac{\partial C}{\partial z_{j}^{l}} \\ &=\sum_{k} \frac{\partial C}{\partial z_{k}^{l+1}} \frac{\partial z_{k}^{l+1}}{\partial z_{j}^{l}} \\ &=\sum_{k} \frac{\partial z_{k}^{l+1}}{\partial z_{j}^{l}} \delta_{k}^{l+1} \end{aligned}
$$
对于上述最后一个式子的第一个因子，我们有：
$$
z_{k}^{l+1}=\sum_{i} w_{k j}^{l+1} a_{j}^{l}+b_{k}^{l+1}=\sum_{i} w_{k j}^{l+1} \sigma\left(z_{j}^{l}\right)+b_{k}^{l+1}
$$
Differentiating, we obtain：
$$
\frac{\partial z_{k}^{l+1}}{\partial z_{j}^{l}}=w_{k j}^{l+1} \sigma^{\prime}\left(z_{j}^{l}\right)
$$
再把上式代入，得：
$$
\delta_{j}^{l}=\sum_{k} w_{k j}^{l+1} \delta_{k}^{l+1} \sigma^{\prime}\left(z_{j}^{l}\right)
$$
这就是等式2的component form

### [The backpropagation algorithm](http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm)

![1555680454072](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555680454072.png)

通过算法描述你就可以知道为什么它叫反向传播，我们从后往前计算error

通常，将backpropagation与其他如gradient descent的学习算法结合起来是很常见的，举个例子

![1555681307353](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555681307353.png)

Of course, to implement stochastic gradient descent in practice you also need an outer loop generating mini-batches of training examples, and an outer loop stepping through multiple epochs of training. I've omitted those for simplicity.

### [The code for backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html#the_code_for_backpropagation)

回忆上一章最后我们实施的算法，现在应当比较清楚了

```python
class Network(object):
...
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
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
```

![1555682110019](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555682110019.png)

```python
class Network(object):
...
   def backprop(self, x, y):
        """Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights"."""
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

...

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y) 

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
```

#### [Problem](http://neuralnetworksanddeeplearning.com/chap2.html#problem_269962)

- **Fully matrix-based approach to backpropagation over a mini-batch** Our implementation of stochastic gradient descent loops over training examples in a mini-batch. It's possible to modify the backpropagation algorithm so that it computes the gradients for all training examples in a mini-batch simultaneously.

- ![1555682646145](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555682646145.png)

  ### [In what sense is backpropagation a fast algorithm?](http://neuralnetworksanddeeplearning.com/chap2.html#in_what_sense_is_backpropagation_a_fast_algorithm)

  后向传播这个算法很快主要就是因为它只需一次foreword和一次backword就能算出所有的partial derivative

  ### [Backpropagation: the big picture](http://neuralnetworksanddeeplearning.com/chap2.html#backpropagation_the_big_picture)

  我们讲了这么多，但针对backpropagation,我们还有2个疑问，一是对于算法中的矩阵和向量乘积，它实际上到底是完成了什么？二是这个算法是如何被合理地推理出来地呢？

- 先看问题一，让我们假设我们对w做了一个很小的改变

  ![img](http://neuralnetworksanddeeplearning.com/images/tikz22.png)

  weight的改变会导致相应输出神经元activation的改变

  ![img](http://neuralnetworksanddeeplearning.com/images/tikz23.png)

  更进一步，会导致下一层所有activations的改变

  ![img](http://neuralnetworksanddeeplearning.com/images/tikz24.png)

  这些改变会不断传递下去，直到最后的输出层和cost function

  ![img](http://neuralnetworksanddeeplearning.com/images/tikz25.png)

  ![1555683620469](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555683620469.png)
  $$
  \Delta C \approx \frac{\partial C}{\partial w_{j k}^{l}} \Delta w_{j k}^{l}
  $$
  ![1555683747273](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555683747273.png)

  ![1555683794621](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555683794621.png)
  $$
  \Delta a_{j}^{l} \approx \frac{\partial a_{j}^{l}}{\partial w_{j k}^{l}} \Delta w_{j k}^{l}
  $$
  ![1555683879580](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555683879580.png)

  ![img](http://neuralnetworksanddeeplearning.com/images/tikz26.png)

  

事实上，它会造成以下改变
$$
\Delta a_{q}^{l+1} \approx \frac{\partial a_{q}^{l+1}}{\partial a_{j}^{l}} \Delta a_{j}^{l}
$$
将这个式子与上个式子联立
$$
\Delta a_{q}^{l+1} \approx \frac{\partial a_{q}^{l+1}}{\partial a_{j}^{l}} \frac{\partial a_{j}^{l}}{\partial w_{j k}^{l}} \Delta w_{j k}^{l}
$$
![1555684083379](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555684083379.png)
$$
\Delta C \approx \frac{\partial C}{\partial a_{m}^{L}} \frac{\partial a_{m}^{L}}{\partial a_{n}^{L-1}} \frac{\partial a_{n}^{L-1}}{\partial a_{p}^{L-2}} \cdots \frac{\partial a_{q}^{l+1}}{\partial a_{j}^{l}} \frac{\partial a_{j}^{l}}{\partial w_{j k}^{l}} \Delta w_{j k}^{l}
$$
![1555684174616](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555684174616.png)
$$
\Delta C \approx \sum_{m n p . . . q} \frac{\partial C}{\partial a_{m}^{L}} \frac{\partial a_{m}^{L}}{\partial a_{n}^{L-1}} \frac{\partial a_{n}^{L-1}}{\partial a_{p}^{L-2}} \ldots \frac{\partial a_{q}^{l+1}}{\partial a_{j}^{l}} \frac{\partial a_{j}^{l}}{\partial w_{j k}^{l}} \Delta w_{j k}^{l}
$$
结合一开始的式子
$$
\Delta C \approx \frac{\partial C}{\partial w_{j k}^{l}} \Delta w_{j k}^{l}
$$
我们可以得到
$$
\frac{\partial C}{\partial w_{j k}^{l}}=\sum_{m n p . \ldots q} \frac{\partial C}{\partial a_{m}^{L}} \frac{\partial a_{m}^{L}}{\partial a_{n}^{L-1}} \frac{\partial a_{n}^{L-1}}{\partial a_{p}^{L-2}} \ldots \frac{\partial a_{q}^{l+1}}{\partial a_{j}^{l}} \frac{\partial a_{j}^{l}}{\partial w_{j k}^{l}}
$$
![1555684534628](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555684534628.png)

![img](http://neuralnetworksanddeeplearning.com/images/tikz27.png)

![1555684701853](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555684701853.png)

![1555684733863](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555684733863.png)

![1555684871227](C:\Users\win10\AppData\Roaming\Typora\typora-user-images\1555684871227.png)

