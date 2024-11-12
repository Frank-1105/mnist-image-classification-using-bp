# dbooker11-mnist-image-classification-using-bp
# BP神经网络实现 - 手写数字分类（MNIST数据集）
该项目使用了一个简单的BP（反向传播）神经网络来对MNIST手写数字数据集进行分类。我们通过定义一个包含两层隐藏层的全连接神经网络，并实现了前向传播、反向传播、训练过程以及最终的分类任务。

# 依赖
运行此代码需要安装以下Python库：
numpy
matplotlib
scikit-learn

# 代码概述
1. 激活函数和损失函数
ReLU函数：用于隐藏层的激活，计算公式为 relu(x) = max(0, x)，通过此函数将负数置为0，保留正数。
Sigmoid函数：用于二分类任务的激活函数，公式为 sigmoid(x) = 1 / (1 + exp(-x))。
Softmax函数：用于输出层，归一化输出为概率分布，适用于多分类问题。
交叉熵损失：用于多分类任务的损失函数，计算公式为 cross_entropy_loss(y_true, y_pred)。
2. BP神经网络（BPNeuralNetwork类）
该类实现了一个具有两层隐藏层的全连接神经网络，支持训练和预测。
网络结构：
1.输入层：784个节点（MNIST图像大小为28x28，即784个像素）
2.第一隐藏层：128个节点，ReLU激活函数
3.第二隐藏层：64个节点，ReLU激活函数
4.输出层：10个节点，Softmax激活函数（对应10个数字类别）
主要功能：
1.forward(X)：前向传播函数，计算每层的输出。
2.backward(X, y, output)：反向传播函数，计算梯度并更新网络权重。
3.train(X, y, epochs=1000)：训练函数，执行多次迭代，更新网络参数，降低损失。
4.predict(X)：预测函数，给定输入，返回网络预测的类别标签。
3. 数据预处理
该代码使用fetch_openml加载MNIST数据集。数据经过以下处理：
1.归一化：将每个像素值除以255，使得像素值范围从[0, 255]映射到[0, 1]。
2.One-Hot编码：将标签（0-9的数字）转换为One-Hot编码，以便适应多分类的损失函数。
4. 训练过程
1.学习率（learning_rate）：初始学习率为0.036。
2.正则化系数（reg_lambda）：用于L2正则化，值为0.005。
3.衰减（decay）：学习率衰减系数，这里设置为0以便学习率不衰减。
4.训练周期（epochs）：模型训练2000个epoch。
每经过一定的epoch，程序会打印当前的损失值。
5. 预测与准确率
训练完成后，使用训练得到的模型对测试集进行预测，并计算准确率。最终输出测试集的准确率。
# 运行方法
1. 数据加载与预处理
首先，程序加载MNIST数据集，并进行归一化和One-Hot编码。
2. 模型训练
使用BPNeuralNetwork类创建一个神经网络对象，并通过调用train方法开始训练。
3. 评估模型
训练完成后，使用predict方法对测试集进行预测，并计算准确率。
# 示例
```python
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(np.int32)

X = X / 255.0  # 归一化

encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(np.array(y).reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
print("划分成功...")

# 创建神经网络对象
nn = BPNeuralNetwork(input_size=784, hidden_sizes=[128, 64], output_size=10, learning_rate=0.036, decay=0, reg_lambda=0.005)
print("开始训练...")
nn.train(X_train, y_train, epochs=2000)

# 预测并计算准确率
y_pred = nn.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```
# 结果示例
```python
Epoch 0/2000 - Loss: 0.301982
Epoch 1000/2000 - Loss: 0.001568
Epoch 1999/2000 - Loss: 0.000942
Test Accuracy: 94.15%
```
# 总结
本项目实现了一个简单的三层（输入层 + 两层隐藏层 + 输出层）BP神经网络来处理MNIST数据集的分类任务。通过使用反向传播算法进行训练，最终得到了较为优异的分类准确率。这个神经网络虽然结构较为简单，但它展示了神经网络的基本实现方法，可以为更复杂的深度学习模型打下基础。
# 未来改进
优化模型结构：增加更多的隐藏层，尝试其他激活函数（如Leaky ReLU、Tanh等）。
改进训练过程：使用更多的优化算法（如Adam、RMSprop等）来加速收敛。

