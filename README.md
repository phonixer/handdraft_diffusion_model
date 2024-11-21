# 自学扩散模型
太有意思了
![alt text](image.png)


第一步看科华的代码有没有地图

## 2024 1121
看一下MID的数据处理是怎么处理的

### Process_data
这个代码的主要功能是处理和预处理轨迹数据，并将其保存为 .pkl 文件，以便后续用于训练和评估轨迹预测模型。以下是代码的详细解释：

"""from environment import Environment, Scene, Node, derivative_of"""

定义了一些新的数据结构

为每个数据源和数据类别创建一个 Environment 对象，并设置注意力半径。

遍历原始数据文件夹中的所有 .txt 文件，读取并处理数据。

将数据转换为 Scene 和 Node 对象，并计算速度和加速度。

对训练数据进行数据增强（旋转）

将处理后的场景添加到环境中，并使用 dill 序列化保存为 .pkl 文件。