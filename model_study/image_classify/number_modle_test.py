import base64
import hashlib
from random import random

import paddle
import numpy as np
import paddle.nn.functional as F
from PIL import Image
# from paddle.vision.transforms import Compose, Normalize
from flask import Flask, request
from flask import jsonify

app = Flask(__name__)


class LeNet(paddle.nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

model = paddle.Model(LeNet())
model.load('./mnist_model')

# transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])
# test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)


@app.route('/number_classify', methods=['POST'])
def number_classify():
    # 从测试集中取出一张图片
    request_body = request.json
    image_base64 = request_body['image']
    print(image_base64)
    image_data = base64.b64decode(image_base64)
    file_name = './tmp/' + hashlib.md5(image_base64.encode(encoding='utf-8')).hexdigest() + '.jpg'
    file = open(file_name, 'wb')
    file.write(image_data)
    file.close()

    print(file_name)
    img = Image.open(file_name)
    print(img)
    img = img.resize((28, 28), Image.ANTIALIAS)  # 把输入图片resieze成28*28尺寸的图片，转换为灰度图
    img_arr = np.array(img.convert('L'))
    print(img_arr)

    # img, label = test_dataset[0]
    # 将图片shape从1*28*28变为1*1*28*28，增加一个batch维度，以匹配模型输入格式要求
    # img_arr = img_arr / 255.0
    img_batch = np.array(img_arr, dtype='float32').reshape(1, 1, 28, 28)
    img_batch = img_batch / 255.0 * 2.0 - 1.0
    # img_batch = np.expand_dims(img_arr.astype('float32'), axis=0)

    # 执行推理并打印结果，此处predict_batch返回的是一个list，取出其中数据获得预测结果
    out = model.predict_batch(img_batch)[0]
    pred_label = out.argmax()
    print('pred label: {}'.format(str(pred_label)))
    # 可视化图片
    # from matplotlib import pyplot as plt
    # plt.imshow(img[0])

    response = {'pred_label': str(pred_label)}

    return response


if __name__ == '__main__':
    app.run()




