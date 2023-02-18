import paddle
import paddlehub as hub
import paddle.fluid as fluid


def get_feature(path):
    # 使用paddlehub的轻量预训练模型
    module = hub.Module(name="shufflenet_v2_imagenet")
    inputs, outputs, program = module.context(trainable=False)
    data = [path]
    img = inputs["image"]
    feed_list = [img.name]
    feature_map = outputs["feature_map"]

    # 图片读取器
    data_reader = hub.reader.ImageClassificationReader(
        image_width=module.get_expected_image_width(),
        image_height=module.get_expected_image_height(),
        images_mean=module.get_pretrained_images_mean(),
        images_std=module.get_pretrained_images_std(),
        dataset=None)

    predict_reader = data_reader.data_generator(
        phase="predict", batch_size=1, data=data)

    # 只获取预训练图像分类模型的特征层
    fetch_list = list(set([outputs['feature_map']]))
    # fetch_list = list(set([value for key, value in outputs.items()]))

    # 喂入图片，获取图片的特征层（特征提取）
    with fluid.program_guard(program):
        # 网络参数初始化
        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)

        feeder = fluid.DataFeeder(feed_list=feed_list, place=cpu)
        # result获取图片的特征层
        for batch in predict_reader():
            result, = exe.run(feed=feeder.feed(batch), fetch_list=fetch_list)

    # 输出图片特征层
    return result


if __name__ == '__main__':
    print(get_feature('/Users/caozibiao/Desktop/github/py_study/model_study/extract_image_feature/image/'))