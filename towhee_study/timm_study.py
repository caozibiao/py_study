from towhee.dc2 import pipe, ops, DataCollection

p = (
        pipe.input('path')
            .map('path', 'img', ops.image_decode())
            .map('img', 'vec', ops.image_embedding.timm(model_name='resnet50'))
            .output('img', 'vec')
    )

if __name__ == '__main__':
    feature = DataCollection(p('https://towhee.io/assets/img/logo-title.png'))[0]['vec']
    print(len(feature))