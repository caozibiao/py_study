import cv2
from pathlib import Path
from glob import glob
import pandas as pd
import towhee
from towhee._types.image import Image
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
connections.connect(host='101.43.48.79', port='19530')


def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    print(df.head())


def ground_truth(path):
    train_path = str(Path(path).parent).replace('test', 'train')
    return glob(train_path + '/*.JPEG')


def read_images(results):
    imgs = []
    for re in results:
        imgs.append(Image(cv2.imread(re.id), 'BGR'))
    return imgs


def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='path', dtype=DataType.VARCHAR, descrition='path to image', max_length=500,
                    is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='image embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        'metric_type': 'L2',
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


def get_collection(file_path):
    collection = create_milvus_collection('reverse_image_search', 2048)

    dc = (
        towhee.glob['path'](file_path)
        .image_decode['path', 'img']()
        .image_embedding.timm['img', 'vec'](model_name='resnet50')
        .ann_insert.milvus[('path', 'vec'), 'mr'](collection=collection)
    )
    print('Total number of inserted data is {}.'.format(collection.num_entities))


if __name__ == '__main__':
    # read_csv('./reverse_image_search/reverse_image_search.csv')
    # img = ground_truth('./reverse_image_search/train')
    # print(img)
    get_collection('./reverse_image_search/train/*/*.JPEG')