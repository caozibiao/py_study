import time

import numpy as np
import towhee
from towhee.dc2 import pipe, ops, DataCollection
from flask import Flask, request
from flask import jsonify
app = Flask(__name__)
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

HOST = '101.43.48.79'
PORT = '19530'
COLLECTION_NAME = 'text_image_search'
INDEX_TYPE = 'IVF_FLAT'
METRIC_TYPE = 'IP'
DIM = 512
TOPK = 10
fmt = "\n=== {:30} ===\n"

img_pipe = (
        pipe.input('url')
        .map('url', 'img', ops.image_decode.cv2_rgb())
        .map('img', 'vec',
             ops.image_text_embedding.taiyi(model_name='taiyi-clip-roberta-102m-chinese', modality='image'))
        .output('vec')
    )

p = (
        pipe.input('path')
            .map('path', 'img', ops.image_decode())
            .map('img', 'vec', ops.image_embedding.timm(model_name='resnet50'))
            .output('img', 'vec')
    )


@app.route('/get_feature', methods=['POST'])
def get_feature():
    start = time.time()
    # 从测试集中取出一张图片
    request_body = request.json
    image_url = request_body['image_url']
    print(image_url)
    # image_data = base64.b64decode(image_url)

    data = DataCollection(img_pipe(image_url))
    print(data[0]['vec'])
    print(time.time() - start)

    response = {'feature': data[0]['vec'].tolist()}

    print(len(data[0]['vec'].tolist()))

    return jsonify(response)


def create_collection(exist_ok=False):
    try:
        connections.connect(host=HOST, port=PORT)
    except Exception:
        print(f'Fail to connect Milvus with {HOST}:{PORT}')

    if utility.has_collection:
        collection = Collection(COLLECTION_NAME)
        if exist_ok:
            print(f'Using existed collection: {COLLECTION_NAME}.')
            return collection
        else:
            print('Deleting previous collection...')
            collection.drop()

    # Create collection
    print('Creating collection...')
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="random", dtype=DataType.DOUBLE),
        FieldSchema(name="feature", dtype=DataType.FLOAT_VECTOR, dim=DIM)
    ]
    schema = CollectionSchema(fields=fields, description='text image search')
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create index
    print('Creating index...')
    index_params = {
        'metric_type': METRIC_TYPE,
        'index_type': INDEX_TYPE,
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name='feature', index_params=index_params)

    print(f'Milvus collection is ready: {COLLECTION_NAME} ({INDEX_TYPE}, {METRIC_TYPE}).')
    return collection


def insert_data(collection, feature_list):
    num_entities = len(feature_list)
    print(fmt.format("Start inserting entities"))
    rng = np.random.default_rng(seed=19530)
    entities = [
        # provide the pk field because `auto_id` is set to False
        [str(i) for i in range(num_entities)],
        rng.random(num_entities).tolist(),  # field random, only supports list
        feature_list,  # field embeddings, supports numpy.ndarray and list
    ]

    insert_result = collection.insert(entities)
    print(insert_result)

    collection.flush()
    print(f"Number of entities in Milvus: {collection.num_entities}")  # check the num_entites
    return insert_result


@app.route('/insert_milvus', methods=['POST'])
def insert_milvus():
    start = time.time()

    collection = create_collection()
    request_body = request.json
    image_url = request_body['image_url']
    print(image_url)
    data = DataCollection(img_pipe(image_url))
    print(data[0])
    insert_data(collection, [data[0]['vec']])

    print('Total vectors in collection: '.format(collection.num_entities))

    response = {}

    print('spend time: {}'.format(time.time() - start))
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=4000)