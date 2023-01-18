import time
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
dim = 128


def connect_milvus(host, port):
    print(fmt.format("start connecting to Milvus"))
    connections.connect("default", host=host, port=port)

    has = utility.has_collection("hello_milvus_2")
    print(f"Does collection hello_milvus_2 exist in Milvus: {has}")


def create_collection(collection_name, field_name):
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="random", dtype=DataType.DOUBLE),
        FieldSchema(name=field_name, dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    schema = CollectionSchema(fields, "{} is the simplest demo to introduce the APIs".format(collection_name))

    print(fmt.format("Create collection `{}`".format(collection_name)))
    collection = Collection(collection_name, schema, consistency_level="Strong")
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


def create_index(collection, index_type, metric_type, nlist):
    print(fmt.format("Start Creating index IVF_FLAT"))
    index = {
        "index_type": index_type,  # "IVF_FLAT",
        "metric_type": metric_type, # L2
        "params": {"nlist": nlist},
    }

    collection.create_index("audio_feature", index)

    print(fmt.format("Start loading"))
    collection.load()


def search(collection, field_name, vectors_to_search):
    print(fmt.format("Start searching based on vector similarity"))
    # vectors_to_search = entities[-1][0:1]
    print(vectors_to_search)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
        "limit": 1,
    }

    start_time = time.time()
    result = collection.search(vectors_to_search, field_name, search_params, limit=3, output_fields=["random"])
    end_time = time.time()

    for hits in result:
        for hit in hits:
            print(f"hit: {hit}, random field: {hit.entity.get('random')}")
    print(search_latency_fmt.format(end_time - start_time))


def filter_query(collection, field_name):
    # query based on scalar filtering(boolean, int, etc.)
    print(fmt.format("Start querying with `random > 0.5`"))

    start_time = time.time()
    result = collection.query(expr="random > 0.5", output_fields=["random", field_name])
    end_time = time.time()

    print(f"query result:\n-{result[0]}")
    print(search_latency_fmt.format(end_time - start_time))


def pagination_query(collection, offset, limit):
    # pagination
    r1 = collection.query(expr="random > 0.5", limit=limit, output_fields=["random"])
    r2 = collection.query(expr="random > 0.5", offset=offset, limit=limit, output_fields=["random"])
    print(f"query pagination(limit=4):\n\t{r1}")
    print(f"query pagination(offset=0, limit=3):\n\t{r2}")


def hybrid_search(collection, vectors_to_search, field_name, search_params):
    # hybrid search
    print(fmt.format("Start hybrid searching with `random > 0.5`"))

    start_time = time.time()
    result = collection.search(vectors_to_search, field_name, search_params, limit=3, expr="random > 0.5", output_fields=["random"])
    end_time = time.time()

    for hits in result:
        for hit in hits:
            print(f"hit: {hit}, random field: {hit.entity.get('random')}")
    print(search_latency_fmt.format(end_time - start_time))


def delete_entities_by_pk(collection, field_name, expr):
    print(fmt.format(f"Start deleting with expr `{expr}`"))

    result = collection.query(expr=expr, output_fields=["random", field_name])
    print(f"query before delete by expr=`{expr}` -> result: \n-{result[0]}\n-{result[1]}\n")

    collection.delete(expr)

    result = collection.query(expr=expr, output_fields=["random", field_name])
    print(f"query after delete by expr=`{expr}` -> result: {result}\n")


if __name__ == '__main__':
    # get feature data
    feature_list = np.load('./feature_np_arr.npy')

    # 1. connect to Milvus
    connect_milvus("127.0.0.1", "19530")

    # 2. create collection
    field_name = "audio_feature"
    collection = create_collection("milvus_collection_1", field_name)

    # 3. insert data
    insert_result = insert_data(collection, feature_list)

    # 4. create index
    create_index(collection, "HNSW", "L2", 128)

    # 5. search, query, and hybrid search on entities
    search(collection, field_name, [feature_list[0]])

    filter_query(collection, field_name)

    pagination_query(collection, 1, 4)

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
        "limit": 1,
    }
    hybrid_search(collection, [feature_list[0]], field_name, search_params)

    # 6. delete entities by PK
    # expr = f'pk in ["{0}" , "{1}"]'
    # delete_entities_by_pk(collection, field_name, expr)

    # 7. drop collection
    # print(fmt.format("Drop collection `hello_milvus`"))
    # utility.drop_collection("audio_milvus")