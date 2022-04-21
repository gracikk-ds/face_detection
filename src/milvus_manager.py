import pickle
import numpy as np
import pandas as pd
from typing import List
from dynaconf import settings
from logging import getLogger
from model import FaceDetector
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection


log = getLogger("face-detection")
HOST_MILVUS = settings["milvus"]["host"]
PORT_MILVUS = settings["milvus"]["port"]


def remap_it(idx_class, mapper, decode=True):
    if decode:
        result = list(mapper.keys())[list(mapper.values()).index(idx_class)]
    else:
        result = list(mapper.values())[list(mapper.keys()).index(idx_class)]
    return result


def milvus_search(
    embeddings: List[List[float]],
    host: str = HOST_MILVUS,
    port: str = PORT_MILVUS,
    collection_name: str = "embeddings_faces",
):
    connection = connections.connect(host=host, port=port)

    ids_range = list(
        range(connection.get_collection_stats(collection_name)["row_count"])
    )

    maping_df = connection.query(
        collection_name=collection_name,
        expr=f"embedding_id in {ids_range}",
        output_fields=["embedding_id", "label_id"],
    )

    maping_df = (
        pd.DataFrame.from_dict(maping_df)
        .sort_values(by="embedding_id")
        .reset_index(drop=True)
    )

    search_params = {"metric_type": "L2", "params": {"nprobe": 1024}}

    results = connection.search(
        collection_name=collection_name,
        data=embeddings,
        anns_field="embeddings",
        param=search_params,
        limit=3,
        expression=None,
    )

    result_ids = [element.ids for element in results]
    result_dsts_top = [element.distances for element in results]

    concat_df = pd.DataFrame(index=range(len(np.array(result_ids).T[0])))
    for i in range(len(np.array(result_ids).T)):
        df = (
            maping_df.iloc[np.array(result_ids).T[i]].reset_index().loc[:, ["label_id"]]
        )
        concat_df = pd.concat([concat_df, df], axis=1)

    concat_df["result"] = concat_df.mode(axis=1)[0].values.astype(int)
    predictions_label_id = concat_df.loc[:, "result"].values.astype(int)
    indexes = [
        i
        for i, e in enumerate(list(concat_df.values[0][:-1]))
        if e == predictions_label_id[0]
    ]
    dist_mean = np.array(result_dsts_top[0])[indexes].mean()

    with open("./pickles/mapper_faces.pickle", "rb") as handle:
        mapper_dict = pickle.load(handle)

    if dist_mean <= 1:

        predictions = [
            remap_it(class_id, mapper_dict, decode=True)
            for class_id in predictions_label_id
        ]

    else:
        predictions = ["unknown"]

    return predictions


def insert_data_to_milvus_collection(
    labels: List[int],
    embeddings: List[List[float]],
    dim=512,
    host: str = HOST_MILVUS,
    port: str = PORT_MILVUS,
    collection_name: str = "embeddings_faces",
):
    log.info("insert_data_to_milvus_collection")
    connection = connections.connect(host=host, port=port)

    if collection_name not in connection.list_collections():
        schema = CollectionSchema(
            [
                FieldSchema("embedding_id", DataType.INT64, is_primary=True),
                FieldSchema("label_id", DataType.INT64),
                FieldSchema("embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
        )

        collection = Collection(
            name=collection_name, schema=schema, using="default", shards_num=2
        )

        log.info(f"Collection with name {collection.name} was created!")

        # create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        collection.create_index("embeddings", index_params=index_params)

    # insert given data to collection
    collection = Collection("embeddings_faces")
    num_entities = collection.num_entities
    log.info(
        f"num_entities before insert operation: {num_entities}",
    )

    ids = [int(i) for i in range(num_entities, num_entities + len(labels))]
    data = [ids, [int(i) for i in labels], embeddings]
    collection.insert(data=data)
    collection.load()

    log.info(f"num_entities after insert operation: {collection.num_entities}")
    log.info("The process has finished successfully!")


def add_embedding_to_collection(images, labels):
    log.info("add_embedding_to_collection...")

    # open map_dict
    with open("pickles/mapper_faces.pickle", "rb") as handle:
        mapper_dict = pickle.load(handle)
    print("dict len before: ", len(mapper_dict))

    # add new classes
    for label in labels:
        if label not in list(mapper_dict.keys()):
            mapper_dict[label] = len(mapper_dict.keys())

    # save updated dict
    with open("pickles/mapper_faces.pickle", "wb") as handle:
        pickle.dump(mapper_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("dict len after: ", len(mapper_dict))

    # remap labels to idx
    labels = [remap_it(class_id, mapper_dict, decode=False) for class_id in labels]

    # create embeddings for images
    detector = FaceDetector()
    _, embeddings = detector(images)

    # insert embeddings and idx of labels to Milvus collection
    insert_data_to_milvus_collection(labels, embeddings.tolist())
    log.info("The process has finished successfully!")
