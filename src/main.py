import os
import time
import pickle
import numpy as np
from PIL import Image
from mpipe import mesh
import streamlit as st
from pathlib import Path
from dynaconf import settings
from logging import getLogger
from model import FaceDetector
from support import process_the_image, draw_uploaded_images
from milvus_manager import add_embedding_to_collection, milvus_search
from visualization import draw_objects
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    drop_collection
)


log = getLogger("face-detection")
HOST_MILVUS = settings["milvus"]["host"]
PORT_MILVUS = settings["milvus"]["port"]

# Constants for sidebar dropdown
MAIN_PAGE = "Main_Page"
SIDEBAR_OPTION_ADD_NEW_EMBEDDING = "Add New Embedding"
SIDEBAR_OPTION_RUN_DETECTION = "Run Detection"
SIDEBAR_OPTION_RESTORE = "Restore Collection"

SIDEBAR_OPTIONS = [
    MAIN_PAGE,
    SIDEBAR_OPTION_ADD_NEW_EMBEDDING,
    SIDEBAR_OPTION_RUN_DETECTION,
    SIDEBAR_OPTION_RESTORE,
]


@st.cache(max_entries=2, suppress_st_warning=True)
def restore_collection(
    host: str = HOST_MILVUS,
    port: str = PORT_MILVUS,
    collection_name: str = "embeddings_faces",
    value_to_change: int = 1,
):
    log.info("connecting to collection ...")

    connections.connect(host=host, port=port)
    log.info("done!")

    log.info("dropping the collection ...")
    drop_collection(collection_name=collection_name)
    log.info("done!")

    # create new collection
    schema = CollectionSchema(
        [
            FieldSchema("embedding_id", DataType.INT64, is_primary=True),
            FieldSchema("label_id", DataType.INT64),
            FieldSchema("embeddings", dtype=DataType.FLOAT_VECTOR, dim=512),
        ]
    )

    log.info("creating a new collection ...")
    collection = Collection(
        name=collection_name, schema=schema, using="default", shards_num=2
    )
    log.info("done!")

    log.info("inserting embeddings to collection ...")
    # upload embeddings
    with open("./pickles/embeddings_faces.pickle", "rb") as handle:
        embeddings = pickle.load(handle)
    # fill collection with embeddings
    collection.insert(embeddings)
    log.info("done!")

    # create index for fast search
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }
    collection.create_index("embeddings", index_params=index_params)
    collection.load()

    log.info("restoring initial dict state ...")

    # restore initial dict state
    with open("pickles/mapper_faces_storage.pickle", "rb") as handle:
        mapper_dict = pickle.load(handle)
    with open("pickles/mapper_faces.pickle", "wb") as handle:
        pickle.dump(mapper_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("value_to_change: ", value_to_change)

    st.info("The collection was successfully restored to the initial state.")


def show_main_img():
    img = np.array(Image.open("../branded/img_main.png"))
    st.image(image=img)


def add_new_embedding_option_func():
    """create functionality for add_embedding streamlit demo page"""
    st.sidebar.write(" ------ ")
    st.sidebar.write("Here you can upload new images to embedding collection")
    st.sidebar.write(
        "The name of the uploaded image will be "
        "recognized as the embedding class name"
    )
    # Create image uploader button
    embeddings_loadder = st.sidebar.file_uploader(
        label="Upload images",
        type=["png", "jpg", "JPG", "jpeg"],
        accept_multiple_files=True,
    )
    # Create run detection button
    run_button = st.sidebar.button(label="Run")

    image_container = []
    caption_container = []
    if embeddings_loadder and run_button:
        st.write("Uploading embeddings to collection...")
        for embedding in embeddings_loadder:
            result_img, embedding_class = process_the_image(embedding, main_flag=False)
            embedding_class = Path(embedding_class).stem
            os.mkdir("../examples/" + embedding_class)
            Image.fromarray(result_img).save("../examples/" + embedding_class + "/img.jpg")
            i = 0
            while i != 2:
                image_container.append(result_img)
                caption_container.append(embedding_class)
                i += 1

        # add uploaded objects to Milvus collection
        add_embedding_to_collection(image_container, caption_container)

        # draw uploaded images
        draw_uploaded_images(image_container, caption_container)

        st.success("All embeddings was uploaded to collection successfully")


def run_detection_option_func():
    """create functionality for detection streamlit demo page"""
    # Create image uploader button
    image_loader = st.sidebar.file_uploader(
        label="Upload image", type=["png", "jpg", "JPG", "jpeg"]
    )
    image = None

    # Create run detection button
    run_button = st.sidebar.button(label="Run")

    # Image preprocessing
    if image_loader is not None:
        image, image_name = process_the_image(image_loader)

    # Running the model
    if run_button and image is not None:
        start = time.process_time()

        model = FaceDetector()
        result_img, embeddings = model(image)

        result = milvus_search(embeddings)

        if result_img is None:
            st.write("Objects not found")

        else:
            result_img = result_img.astype(np.uint8)
            result_img = np.moveaxis(result_img, 0, -1)
            result_img = np.ascontiguousarray(result_img, dtype=np.uint8)
            result_img = mesh(result_img)
            img_path = [x for x in Path("../examples/" + result[0]).glob("*")][0]
            img = np.array(Image.open(img_path))
            fig = draw_objects(result_img, img, result[0])
            st.pyplot(fig=fig)
            # st.plotly_chart(fig, use_container_width=True)
            st.write("Product Detection, Classification")

        end = time.process_time()
        st.text(f"Model running time: {end - start: 2f} s")

    elif run_button and image is None:
        st.write("Image is not selected.")


def config():
    """Core demo configuration"""
    logo = Image.open("../branded/logoq.jpg")
    st.set_page_config(
        page_title="Face Detection",
        page_icon=logo,
        layout="wide",
    )
    # restoring the collection before run the demo
    restore_collection()

    st.title("Welcome to FaceDetection Demo")
    st.sidebar.info(
        "Any changes made during the demo will be revoked when the session is closed."
    )
    st.sidebar.title("Explore the Following")


def run():
    """main run demo script"""
    log.info("Starting streamlit service...")
    app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)

    # condition for NEW_EMBEDDING page
    if app_mode == SIDEBAR_OPTION_ADD_NEW_EMBEDDING:
        # try:
        add_new_embedding_option_func()
        # except Exception:
        #     st.error("Ups... something went wrong here. Please reload the page")

    # condition for RUN_DETECTION page
    elif app_mode == SIDEBAR_OPTION_RUN_DETECTION:
        # try:
        run_detection_option_func()
        # except Exception:
        #     st.error("Ups... something went wrong here. Please reload the page")

    # condition for restore option
    if app_mode == SIDEBAR_OPTION_RESTORE:
        # restore the collection manually
        button = st.sidebar.button(label="Restore Collection")

        if button:
            restore_collection(
                value_to_change=np.random.randint(low=2, high=1000, size=1)[0]
            )

    if app_mode == MAIN_PAGE:
        show_main_img()


if __name__ == "__main__":
    config()
    run()
