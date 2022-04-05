import numpy as np
from math import ceil
import streamlit as st
from PIL import Image, ImageOps


def process_the_image(image_loader, main_flag=True):
    """Preprocessing uploaded image"""

    image_name = image_loader.name
    if main_flag:
        st.sidebar.image(image_loader)
    image = Image.open(image_loader).convert("RGB")
    image = ImageOps.exif_transpose(image)
    max_side = max(image.size)
    if max_side > 2000:
        ratio = 2000 / max_side
        size = (int(s * ratio) for s in image.size)
        image = image.resize(size, Image.ANTIALIAS)
    image = np.array(image)

    return image, image_name


def draw_uploaded_images(image_container, caption_container):
    """
    Draw images uploaded to milvus collection
    :param image_container: container of images uploaded to milvus
    :param caption_container: container of images classes
    :return: None
    """
    # show uploaded objects
    images_to_show = image_container[::2]
    caption_to_show = caption_container[::2]
    len_ = len(images_to_show)
    rows = [element for element in range(ceil(len_ / 3))]

    for row in rows:
        image_container_filtred = images_to_show[row * 3: (row + 1) * 3]
        caption_container_filtred = caption_to_show[row * 3: (row + 1) * 3]
        cols = st.columns(len(image_container_filtred))
        for i, col in enumerate(cols):
            col.image(
                image_container_filtred[i],
                width=300,
                caption=caption_container_filtred[i],
            )

