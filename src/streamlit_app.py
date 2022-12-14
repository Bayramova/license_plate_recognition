# type: ignore

import numpy as np
from PIL import Image
import streamlit as st
import torch

from license_plate_recognition.recognize import (
    load_label_converter,
    load_model,
    preprocess_image,
)


@st.cache
def cached_label_converter():
    label_converter = load_label_converter()
    return label_converter


@st.cache(allow_output_mutation=True)
def cached_model(label_converter):
    model = load_model(label_converter)
    model.eval()
    return model


def main():
    label_converter = cached_label_converter()
    model = cached_model(label_converter)

    st.title("License Plate Recognition")

    uploaded_file = st.file_uploader("Choose an image:", type="jpg")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, channels="BGR")

        img = preprocess_image(np.array(img))

        with torch.no_grad():
            pred = model(img.unsqueeze(0))

        st.write("Prediction:")
        st.success(f"{label_converter.decode(pred.argmax(-1))}")


if __name__ == "__main__":
    main()
