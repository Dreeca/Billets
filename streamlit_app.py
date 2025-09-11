import streamlit as st
import numpy as np
import pandas as pd
from onnxruntime import InferenceSession

# Show title and description.
st.title("📄 Détection de faux billets")
st.write(
    "Selectionner un fichier pour détecter les faux billets"
)

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.csv)", type=("csv")
)

model_path= 'model.onnx'

if uploaded_file:
    df_billets = pd.read_csv(uploaded_file, index_col='id')
    st.write(df_billets)

    # Stream the response to the app using `st.write_stream`.
    # st.write_stream(stream)
    model = InferenceSession(model_path, providers=['CPUExecutionProvider'])


    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    prediction = model.run([label_name], {input_name: df_billets.to_numpy().astype(np.float32)})[0]
    df_result = pd.DataFrame(prediction, index = df_billets.index, columns=['is_genuine'])
    df_result = df_result.map(lambda x: x == 1)
    st.write(df_result)
