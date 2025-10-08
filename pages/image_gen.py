import requests
import streamlit as st

token = st.secrets["hugging_face_token"]
model = "black-forest-labs/FLUX.1-dev"
API_URL = f"https://api-inference.huggingface.co/models/{model}"
headers = {
"Authorization": f"Bearer {token}",
}
prompt = st.text_input("Prompt")
data = {
"inputs": prompt
}
if st.button("Generate"):
    response = requests.post(API_URL, headers=headers, json=data)
    st.image(response.content)