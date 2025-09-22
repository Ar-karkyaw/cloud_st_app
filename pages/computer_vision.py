import json
import streamlit as st
from google.cloud import vision

st.set_page_config(page_title="画像認識アプリ", page_icon="🖼️", layout="centered")


credentials_dict = json.loads(st.secrets["google_credentials"], strict=False)
client = vision.ImageAnnotatorClient.from_service_account_info(info=credentials_dict)

@st.cache_data
def get_response(content):
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    return response

@st.cache_data(show_spinner=False)
def get_text(content):
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return response

@st.cache_data(show_spinner=False)
def get_objects(content):
    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    return response

st.title("🖼️ Google Cloud Vision 画像認識アプリ")
st.write("画像をアップロードして、Google Cloud Vision APIで解析してみましょう！")

file =  st.file_uploader("画像ファイルをアップロードしてください", type=["jpg", "jpeg", "png"])

#if file is not None:
    #content = file.getvalue()
    #st.image(content)

#if st.button("解析する"):
    #response = get_response(content)
    #labels =  response.label_annotations
    #.write("Labels:")
    #if response.error.message:
        #raise Exception(
            #f"{response.error.message}\nFor more info on error messages,check: "
            #"https://cloud.google.com/apis/design/errors"
        #)
    #for label in labels:
        #st.write(label.description)

if file:
    content = file.getvalue()
    st.image(content, caption="アップロードされた画像", use_container_width=True)

    # Analysis options
    option = st.radio(
        "解析タイプを選んでください:",
        ("ラベル検出", "テキスト検出", "物体検出"),
        horizontal=True
    )

    if st.button("解析する 🚀"):
        try:
            if option == "ラベル検出":
                response = get_response(content)
                if response.error.message:
                    st.error(f"Error: {response.error.message}")
                else:
                    labels = [
                        {"ラベル": l.description, "スコア": round(l.score * 100, 2)}
                        for l in response.label_annotations
                    ]
                    st.success("ラベル検出の結果")
                    st.dataframe(labels, use_container_width=True)

            elif option == "テキスト検出":
                response = get_text(content)
                if response.error.message:
                    st.error(f"Error: {response.error.message}")
                else:
                    texts = response.text_annotations
                    if texts:
                        st.success("検出されたテキスト")
                        st.write(texts[0].description)
                    else:
                        st.warning("テキストは検出されませんでした。")

            elif option == "物体検出":
                response = get_objects(content)
                if response.error.message:
                    st.error(f"Error: {response.error.message}")
                else:
                    objects = [
                        {"物体": o.name, "スコア": round(o.score * 100, 2)}
                        for o in response.localized_object_annotations
                    ]
                    if objects:
                        st.success("物体検出の結果")
                        st.dataframe(objects, use_container_width=True)
                    else:
                        st.warning("物体は検出されませんでした。")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
else:
    st.info("画像をアップロードしてください。")