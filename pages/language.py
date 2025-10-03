import json
import requests
import streamlit as st
from annotated_text import annotated_text

class YahooNlpApi:
    post_id = 0

    def __init__(self, client_id):
        self.__client_id = client_id

    @classmethod
    def get_id(cls):
        post_id = cls.post_id
        cls.post_id += 1
        return str(post_id)

    def get_headers(self):
        headers = {
            "Content-Type":"application/json",
            "User-Agent":f"Yahoo AppID: {self.__client_id}"
        }
        return headers

    def parameterize(self, post_id=None, jsonrpc="2.0", method="", params={}):
        if post_id is None:
            post_id = self.get_id()
        else:
            post_id = str(post_id)
        req_obj = {
            "id":post_id,
            "jsonrpc":jsonrpc,
            "method":method,
            "params":params
        }
        return json.dumps(req_obj).encode("utf-8")

    def post(self, url, *args, **kwargs):
        headers = self.get_headers()
        payload = self.parameterize(*args, **kwargs)
        resp = requests.post(url, headers=headers, data=payload)
        return json.loads(resp.content)

    @staticmethod
    def tokenize(token):
        var_names = ["表記","読みがな","基本形表記","品詞","品詞細分類","活用型","活用系"]
        return dict(zip(var_names, token))

    def parse(self, q):
        url = "https://jlp.yahooapis.jp/MAService/V2/parse"
        method = "jlp.maservice.parse"
        params = {"q":q}
        data = self.post(url=url, method=method, params=params)
        tokens = data["result"]["tokens"]
        tokens = list(map(self.tokenize, tokens))
        return tokens

    def extract(self, q):
        url = "https://jlp.yahooapis.jp/KeyphraseService/V2/extract"
        method = "jlp.keyphraseservice.extract"
        params = {"q":q}
        data = self.post(url=url, method=method, params=params)
        tokens = data["result"]["phrases"]
        return tokens

api = YahooNlpApi(st.secrets["yahoo_app_id"])

if "result" not in st.session_state:
    st.session_state["result"] = None
if "keyword" not in st.session_state:
    st.session_state["keyword"] = None

def reset():
    st.session_state["result"] =  None
    st.session_state["keyword"] = None

st.markdown("# 品詞解析")
st.markdown("## 入力")
document = st.text_area("分析したい文章を入力してください。")
mode = st.radio("分析モード", ["形態素解析", "キーワード抽出"], on_change=reset)


if st.button("分析"):
    if mode == "形態素解析":
        st.session_state["result"] = api.parse(document)
    elif mode == "キーワード抽出":
        st.session_state["result"] = api.extract(document)

if st.session_state["result"]:
    st.markdown("## 分析結果")
    if mode == "形態素解析":
        words = list(map(lambda wd:(wd["表記"], wd["品詞"]), st.session_state["result"]))
        annotated_text(words)
    elif mode == "キーワード抽出":
        keywords = list(map(lambda kw:kw["text"], st.session_state["result"]))
        kw = st.selectbox("キーワード", keywords)
        words = document.split(kw)
        for i in range(len(words)-1):
            words.insert(2*i+1, (kw,))
        annotated_text(words)

