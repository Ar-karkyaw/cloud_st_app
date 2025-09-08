import streamlit as st
import pickle
import pandas as pd

MODEL_PATH = "./assets/model2.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

if "done" not in st.session_state:
    st.session_state["done"] = False

def toggle_done(value=True):
    st.session_state["done"] = value

st.markdown("# メンタルヘルススコアの見積もり")

load_state = st.markdown("学習済みモデルの読み込み中...")
model = load_model()
load_state.markdown("")

prep = model.named_steps["preprocessor"]
colnames = prep.feature_names_in_.tolist()
cat_trans = prep.named_transformers_["cat"]
cat_colnames = cat_trans.feature_names_in_.tolist()
cat_values_dict = dict(zip(cat_colnames, cat_trans.categories_))
regressor_columns = prep.get_feature_names_out()
regressor = model.named_steps["regressor"]
feature_importances = pd.DataFrame({
    "columns": regressor_columns,
    "importance": regressor.feature_importances_
})

with st.form("入力"):
    col1_1, col1_2, col1_3 = st.columns([1,1,1])
    col2_1, col2_2, col2_3 = st.columns([1,1,1])
    col3_1, col3_2, col3_3 = st.columns([1,1,1])

    with col1_1:
        age = st.number_input("年齢", min_value=0, max_value=200, value=25, step=1)

    with col1_2:
        avg_hour = st.number_input("1日平均SNS使用時間", min_value=0., max_value=24., value=0., step=0.1)

    with col1_3:
        gender = st.radio("性別", ["Female", "Male"], index=None)

    with col2_1:
        jp_idx = cat_values_dict["Country"].tolist().index("Japan")
        country = st.selectbox("国", cat_values_dict["Country"], index=jp_idx)

    with col2_2:
        platform = st.selectbox("最もよく使用するSNS", cat_values_dict["Most_Used_Platform"], index=None)

    with col2_3:
        academic_level = st.radio("現在の所属", ["High School", "Undergraduate", "Graduate"], index=None)

    with col3_1:
        sleep = st.number_input("1日平均睡眠時間", min_value=0, max_value=24, value=8)

    with col3_2:
        conflicts = st.number_input("SNSに関する家族や友人などとの言い争いの数", min_value=0, value=0)

    with col3_3:
        relationship = st.radio("交際状況", ["Single", "In Relationship", "Complicated"], index=None)

    st.form_submit_button("決定", on_click=toggle_done, args=[True])

if st.session_state["done"]:
    record = {"Age": age,
            "Gender": gender,
            "Academic_Level": academic_level,
            "Country": country,
            "Avg_Daily_Usage_Hours": avg_hour,
            "Most_Used_Platform": platform,
            "Sleep_Hours_Per_Night": sleep,
            "Relationship_Status": relationship,
            "Conflicts_Over_Social_Media": conflicts
            }
    features = pd.DataFrame([record])
    prediction = model.predict(features)[0]
    st.metric("メンタルヘルス指数", prediction)

    with st.expander("参考:入力データ", expanded=False):
        st.write(record)

with st.expander("参考:モデルの特徴重要度", expanded=False):
        st.bar_chart(feature_importances.query("importance > 0"),
                    x="columns", y="importance", horizontal=True)