import streamlit as st
from gensim.models import word2vec
import pandas as pd

# 単語の分散表現を学習するために必要なデータを用意します。
# ここでは、サンプルとして、ダミーのデータを使用します。
# data = ["apple banana orange", "banana mango kiwi", "orange grapefruit lemon"]

# 単語の分散表現を学習します。
df = pd.read_csv('unwiki/unwiki_deploy.csv')
model = word2vec.Word2Vec.load('unwiki/unwiki_model.model')


col1, col2 = st.columns(2)

word = st.text_input("Search word:")

with col1:
  st.title("類義語検索")
  if word != "":
    try:
      similar_words1 = model.wv.most_similar(word)
    except:
      st.write("他の単語を試してください")
    else:
      st.write("Similar words:")
      for w, s in similar_words1:
          st.write(f"{w}: {s:.4f}")
  else:
    st.write("^ 好きな単語を入力してください")


with col2:
  st.title("記事検索")

