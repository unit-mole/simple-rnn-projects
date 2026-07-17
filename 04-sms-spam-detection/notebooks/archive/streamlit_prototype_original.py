
import re
import numpy as np
import pandas as pd
import streamlit as st

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

MAX_WORDS = 5000
MAX_LEN = 45
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"[^a-z0-9£$%\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_synthetic_sms_dataset(n_per_class=120):
    ham = ["are we meeting today", "please call me later", "thanks for your help", "see you at lunch", "send me the notes"]
    spam = ["free prize claim now", "urgent win cash today", "claim your bonus money", "limited offer click now", "you won free voucher"]
    rows=[]
    for i in range(n_per_class):
        rows.append({"text": ham[i%len(ham)], "label":0})
        rows.append({"text": spam[i%len(spam)], "label":1})
    df=pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df["clean_text"]=df["text"].map(clean_text)
    return df

@st.cache_resource
def train_app_model():
    df=build_synthetic_sms_dataset()
    if TF_AVAILABLE:
        tok=Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tok.fit_on_texts(df["clean_text"])
        X=pad_sequences(tok.texts_to_sequences(df["clean_text"]), maxlen=MAX_LEN, padding="post", truncating="post")
        y=df["label"].values
        model=Sequential([Embedding(MAX_WORDS,32,input_length=MAX_LEN), SimpleRNN(32), Dropout(0.2), Dense(16,activation="relu"), Dense(1,activation="sigmoid")])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X,y,epochs=3,batch_size=16,verbose=0)
        return {"backend":"Simple RNN", "model":model, "tokenizer":tok}
    if SKLEARN_AVAILABLE:
        pipe=Pipeline([("tfidf",TfidfVectorizer(max_features=3000,ngram_range=(1,2))),("clf",LogisticRegression(max_iter=500))])
        pipe.fit(df["clean_text"], df["label"])
        return {"backend":"TF-IDF fallback", "model":pipe, "tokenizer":None}
    return {"backend":"keyword fallback", "model":None, "tokenizer":None}

def predict(system, texts):
    clean=[clean_text(t) for t in texts]
    if system["backend"]=="Simple RNN":
        X=pad_sequences(system["tokenizer"].texts_to_sequences(clean), maxlen=MAX_LEN, padding="post", truncating="post")
        probs=system["model"].predict(X, verbose=0).reshape(-1)
    elif system["backend"]=="TF-IDF fallback":
        probs=system["model"].predict_proba(clean)[:,1]
    else:
        spam_words={"free","win","claim","urgent","prize","cash","offer","bonus"}
        probs=np.array([min(0.95,0.15+0.15*sum(w in c.split() for w in spam_words)) for c in clean])
    return pd.DataFrame({"text":texts,"clean_text":clean,"spam_probability":probs,"prediction":["spam" if p>=0.5 else "ham" for p in probs]})

st.set_page_config(page_title="SMS Spam Detection - Simple RNN", layout="wide")
st.title("SMS Spam Detection using Simple RNN")
st.write("Synthetic validation first, real-style SMS spam classification pipeline. The app uses a lightweight trained model for interactive inference.")
system=train_app_model()
st.sidebar.success(f"Backend: {system['backend']}")
msg=st.text_area("Enter SMS text", "Congratulations you won a free prize claim now")
if st.button("Predict SMS"):
    result=predict(system,[msg])
    st.dataframe(result, use_container_width=True)
    st.download_button("Download prediction CSV", result.to_csv(index=False).encode("utf-8"), "sms_prediction.csv", "text/csv")
upload=st.file_uploader("Batch CSV with a text column", type=["csv"])
if upload is not None:
    df=pd.read_csv(upload)
    col=st.selectbox("Text column", list(df.columns))
    if st.button("Run batch"):
        out=predict(system, df[col].astype(str).tolist())
        st.dataframe(out.head(100), use_container_width=True)
        st.download_button("Download batch CSV", out.to_csv(index=False).encode("utf-8"), "sms_batch_predictions.csv", "text/csv")
