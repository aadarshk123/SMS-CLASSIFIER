import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

ps=PorterStemmer()

def textTransform(txt):
    txt=txt.lower()
    txt=nltk.word_tokenize(txt)
    
    y=[]
    for i in  txt:
        if(i.isalnum()):
            y.append(i)
    
    txt=y[0:len(y)]
    y.clear()
    
    for i in txt:
        if(i not in stopwords.words('english') and i not in string.punctuation):
            y.append(i)
            
    txt=y[0:len(y)]
    y.clear()
    
    for i in txt:
        y.append(ps.stem(i))
    
    return " ".join(y)


tfidf=pickle.load(open('Vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email Spam/Ham Classifier")
input_sms=st.text_area("Enter your Email :: ")

if st.button('predict'):
    transformed_sms=textTransform(input_sms)
    vector_input=tfidf.transform([transformed_sms])
    result=model.predict(vector_input[0])

    if(result==1):
        st.header("Spam")
    else:
        st.header("Not Spam")