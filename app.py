from flask import Flask, request, render_template,session
from flask import jsonify
import easygui
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import tensorflow as tf
import numpy as np
from sklearn import metrics
import os
app = Flask(__name__)
app.config['SECRET_KEY'] =os.urandom(24)

tokenizer = Tokenizer()

model = load_model('model/best_model.h5',compile=False)
Confusion_matrix=[]
labels=[]
global graph
graph = tf.get_default_graph()
	
MAX_SEQUENCE_LENGTH=463
CARRAIGE_REPLACE_SPACE = re.compile('[\r]')
NEWLINE_REPLACE_SPACE = re.compile('[\n]')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;.]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
URL = re.compile("ftp://[0-9]+.[0-9]+.[0-9]+.[0-9]+/")
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = CARRAIGE_REPLACE_SPACE.sub(' ', text)
    text = NEWLINE_REPLACE_SPACE.sub(' ', text)
    text = URL.sub('', text)
    text = text.lower() 
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text




 
@app.route("/")
def root(): 
    print("page_loaded")
    
    app.config['SECRET_KEY'] =os.urandom(24)
    session.clear()
    return render_template('index.html',)

@app.route('/api', methods=["POST"])
def choose_csv():
    try:
	    
	    app.config['SECRET_KEY'] =os.urandom(24)
	    session.clear()
	    global Confusion_matrix
	    global labels	   
	    print("button_clicked")
	    response = request.get_json()
	    print(response)
	    filepath = easygui.fileopenbox(msg='Please Select the csv file',filetypes='*.csv',default='')
	    #print(filepath)
	    #filepath='/home/rithesh/Desktop/Deep_triage_app/Deep_triage_app/data/Hotkey_legacyAndUWP_Exported_050720-140818.csv'
	    df = pd.read_csv(filepath,usecols=['Component','Long Description','Short Description'],encoding='latin1') 
	    df=df.head(100)
	    print(df) 
	    df["Description"] = df["Short Description"] + " " + df["Long Description"]
	    df = df.reset_index(drop=True)
	    #df['Description'] = df['Description'].apply(clean_text)
	    tokenizer.fit_on_texts(df['Description'].values)
	    
	    labels=['HP 3D DriveGuard 6','HP Hotkey Support - CMIT','HP Hotkey Support - UWP','HP Wireless Button Driver']
	    labels.sort()
	    
	    X = tokenizer.texts_to_sequences(df['Description'].values)
	    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
	    pred_list=[]
	    print("preprocess Completed",X, X.shape,X[1].shape)
	    with graph.as_default():
	    	pred = model.predict(X)
	    for elem in pred:
	    	
	    	pred_list.append(labels[np.argmax(elem)])
	    
	    
	    
	    print(pred,pred_list)	
	    df['predicted']=pred_list
	    
	    df_list2 = []
	    df_list2.append(df[df['Component'] == df['predicted']])
	    print("--------Correctly Predicted Number = ",len(df_list2[0]))
	    df_list2.append(df[df['Component'] != df['predicted']])
	    df=pd.concat(df_list2)
	    
	    to_ui_list=[]
	    to_ui_list.append(df['Component'].tolist())
	    to_ui_list.append(df['Long Description'].tolist())
	    to_ui_list.append(df['Short Description'].tolist())
	    to_ui_list.append(df['predicted'].tolist())
	    
	    Confusion_matrix=metrics.confusion_matrix(to_ui_list[0],to_ui_list[3], labels)
	    print(Confusion_matrix)
    except Exception as e:
    	   to_ui_list=[]
    	   print(e)
    return jsonify(to_ui_list)
    
@app.route('/matrix', methods=["POST"])
def confusion_matrix():
	print("hai_matrix")
	print(labels)
	print(Confusion_matrix)
	return render_template('matrix.html',Confusion_matrix=Confusion_matrix , labels = labels)
    
if __name__ == "__main__":
    app.run(debug=True,threaded=True)
