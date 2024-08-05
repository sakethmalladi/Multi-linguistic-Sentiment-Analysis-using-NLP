import tkinter as tk
from tkinter import *
from tkinter import ttk 

from textblob import TextBlob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tweepy 
import pandas as pd
from pandas import DataFrame

import re
import string
import inflect
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')

import numpy as np
from googletrans import Translator
from nltk.collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.svm import SVC as SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()
p = inflect.engine()  


window = Tk()
window.title("Sorting of Specific Tweets On Twitter")


width  = window.winfo_screenwidth()//2
height = window.winfo_screenheight()

window.geometry(f'{width}x{height}')

      

tab_control = ttk.Notebook(window)

tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)


tab_control.add(tab1,text = "Sentiment Analysis")
#tab_control.add(tab2,text = "Twitter Sentiment Analysis")
#tab_control.add(tab3,text = "Twitter Analysis Graph")
tab_control.add(tab4,text = "About")
tab_control.pack(expand=1,fill='both')


label1 = Label(tab1,text="Text Sentiment Analysis",padx=10,pady=10,anchor=N,font=22)
label1.grid(row=0,column=0)
#label2 = Label(tab2,text="Twitter Sentiment Analysis",padx=10,pady=10, anchor=CENTER,font=22)
#label2.grid(row=0,column=0)
#label3 = Label(tab3,text="Twitter Analysis Graph",padx=10,pady=10, anchor=CENTER,font=22)
#label3.grid(row=0,column=0)
label4 = Label(tab4,text="About Us",padx=10,pady=10, anchor=CENTER,font=22)
label4.grid(row=0,column=0)

#
#
#
#

#TAB 1 FUNCTIONS

def create_lexicon(pos, neg):
    lexicon = []
    for file_name in [pos, neg]:
        with open(file_name, 'r') as f:
            contents = f.read()
            for line in contents.split('\n'):
                data = line.strip('\n')
                if data:
                    all_words = word_tokenize(data)
                    lexicon += list(map((lambda x: x.lower()), all_words))
    lexicons = []
    for word in lexicon:
        if not word in stop_words:
            lexicons.append(word)
    word_counts = Counter(lexicons) 
    l2 = []
    for word in word_counts:
        if 4000 > word_counts[word]:
            l2.append(word)
    print(l2)
    return l2


def samplehandling(sample, lexicons, classification):
    featureset = []
    with open(sample, 'r', encoding="utf8") as f:
        contents = f.read()
        for line in contents.split('\n'):
            data = line.strip('\n')
            if data:
                all_words = word_tokenize(data)
                all_words = list(map((lambda x: x.lower()), all_words))
                all_words_new = []
                for word in all_words:
                    if not word in stop_words:
                        all_words_new.append(word)
                features = np.zeros(len(lexicons))
                for word in all_words_new:
                    if word in lexicons:
                        idx = lexicons.index(word)
                        features[idx] += 1
                features = list(features)
                featureset.append([features, classification])
    return featureset


def create_feature_set(pos, neg):
    featuresets = []
    lexicons = create_lexicon(pos, neg)
    featuresets += samplehandling(pos, lexicons, 1)
    featuresets += samplehandling(neg, lexicons, 0)
    return featuresets

def create_test_data_for_unigram(pos):
    lexicons = create_lexicon('p_e.txt', 'n_e.txt')
    translator = Translator()
    testset = []
    line = pos.strip('\n')
    line = translator.translate(line, dest="english").text
    featureset = np.zeros(len(lexicons))
    line = word_tokenize(line)
    words = list(set([w.lower() for w in line]))
    for w in lexicons:
        if w in words:
            idx = lexicons.index(w.lower())
            featureset[idx] += 1
    featureset = list(featureset)
    testset.append([featureset,1])
          
    return testset

def test_by_unigram(text_enter):
    testset = create_test_data_for_unigram(text_enter)
    testset = np.array(testset)
    test_x = list(testset[:,0])
    prediction = clf.predict(test_x)
    y_pred=list(np.array(prediction))
    print("Logistic Regression:")
    print(y_pred)
    return y_pred


def preprocessing_function(text_enter):
    temp = re.sub(r"#(\w+)", "", text_enter, flags=re.MULTILINE) #removing hashtags
    temp = re.sub(r"@(\w+)", "", temp, flags=re.MULTILINE) #removing mentions
    temp = re.sub(r"http\S+", "", temp, flags=re.MULTILINE) #removing urls
    temp = temp.lower() #converting into lowercase
    translator = str.maketrans('', '', string.punctuation) #removing punctuation
    temp = temp.translate(translator) 
    temp = " ".join(temp.split()) #removing whitespaces


    # split string into list of words 
    temp_str = temp.split() 
    #initialise empty list 
    new_string = [] 
    for word in temp_str: 
      # if word is a digit, convert the digit 
      # # to numbers and append into the new_string list 
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)
        # append the word as it is 
        else:
            new_string.append(word) 
        # join the words of new_string to form a string 
        
    temp_str = ' '.join(new_string)

    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    temp = emoji_pattern.sub(r'', temp_str)


    word_tokens = word_tokenize(temp)

    return word_tokens


def preprocess_fun():
    text_entered = str(data1_entered.get())
    filtered_words = preprocessing_function(text_entered)
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in filtered_words]
    
    result = '\nTokens : {}'.format(lemmas)
    tab1_display.insert(tk.END,result)
    tab1_display.grid()
    

def get_sentiment():
    text_entered = str(data1_entered.get())
    polarity=test_by_unigram(text_entered)
    result = '\n Polarity:{}\n'.format(polarity)
    tab1_display.insert(tk.END,result)
    tab1_display.grid()


def clear_text_entered1():
    entry1.delete(0,END)


def clear_result1():
    tab1_display.delete('1.0',END)
    tab1_display.grid_forget()


#TAB 1
featureset = create_feature_set('p_e.txt', 'n_e.txt')
featureset = np.array(featureset)
x = list(featureset[:, 0])
y = list(featureset[:, 1])
clf = LR()
clf.fit(x,y)

l1 = Label(tab1, text = "Enter Text to analyze" )
l1.grid(row=1,column=0)

data1_entered = StringVar()
entry1 = Entry(tab1,textvariable = data1_entered, width=100,bd=5)
entry1.grid(row=4,column=0,padx=20)


button1 = Button(tab1, text = "Tokenize",width = 15, bg='blue', fg ='#FFFFFF',justify = CENTER, command = preprocess_fun)
button1.grid(row=6,column=0,padx=10,pady=10)

button2 = Button(tab1, text = "Analyze",width = 15, bg='blue', fg ='#FFFFFF',justify = CENTER, command = get_sentiment)
button2.grid(row=7,column=0,padx=10,pady=10)

button3 = Button(tab1, text = "Reset",width = 15, bg='blue', fg ='#FFFFFF',justify = CENTER, command = clear_text_entered1)
button3.grid(row=8,column=0,padx=10,pady=10)

button4 = Button(tab1, text = "Clear Result",width = 15, bg='blue', fg ='#FFFFFF',justify = CENTER, command = clear_result1)
button4.grid(row=9,column=0,padx=10,pady=10)


tab1_display = Text(tab1,height=10,width=75,bd=5)
tab1_display.grid(row=11,column=0,columnspan=2, rowspan=2,padx=2,pady=2)
tab1_display.grid_forget()

#TAB4

l4 = Label(tab4, text = "Malladi Saketh (2451-17-733-104)",padx=10,pady=10, anchor=CENTER,font=20 )
l4.grid(row=2,column=1)
l4 = Label(tab4, text = "B.V.S.S Srikanth (2451-17-733-112)",padx=10,pady=10, anchor=CENTER,font=20 )
l4.grid(row=3,column=1)
l4 = Label(tab4, text = "Byreddy Rohith Reddy (2451-17-733-061)",padx=10,pady=10, anchor=CENTER,font=20 )
l4.grid(row=4,column=1)
l4 = Label(tab4, text = "Guide:",padx=10,pady=10, anchor=CENTER,font=20 )
l4.grid(row=5,column=1)
l4 = Label(tab4, text = "B.Venkataramana,Asst.Prof,CSED, MVSREC, Hyderabad ",padx=10,pady=10, anchor=CENTER,font=20 )
l4.grid(row=6,column=1)



window.mainloop()