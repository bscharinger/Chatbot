import nltk
import pickle
import numpy as np
from tensorflow import keras
import tensorflow as tf
import json
import random
import tkinter
from datetime import datetime
import requests
import python_weather

from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

lemmatizer = nltk.stem.WordNetLemmatizer()

# load model
model = keras.models.load_model('chatbot_model2')

# load words
intents = json.loads(open('data/intents.json').read())
words = pickle.load(open('data/words.pkl', 'rb'))
classes = pickle.load(open('data/classes.pkl', 'rb'))

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_words(sentence, words, show_details=True):
    sente_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for s in sente_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print('found in bag: %s' % word)
    return np.array(bag)

def predict_class(sentence, words, model):
    p = bag_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    err_thresh = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > err_thresh]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag']==tag):
            result = random.choice(i['responses'])
            break
    return result

def weather(city):
    city = city.replace(" ", "+")
    print(city)
    res = requests.get(f'https://www.google.com/search?q={city}&oq={city}&aqs=chrome.0. i635i39l2j0l4j46j690.6128j1j7&sourceid=chrome&ie=UTF-8', headers=headers)
    print("Searching...\n")
    print(res)
    soup = BeautifulSoup(res.text, 'html.parser')
    print(soup.select('#wob_loc'))
    location = soup.select('#wob_loc')[0].getText().strip()
    time = soup.select('#wob_dts')[0].getText().strip()
    info = soup.select('#wob_dc')[0].getText().strip()
    weather = soup.select('#wob_tm')[0].getText().strip()
    return location, time, info, weather+"°C"

root = tkinter.Tk()
root.title("Chatbot")
root.geometry("500x600")
root.resizable(width=False, height=False)
EntryBox = tkinter.Text(root, bd=0, bg="white", width=29, height=5, font="Arial")
ChatBox = tkinter.Text(root, bd=0, bg="white", height=8, width=50, font="Arial")
ChatBox.insert(tkinter.END, "Test")
scrollbar = tkinter.Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0",tkinter.END)
    print("message recieved")
    if msg != '':
        ChatBox.insert(tkinter.END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12))

        ints = predict_class(msg, words, model)
        res = get_response(ints, intents)
        print("response claculated")
        ChatBox.insert(tkinter.END, "Bot: " + res + '\n\n')
        print(res)
        if res == "Date and Time":
            now = datetime.now()
            ChatBox.insert(tkinter.END, "Bot: " + "Current date and time are: " +
                           now.strftime("%d/%m/%Y %H:%M:%S") +'\n\n')
        if msg[:7]=="google:":
            try:
                from googlesearch import search
            except ImportError:
                print("No module named 'google' found")
            query = msg.strip()[7:]
            for j in search(query, tld="co.in", num=10, stop=10, pause =2):
                ChatBox.insert(tkinter.END, "Bot: " + j + '\n')
        if msg[:8] == "weather:":
            city = msg.strip()[8:]
            print(city)
            loc, tim, inf, wea = weather(city)
            ChatBox.insert(tkinter.END, "Bot: " + "I cannot currently access any weather information, try again later")
        if msg[:5] == "news:":
            topic = msg[5:]
        ChatBox.yview(tkinter.END)

SendButton = tkinter.Button(root, font=("Verdana",12, 'bold'), text="Send", width=12, height=5, bd=0, bg="#f9a602",
                            activebackground="#3c9d9b", fg="#000000", command=send)

scrollbar.place(x=376,y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90, width=80)

root.mainloop()

