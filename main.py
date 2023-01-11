import json
import pickle
import nltk
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models import model
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

keras.backend.clear_session()

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()




nltk.download('all')
intent_path = "data/intents.json"

words = []
classes = []
documents = []
lemmatizer = nltk.stem.WordNetLemmatizer()
ignore = ['!', '?', ',', '.']

intents = json.loads(open(intent_path).read())

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
pickle.dump(words, open('data/words.pkl','wb'))
pickle.dump(classes, open('data/classes.pkl', 'wb'))

training = []
validation = []

output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])



model = model.get_model((len(train_x[0]),), len(train_y[0]))
a = model.summary(line_length=150)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=0.01), loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(x=np.array(train_x), y=np.array(train_y), batch_size=10, verbose=1, shuffle=True, epochs=3000)
model.save("chatbot_model2")