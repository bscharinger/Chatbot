import json
import pickle
import nltk

def load_intents(intent_path):
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


