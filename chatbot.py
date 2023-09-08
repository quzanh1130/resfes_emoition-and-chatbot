import random
import json
import pickle
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt', quiet=True)
# nltk.download('wordnet', quiet=True)

class Chatbot():
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open('/home/mycar1130/quocanh/resfes/json/intents_eng.json').read())

        self.words = []
        self.classes = []
        self.documents = []
        self.ignoreLetters = ['?', '!', '.', ',']

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                wordList = nltk.word_tokenize(pattern)
                self.words.extend(wordList)
                self.documents.append((wordList, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(word) for word in self.words if word not in self.ignoreLetters]
        self.words = sorted(set(self.words))

        self.classes = sorted(set(self.classes))

        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))

        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))
        self.interpreter = tf.lite.Interpreter(model_path="/home/mycar1130/quocanh/resfes/model/chatbot_model_eng.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()




    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words] #jjj
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)

        # using tflite model
        self.input_shape = self.input_details[0]['shape']
        input_data = np.array(np.array([bow]), dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        res = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, intents_list):
        try:
            tag = intents_list[0]['intent']
            list_of_intents = self.intents['intents']
            for i in list_of_intents:
                if i['tag']  == tag:
                    result = random.choice(i['responses'])
                    break
        except IndexError:
            result = "I don't understand!"
        return result