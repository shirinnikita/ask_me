from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from gru_inh import myGru
from keras import backend as K
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

max_c_len = 300
max_q_len = 7
max_ans_len = 3

app = Flask(__name__)

def custom_categorical_accuracy(y_true, y_pred):
    return K.min(K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx()), axis=-1)



@app.route('/')
def index():
    return render_template('index.html')

# Route that will process the AJAX request, sum up two
# integer numbers (defaulted to zero) and return the
# result as a proper JSON response (Content-Type, etc.)
@app.route('/_compute_answer')
def compute_answer():
    global model
    global new_voc
    global new_inv_voc
    context = request.args.get('context', '', type=str)
    print('CONTEXT: ', '<'*10, context, '>'*10)
    question = request.args.get('question', '', type=str)
    print('QUESTION: ', '<'*10, question, '>'*10)
    context = context.replace('.', ' . ').replace('\n', ' ')
    inp = [w for w in context.lower().split(' ')  if len(w) > 0]
    if '?' in question:
        question = question[:question.find('?')]
    q = [w for w in question.lower().split(' ') if len(w) > 0]

    inp_vector = []
    for w in inp:
        if w in new_voc:
            inp_vector.append(new_voc[w])
        else:
            return jsonify(result = w + ' not in vocabulary')

    quest_vector = []
    for w in q:
        if w in new_voc:
            quest_vector.append(new_voc[w])
        else:
            return jsonify(result = w + ' not in vocabulary')

    cur_context = pad_sequences([inp_vector], maxlen=max_c_len, dtype='int32',
                      padding='post', truncating='post', value=0)
    cur_question = pad_sequences([quest_vector], maxlen=max_q_len, dtype='int32',
                      padding='post', truncating='post', value=0)
    global graph
    with graph.as_default():
        prediction = model.predict([cur_context, cur_question])
    answer = []
    for i in range(max_ans_len):
        cur_word = np.argmax(prediction[0][i])
        if cur_word == 0:
            break
        answer.append(new_inv_voc[cur_word])
    print(answer)
    return jsonify(result = ', '.join(answer))

if __name__ == '__main__':
    model = load_model('jjjj', custom_objects={'myGru':myGru,
        'custom_categorical_accuracy':custom_categorical_accuracy})
    graph = tf.get_default_graph()
    new_voc = {}
    with open('vocabulary.pkl', 'rb') as f:
        new_voc = pickle.load(f)
    new_inv_voc = {v: k for k, v in new_voc.items()}
    app.run(debug=True)
