
# coding: utf-8

# In[1]:

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from scipy import sparse
from random import randint
from collections import defaultdict

import collections
import math
import random
import numpy as np
import tensorflow as tf
import scipy as sc


# # LSA
# Для выполнения первого задания воспользуемся библиотечными функциями. Для парсинга википедии(у меня ее simple-вариант, около 80К статей после парсинга) я использовал фукнции библиотеки gensim, но на втором питоне (с третьим у меня не получилось с первого раза, и я не стал зацикливаться на этом), поэтому ячейка не инициализирована.
# , Можно было выкидывать из итогового текстового файла т.н. стоп-слова, которые не несут особого смысла и выкидывание их из контекстов только улучшает качество этих самых контестов, но я поступил по-другому и просто выкинул слишком редкие (встречаются менее чем в 10 статьях) и слишком частые (встречаются более чем в 1/5 статей) слова. Может показаться, что граница слишком частых достаточно слишком высока, но мне не пришли в голову какие-то слова, смысл которых нас интересуют, которые могут встречаться так часто (ну, я немого сожалел о выкидывании слова new, но жизнь продолжается)
# 
# Выкидывая редкие же слова мы избавились от откровенного мусора, типа транскрипций слов на их оригинальном языке, но и, возможно, повыкидывали какие-то нам интересные специфичные слова, типа CEO компаний

# In[ ]:

from gensim.corpora import WikiCorpus
import logging
import nltk
from nltk.corpus import stopwords

def parse_wiki(wiki_bz_file):
    output = open('./wiki_text_dump_4.txt', 'w')
    wiki = WikiCorpus(wiki_bz_file, lemmatize=False)
    wiki.dictionary.filter_extremes(10, 0.2, 60000)
    voc = set(wiki.dictionary.values())
    for text in wiki.get_texts():
        text = [w for w in text if w in voc]
        output.write(" ".join(text) + "\n")
    output.close()
    return

parse_wiki("./simplewiki-20170201-pages-articles.xml.bz2")


# In[8]:

dump = open('wiki_text_dump.txt', 'r')


# Пользуемся библиотечными функциями построения модели мешка слов с уже использованием метрики tf-idf, заодно бесплатно получаем словари (индекс-слово и слово-индекс). Я выбрал tf-idf метрку, т.к. мне кажется, что из предложенных в ней содержится максимум информации и расстояния между различными документами и словами точнее

# In[9]:

LSA_DIMS = 128


# In[10]:

tfidf_maker = TfidfVectorizer(use_idf=True)
tfidf_matrix = tfidf_maker.fit_transform(dump)


# In[11]:

VOC_SIZE = tfidf_matrix.shape[1]
direct_dict = tfidf_maker.vocabulary_
inverted_dict= dict(zip(tfidf_maker.vocabulary_.values(), tfidf_maker.vocabulary_.keys()))


# У нас остается 49263 слов.
# 
# Транспонируем матрицу чтобы получить в качестве первого аргумента компактные представления слов, а не документов

# In[6]:

tfidf_matrix = tfidf_matrix.transpose()
svd_maker = TruncatedSVD(LSA_DIMS)
lsa_embedding = svd_maker.fit_transform(tfidf_matrix)
np.save('lsa1', lsa_embedding)


# In[3]:

lsa_embedding = np.load('lsa1.npy')


# Для поиска ближайшего соседа воспользуемся scipy.spatial.distance.cdist(X, Y, metric), вычисляющей расстояния между парами строк матриц X и Y (у нас Y это один вектор)

# In[6]:

def find_k_nearest(word, embedding_matrix, k, metric):
    distances = sc.spatial.distance.cdist(embedding_matrix, 
                                embedding_matrix[direct_dict[word]].reshape(1, embedding_matrix.shape[1]),
                                metric)
    distances = distances.flatten()
    order = np.argsort(distances)
    for index in order[:k]:
        print(inverted_dict[index], distances[index])


# In[5]:

def analogy_k_nearest(word1, word2, word3, embedding_matrix, k, metric):
    word1_vector = embedding_matrix[direct_dict[word1]]
    word2_vector = embedding_matrix[direct_dict[word2]]
    word3_vector = embedding_matrix[direct_dict[word3]]
    new_word_vector = word1_vector - word2_vector + word3_vector
    distances = sc.spatial.distance.cdist(embedding_matrix, 
                                new_word_vector.reshape(1, embedding_matrix.shape[1]),
                                metric)
    distances = distances.flatten()
    order = np.argsort(distances)
    for index in order[:k]:
        print(inverted_dict[index], distances[index])


# Попробуем косинусную и евклидову метрики:

# In[44]:

find_k_nearest('moscow', lsa_embedding, 10, 'euclidean')


# In[45]:

find_k_nearest('moscow', lsa_embedding, 10, 'cosine')


# Здесь, например, евклидово расстояние мне показалось точнее косинусного. В следующих примерах они почти не отличаются, поэтому особо не дублирую.
# 
# Попробуем несколько фруктов:

# In[50]:

find_k_nearest('lemon', lsa_embedding, 10, 'euclidean')


# In[56]:

find_k_nearest('apple', lsa_embedding, 10, 'cosine')


# Месяца:

# In[58]:

find_k_nearest('september', lsa_embedding, 10, 'euclidean')


# Имена:

# In[59]:

find_k_nearest('john', lsa_embedding, 10, 'euclidean')


# In[128]:

find_k_nearest('nikolay', lsa_embedding, 10, 'euclidean')


# Цвета:

# In[66]:

find_k_nearest('yellow', lsa_embedding, 10, 'euclidean')


# Репперская тусовка:

# In[74]:

find_k_nearest('tupac', lsa_embedding, 10, 'cosine')


# Популярные поисковые запросы:

# In[80]:

find_k_nearest('anime', lsa_embedding, 10, 'cosine')


# Черепашки-ниндзя:

# In[95]:

find_k_nearest('raphael', lsa_embedding, 10, 'cosine')


# In[99]:

find_k_nearest('toyota', lsa_embedding, 10, 'euclidean')


# In[120]:

find_k_nearest('jupiter', lsa_embedding, 10, 'euclidean')


# In[125]:

find_k_nearest('frog', lsa_embedding, 10, 'euclidean')


# In[141]:

find_k_nearest('detective', lsa_embedding, 10, 'euclidean')


# In[124]:

find_k_nearest('frog', lsa_embedding, 10, 'cosine')


# На лягушках(да и в большинстве проверенных мною примеров) мне больше нравится косинусная мера, выдавшая различные виды лягушек, а не общие слова, которые можно встретить в статье про лягушек.
# 
# Подробнее метрики и варианты представления сравним позже

# В качестве еще одной меры, которую стоит попробовать, можно использовать Манхеттенское растояние

# In[12]:

find_k_nearest('frog', lsa_embedding, 10, 'cityblock')


# In[13]:

find_k_nearest('moscow', lsa_embedding, 10, 'cityblock')


# In[14]:

find_k_nearest('raphael', lsa_embedding, 10, 'cityblock')


# In[15]:

find_k_nearest('yellow', lsa_embedding, 10, 'cityblock')


# Достаточно качественная метрика, в какой-то ситуации она может и покажет себя, но авторами рассматриваемых моделей она не используется

# # Word2vec
# 
# Для начала представим наш дамп в виде листа индексов слов (ячейку с открытием файла дампа нужно перезапустить) 

# In[11]:

data = []
for string in dump:
    for word in string.split():
        if word in direct_dict.keys():
            data.append(direct_dict[word])


# В качестве размерности векторного представления возьмем 200 (из рекомендуемого в статье промежутка 200-300); 5 негативных примеров для каждой пары слов, как рекомендуется в статье; размер окна 4, размер батча 200 и количество использований одного слова 8, чтобы в батч попадали как раз всевозможные пары слово-контекст
# 
# Реализация подсмотрена на просторах [гитхаба](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
# 
# Количество слов в data 13342413, количество шагов 10000000, за один шаг мы обрабатываем 16 слов(показано позже), получается мы делаем 12 проходов по тексту (эпох)
# 
# Сложность при использования негатив сэмплинга это $embedding\_size \times NEG\_SMPLS \times num\_steps$
# 
# Первые два множителя отвечают за количество обновляемых за один шаг параметров матрицы, третий - за количество этих самых шагов. В общем случае, конечно, num_steps выражается через другие параметры (как сделано у меня в предыдущем абзаце), но в вопросе использования разных апроксимаций активации нас скорее всего интересует именно первые два множителя, т.к. в наивной реализации они в итоге дают размер матрицы $embedding\_size \times VOC\_SIZE$, что очень затратно

# In[194]:

batch_size = 200
embedding_size = 200
skip_window = 4
num_skips = 8
NEG_SMPLS = 5
num_steps = 10000000
w2v_graph = tf.Graph()
data_index = 0


# In[195]:

len(data)


# In[196]:

def get_batch(batch_size, num_skips, skip_window):
    global data_index
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


# Так выглядит наш батч на куске текста:
# > out four months days **june september november later year april starts same day week july every year starts same day** january leap years april
# 
# Жирным выделены слова, являющиеся центром окна в какой-то момент

# In[49]:

batch, labels = get_batch(batch_size=200, num_skips=8, skip_window=4)
for i in range(batch_size):
  print(batch[i], inverted_dict[batch[i]],
        '->', labels[i, 0], inverted_dict[labels[i, 0]])


# In[199]:

with w2v_graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    embeddings = tf.Variable(tf.random_uniform([VOC_SIZE, embedding_size], -1.0, 1.0) / float(embedding_size + 1))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    nce_weights = tf.Variable(tf.truncated_normal([VOC_SIZE, embedding_size],
            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([VOC_SIZE]))

    loss = tf.reduce_mean( tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=NEG_SMPLS,
                     num_classes=VOC_SIZE))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    init = tf.global_variables_initializer()


# In[ ]:

with tf.Session(graph=w2v_graph) as session:
    init.run()
    for step in range(num_steps):
        batch_inputs, batch_labels = get_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    w2v_embedding = embeddings.eval()


# In[ ]:

np.save('w2v2', w2v_embedding)


# In[8]:

w2v_embedding = np.load('w2v2.npy')


# Обучается (по сравнению с glove) долго, хотя negative samples у нас 5, и из-за большого количества мусора получилось не очень хорошо:

# In[70]:

analogy_k_nearest('king', 'man', 'women', lsa_embedding, 10, 'cosine')


# Хммм, надо переделать

# # GLOVE
# 
# Используем размерность векторного представления и ширину окна такие же, что и в word2vec. Счетчик совместных вхождений у нас будет с долями - чем дальше находятся слова, тем меньше увеличивается счетчик по формуле $^1/_{ dist}$, структура графа подсмотрена на [гитхабе](https://github.com/GradySimon/tensorflow-glove/blob/master/tf_glove.py)

# In[14]:

cont_width = 4
embedding_size = 200


# In[14]:

batch_size = 100


# In[15]:

X = sparse.lil_matrix((VOC_SIZE, VOC_SIZE), dtype=np.float64)
for i in range(len(data)):
    for j in range(max(0, i - cont_width), i):
        X[data[i], data[j]] += 1.0 / (i - j)
    for j in range(min((i + 1), len(data)), min((i + 1 + cont_width), len(data))):
        X[data[i], data[j]] += 1.0 / (j - i)
X = sparse.coo_matrix(X)


# In[16]:

GAMMA = 0.05


# In[19]:

glove_graph = tf.Graph()
with glove_graph.as_default():

    focal_input = tf.placeholder(tf.int32, shape=[batch_size],
                                        name="focal_words")
    context_input = tf.placeholder(tf.int32, shape=[batch_size],
                                          name="context_words")
    cooccurrence_count = tf.placeholder(tf.float32, shape=[batch_size],
                                               name="cooccurrence_count")

    focal_embeddings = tf.Variable(
        tf.random_uniform([VOC_SIZE, embedding_size], 1.0, -1.0) / float(embedding_size + 1))
    context_embeddings = tf.Variable(
        tf.random_uniform([VOC_SIZE, embedding_size], 1.0, -1.0) / float(embedding_size + 1))

    focal_biases = tf.Variable(tf.random_uniform([VOC_SIZE], 1.0, -1.0) / float(embedding_size + 1))
    context_biases = tf.Variable(tf.random_uniform([VOC_SIZE], 1.0, -1.0) / float(embedding_size + 1))

    focal_embedding = tf.nn.embedding_lookup([focal_embeddings], focal_input)
    context_embedding = tf.nn.embedding_lookup([context_embeddings], context_input)
    focal_bias = tf.nn.embedding_lookup([focal_biases], focal_input)
    context_bias = tf.nn.embedding_lookup([context_biases], context_input)

    f = tf.minimum(1.0, tf.pow(tf.div(cooccurrence_count, 400),0.75))

    embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

    log_cooccurrences = tf.log(tf.to_float(cooccurrence_count))

    distance_expr = tf.square(tf.add_n([
        embedding_product,
        focal_bias,
        context_bias,
        tf.negative(log_cooccurrences)]))

    single_losses = tf.multiply(f, distance_expr)
    total_loss = tf.reduce_sum(single_losses)
    optimizer = tf.train.AdagradOptimizer(GAMMA).minimize(total_loss)

    combined_embeddings = tf.add(focal_embeddings, context_embeddings,
                                        name="combined_embeddings")


# In[20]:

num_epochs = 12


# In[21]:

with tf.Session(graph=glove_graph) as session:
    tf.initialize_all_variables().run()
    for epoch in range(num_epochs):
        step = 0
        i_s = []
        j_s = []
        counts = []
        for (i, j, X_ij) in zip(X.row, X.col, X.data):
            if len(j_s) == batch_size:
                feed_dict = {focal_input: i_s, context_input: j_s, cooccurrence_count: counts}
                session.run([optimizer], feed_dict=feed_dict)
                del i_s[:], j_s[:], counts[:]
            else:
                i_s.append(i)
                j_s.append(j)
                counts.append(X_ij)
            step += 1
    glove_embedding = combined_embeddings.eval()


# In[27]:

np.save('glove_f', glove_embedding)


# In[12]:

glove_embedding = np.load('glove_f.npy')


# # Сравнение моделей

# Перепишем функцию поиска ближайшего слова так, чтобы она не выводила результат, а просто возвращала его.

# In[10]:

def analogy_k_nearest_noprint(word1, word2, word3, embedding_matrix, k, metric):
    word1_vector = embedding_matrix[direct_dict[word1]]
    word2_vector = embedding_matrix[direct_dict[word2]]
    word3_vector = embedding_matrix[direct_dict[word3]]
    new_word_vector = word1_vector - word2_vector + word3_vector
    distances = sc.spatial.distance.cdist(embedding_matrix, 
                                new_word_vector.reshape(1, embedding_matrix.shape[1]),
                                metric)
    distances = distances.flatten()
    order = np.argsort(distances)
    ret = []
    for i in order[:k]:
        ret.append(inverted_dict[i])
    return ret


# Авторы оригинальный статьи о w2v составили большой пак вопросов на поиск аналогии, мы случайным образом возьмем 1/10 этого пака и сравним наши модели
# 
# Аналогия считается угаданной, если слово-ответ есть в топ-10 ближайших векторов

# In[ ]:

questions = open('questions-words.txt', 'r')
questions_2 = open('quest_list.txt', 'w')
for question in questions:
    if (random.random() < 0.1):
        print(question.lower(), file=questions_2, end='')
questions.close()
questions_2.close()


# In[13]:

questions = open('quest_list.txt', 'r')
right = 0
total = 0
lsa_cos_score = 0
lsa_euc_score = 0
w2v_cos_score = 0
w2v_euc_score = 0
glv_cos_score = 0
glv_euc_score = 0
for question in questions:
    words = question.split()
    if words[0] not in direct_dict.keys() or words[1] not in direct_dict.keys() or words[2] not in direct_dict.keys() or words[3] not in direct_dict.keys():
        continue
    if total % 15 == 0:
        print("                                                                     |lsa,c|lsa,e|w2v,c|w2v,e|glv,c|glv,e|")
    total += 1
    print("%4d |%14s -%14s +%14s =%14s |" % (total, words[0], words[1], words[3], words[2]), end='')
    if (words[2] in analogy_k_nearest_noprint(words[0], words[1], words[3], lsa_embedding, 10, 'cosine')):
        print("  \033[1;32;0m✓\033[0m  |", end='')
        lsa_cos_score += 1
    else:
        print("  \033[1;31;0m×\033[0m  |", end='')
    
    if (words[2] in analogy_k_nearest_noprint(words[0], words[1], words[3], lsa_embedding, 10, 'euclidean')):
        print("  \033[1;32;0m✓\033[0m  |", end='')
        lsa_euc_score += 1
    else:
        print("  \033[1;31;0m×\033[0m  |", end='')
    
    if (words[2] in analogy_k_nearest_noprint(words[0], words[1], words[3], w2v_embedding, 10, 'cosine')):
        print("  \033[1;32;0m✓\033[0m  |", end='')
        w2v_cos_score += 1
    else:
        print("  \033[1;31;0m×\033[0m  |", end='')
    
    if (words[2] in analogy_k_nearest_noprint(words[0], words[1], words[3], w2v_embedding, 10, 'euclidean')):
        print("  \033[1;32;0m✓\033[0m  |", end='')
        w2v_euc_score += 1
    else:
        print("  \033[1;31;0m×\033[0m  |", end='')
    
    if (words[2] in analogy_k_nearest_noprint(words[0], words[1], words[3], glove_embedding, 10, 'cosine')):
        print("  \033[1;32;0m✓\033[0m  |", end='')
        glv_cos_score += 1
    else:
        print("  \033[1;31;0m×\033[0m  |", end='')
    
    if (words[2] in analogy_k_nearest_noprint(words[0], words[1], words[3], glove_embedding, 10, 'euclidean')):
        print("  \033[1;32;0m✓\033[0m  |")
        glv_euc_score += 1
    else:
        print("  \033[1;31;0m×\033[0m  |")
print("                                                                     |%5d|%5d|%5d|%5d|%5d|%5d|" % (lsa_cos_score,
                                                                                                          lsa_euc_score,
                                                                                                          w2v_cos_score,
                                                                                                          w2v_euc_score,
                                                                                                          glv_cos_score,
                                                                                                          glv_euc_score))
questions.close()


# # t-SNE

# В качестве словаря будем использовать слова, из которых состоят подготовленные вопросы (849 слов), а изображаться на графике из них будут только несколько выбранные мной слов.

# In[26]:

train_matrix = np.empty((0,embedding_size))
already_added = []
questions = open('quest_list.txt', 'r')
for question in questions:
    words = question.split()
    for word in words:
        if word in direct_dict.keys() and word not in already_added:
            train_matrix = np.append(train_matrix, [glove_embedding[direct_dict[word]]], axis=0)
            already_added.append(word)
questions.close()


# In[27]:

from sklearn.manifold import TSNE
tsne = TSNE(n_components=3, init='pca')
low_dim_embs = tsne.fit_transform(train_matrix)


# In[61]:

get_ipython().magic('matplotlib notebook')
from mpl_toolkits.mplot3d import Axes3D
import pylab
del ax, fig
fig = pylab.figure(figsize=(10,7))
ax = Axes3D(fig)
to_tsne = open('to_tsne.txt', 'r')
for string in to_tsne:
    for word in string.split():
        already_added.index(word)
        x, y, z = low_dim_embs[already_added.index(word)]
        ax.scatter(x, y, z,c='b', s=5) 
        ax.text(x, y, z, word, size=10, zorder=1, color='k')
#ax.mouse_init()
fig.savefig('test')
ax.axis('off')
pylab.show()


# In[ ]:



