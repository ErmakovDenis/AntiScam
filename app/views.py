from django.shortcuts import render
from bs4 import BeautifulSoup
import os
from urllib.request import urlretrieve
from urllib.request import urlopen
from django.http import HttpResponse
from urllib.parse import urlparse, parse_qsl, unquote_plus
from django.template.response import TemplateResponse
import numpy as np
from math import *
import os
import requests
from nltk.corpus import stopwords
import h5py
from collections import Counter
import pandas as pd
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.api.keras.preprocessing import image
from keras.applications.convnext import preprocess_input
from keras.datasets import cifar10
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint


punctuations_marks = ['.', ':', ',', '!', ';', '\'', '\"', '(', ')']


# Функция генерации автокодировщика
def create_deep_conv_ae():
    input_img = Input(shape=(32, 32, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(3, (3, 3), padding='same')(x)

    # Модели
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder


# Генерируем модели
d_encoder, d_decoder, d_autoencoder = create_deep_conv_ae()
d_autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics='acc')
d_autoencoder.load_weights("/home/ermakov/PycharmProjects/AntiScamSait/venv/weights(cifar10).hdf5")  # Загрузка предварительно обученных весов


# Функция подготовки изображения для обработки
def get_img(path):
    img = image.load_img(path, target_size=(32, 32))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def start_training(name_of_weights_file):
    # Загружаем dataset cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
    x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

    weights_file = str(name_of_weights_file)  # Имя файла в который сохранются веса

    # Чекпоинты для сохранения в процессе обучения
    checkpoint = ModelCheckpoint(weights_file, monitor='acc', mode='max', save_best_only=True, verbose=1)

    d_autoencoder.fit(x_train, x_train, epochs=5, batch_size=128, callbacks=[checkpoint], shuffle=True,
                      validation_data=(x_test, x_test))


def get_pca_metrics(img):
    features = [str(x + 1) for x in range(8)]  # Названия колонок для датафрейма

    # Перевод numpy массива в стандартный
    array = []
    for i in range(len(img[0])):
        for e in img[0][i]:
            array += [[e[j] for j in range(len(e))]]

    # Создание pandas dataframe
    result = [[el for el in array[0]], [el for el in array[1]], [el for el in array[2]], [el for el in array[3]]]
    df = pd.DataFrame(data=result, columns=list('12345678'))
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    # Стандартизация данных и переход к 2 числам
    pca = PCA(n_components=2)
    principal_comps = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_comps, columns=['1', '2'])

    return principal_df


# Получение вектора из pd dataframe
def get_vector(dictionary, index):
    x, y = 0, 0
    for key, value in dictionary.items():
        if key == "1":
            for k, v in dictionary[key].items():
                if k == index:
                    x = v
        elif key == "2":
            for k, v in dictionary[key].items():
                if k == index:
                    y = v
    return [x, y]


def compare_images(img1path, img2path):
    image1 = get_img(str(img1path))
    image2 = get_img(str(img2path))

    encoded_img1 = d_encoder.predict(image1)
    encoded_img2 = d_encoder.predict(image2)

    arr = get_pca_metrics(encoded_img1).to_dict()
    arr2 = get_pca_metrics(encoded_img2).to_dict()

    comparer = 0

    # Среднее арифмитическое 4 метрик
    for i in range(4):
        comparer += abs(get_cosine_similarity(get_vector(arr, i), get_vector(arr2, i)))
    return str("Изображения похожи на " + str(comparer / 4 * 100) + " %")





def tokenize_sentences(text, stop_char="."):
    return [x.lower() + ' ' + stop_char + '' for x in text.split(stop_char) if x != ""]


def tokenize_words_from_sent(sentences, stop_words_list=None, punctuation=None):
    if punctuation is None:
        punctuation = []
    if stop_words_list is None:
        stop_words_list = []
    return [x.lower() for sent in sentences for x in sent.split() if
            (x != "" and x not in punctuation and x not in stop_words_list)]


def prepare_vocab(all_words):
    word_counts = Counter([word for text in all_words for word in text])
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return word_counts, sorted_vocab, int_to_vocab, vocab_to_int


def get_word_tf(sorted_vocab, vocab_to_int, words):
    word_tf = np.zeros(len(sorted_vocab), dtype=float)
    for word, count in Counter(words).items():
        word_tf[vocab_to_int[word]] = count / len(words)

    return word_tf


def get_word_idf(all_words, n_texts, sorted_vocab, vocab_to_int):
    word_idf = np.zeros(len(sorted_vocab), dtype=float)
    for word in sorted_vocab:
        n_docs = 0
        for doc in all_words:
            if word in doc:
                n_docs += 1
        word_idf[vocab_to_int[word]] = n_texts / n_docs

    return word_idf


def get_tfidf_vectors(word_tf, word_idf):
    return [tf * idf for tf, idf in zip(word_tf, word_idf)]


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)


def get_cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)


def get_text_similarity(texts, method="cosine", exclude_stopwords=False, exclude_punctuation=False):

    n_texts = len(texts)
    if n_texts < 2:
        return

    sim_matrix = [[0 if i != j else 1 for j in range(n_texts)] for i in range(n_texts)]
    n_similar = 0
    for i in range(n_texts):
        for j in range(i + 1, n_texts):
            if j < n_texts and texts[i] == texts[j]:
                sim_matrix[i][j] = 1
                n_similar += 1
            else:
                pass
    if n_similar == n_texts - 1:
        return sim_matrix

    tokenized_stop_words = []
    punctuations = punctuations_marks if exclude_punctuation else []
    all_words = [tokenize_words_from_sent(tokenize_sentences(text),
                                          tokenized_stop_words, punctuations) for text in texts]


    word_counts, sorted_vocab, int_to_vocab, vocab_to_int = prepare_vocab(all_words)


    word_tfs = [get_word_tf(sorted_vocab, vocab_to_int, words) for words in all_words]


    word_idf = get_word_idf(all_words, n_texts, sorted_vocab, vocab_to_int)


    tfidf_vec = [get_tfidf_vectors(word_tf, word_idf) for word_tf in word_tfs]


    if method == "cosine":
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                if j >= n_texts or sim_matrix[i][j] == 1:
                    break
                sim_matrix[i][j] = get_cosine_similarity(tfidf_vec[i], tfidf_vec[j])
        return sim_matrix


class Url(object):

    def __init__(self, url):
        parts = urlparse(url)
        _query = frozenset(parse_qsl(parts.query))
        _path = unquote_plus(parts.path)
        parts = parts._replace(query=_query, path=_path)
        self.parts = parts

    def __eq__(self, other):
        return self.parts == other.parts

    def __hash__(self):
        return hash(self.parts)





def index(request):
    return render(request, 'ez.html')


def main(request):
    first_url = request.POST.get("first_url")
    second_url = request.POST.get("second_url")

    texts = [first_url, second_url]
    sim_matrix = get_text_similarity(texts, method="cosine")

    result_img = compare_images("/home/ermakov/PycharmProjects/AntiScamSait/templates/img1.jpg",
                                "/home/ermakov/PycharmProjects/AntiScamSait/templates/img2.jpg")

    parts1 = {
        'scheme': Url(first_url).parts.scheme,
        'netloc': Url(first_url).parts.netloc,
        'path': Url(first_url).parts.path,
        'params': Url(first_url).parts.params,
        'query': Url(first_url).parts.query,
        'fragment': Url(first_url).parts.fragment,
    }
    parts2 = {
        'scheme': Url(second_url).parts.scheme,
        'netloc': Url(second_url).parts.netloc,
        'path': Url(second_url).parts.path,
        'params': Url(second_url).parts.params,
        'query': Url(second_url).parts.query,
        'fragment': Url(second_url).parts.fragment,
    }

    arr = []
    for i in parts1:
        if parts1[i] == parts2[i]:
            arr += [str(i) + " equal"]
        else:
            arr += [str(i) + " different " + str(parts1[i]) + " " + str(parts2[i])]

    data = {"text_sim": sim_matrix[0][1], "matrix_sim": sim_matrix, "parts": arr, "img": result_img}

    return render(request, "ez.html", context=data)



# https://github.com/
# https://openedu.ru/


