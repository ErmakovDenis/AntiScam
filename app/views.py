from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
from collections import Counter
from math import sqrt
import requests
from bs4 import BeautifulSoup
from queue import Queue
from urllib.request import urlopen
from urllib.parse import urlparse, parse_qsl, unquote_plus
import pandas as pd
from math import sqrt
from django.views.decorators.csrf import requires_csrf_token
from django.views.decorators.csrf import csrf_exempt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keras.api.keras.preprocessing import image
from keras.applications.convnext import preprocess_input
from keras.datasets import cifar10
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint
from PIL import Image

import urlextract


punctuations_marks = ['.', ':', ',', '!', ';', '\'', '\"', '(', ')']


def scrapping(url_page):
    html_code_of_page = BeautifulSoup(urlopen(url_page).read(), 'html.parser')  # html код страницы
    text_from_page = html_code_of_page.get_text()  # текст страницы
    html_text_img_from_page = []

    parts = {
        'scheme': Url(url_page).parts.scheme,
        'netloc': Url(url_page).parts.netloc,
        'path': Url(url_page).parts.path,
        'params': Url(url_page).parts.params,
        'query': Url(url_page).parts.query,
        'fragment': Url(url_page).parts.fragment}

    list_extensions = ['raw', 'jpeg', 'jpg', 'tif', 'psd', 'bmp', 'gif', 'png', 'jp2', 'ico']
    tags_with_pictures = html_code_of_page.find_all(src=True) + html_code_of_page.find_all(href=True)
    list_attrs_src = []  # список картинок с страницы
    for tag_with_picture in tags_with_pictures:
        value_of_src_attribute = tag_with_picture.attrs.get("src")
        if value_of_src_attribute is not None and \
                value_of_src_attribute[value_of_src_attribute.rfind('.') + 1:] in list_extensions:
            if value_of_src_attribute.startswith('http://www.') or value_of_src_attribute.startswith('https://www.'):
                url_image = 'http://{}'.format(value_of_src_attribute[11:])
            elif value_of_src_attribute.startswith('http://') or value_of_src_attribute.startswith('https://'):
                url_image = value_of_src_attribute
            elif value_of_src_attribute.startswith('//'):
                url_image = 'https:{}'.format(value_of_src_attribute)
            elif value_of_src_attribute.startswith('www.'):
                url_image = 'http://{}'.format(value_of_src_attribute[4:])
            else:
                url_image = '{}{}'.format(parts['scheme'] + '://' + parts['netloc'], value_of_src_attribute)
            list_attrs_src.append(url_image)
        else:
            value_of_href_attribute = tag_with_picture.attrs.get("href")
            if value_of_href_attribute is not None and \
                    value_of_href_attribute[value_of_href_attribute.rfind('.') + 1:] in list_extensions:
                if value_of_href_attribute.startswith('http://www.') or value_of_href_attribute.startswith(
                        'https://www.'):
                    url_image = 'http://{}'.format(value_of_href_attribute[11:])
                elif value_of_href_attribute.startswith('http://') or value_of_href_attribute.startswith('https://'):
                    url_image = value_of_href_attribute
                elif value_of_href_attribute.startswith('//'):
                    url_image = 'https:{}'.format(value_of_href_attribute)
                elif value_of_href_attribute.startswith('www.'):
                    url_image = 'http://{}'.format(value_of_href_attribute[4:])
                else:
                    url_image = '{}{}'.format(parts['scheme'] + '://' + parts['netloc'], value_of_href_attribute)
                list_attrs_src.append(url_image)

    html_text_img_from_page.append(html_code_of_page)
    html_text_img_from_page.append(text_from_page)
    html_text_img_from_page.append(list_attrs_src)
    if 'None' in html_text_img_from_page:
        html_text_img_from_page.remove('None')
    return html_text_img_from_page


def crawler(domain_of_main_page, queue_of_all_pages):
    page_without_protocol = domain_of_main_page.replace('https://', '')
    all_pages_of_site = set()
    filter_of_extra_links = {'#', 'search', 'javascript', 'system', 'default', 'None', 'viewBid', '.jpg', 'uploads'}
    global links_to_all_pages
    links_to_all_pages = []  # ссылки на страницы сайта
    while len(links_to_all_pages) < 10:

        if queue_of_all_pages.qsize() == 0:
            break

        url_of_page = queue_of_all_pages.get()
        all_pages_of_site.add(url_of_page)
        response_of_call_of_page = requests.get(url_of_page)
        response_of_call_of_page.raise_for_status()
        links_to_all_pages.append(url_of_page)

        html_of_page = BeautifulSoup(response_of_call_of_page.content, 'lxml')

        for tag in html_of_page.find_all('a'):
            value_of_attribute = tag.get('href')

            if page_without_protocol not in str(value_of_attribute):
                continue

            if any(part_of_html in str(value_of_attribute) for part_of_html in filter_of_extra_links):
                continue

            queue_of_all_pages.put(value_of_attribute)


@csrf_exempt
def getting_data_from_site(domain_of_main_page):
    html_text_img_of_all_pages = []  # данные со всех страниц сайта
    queue_of_all_pages = Queue()
    queue_of_all_pages.put(domain_of_main_page)

    try:
        crawler(domain_of_main_page, queue_of_all_pages)
    except requests.HTTPError as error:
        error

    for new_link in links_to_all_pages:
        html_text_img_of_all_pages.append(scrapping(new_link))

    return html_text_img_of_all_pages

#------------------------------------------------------------------------------------------------------------

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
    x = UpSampling2D((4, 4))(x)

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
d_autoencoder.load_weights("/home/ermakov/PycharmProjects/AntiScamSait/venv/cifar100_weightsnew.hdf5")


# Функция подготовки изображения для обработки
def get_img(img):
    x = image.img_to_array(img)
    new_img = np.reshape(x, (32, 32, 3))
    new_img = np.expand_dims(new_img, axis=0)
    new_img = preprocess_input(new_img)
    return new_img


def get_pca_metrics(img):
    features = [str(x + 1) for x in range(8)]  # Названия колонок для датафрейма

    # Перевод numpy массива в стандартный
    array = []
    for i in range(len(img[0])):
        for e in img[0][i]:
            array += [[e[j] for j in range(len(e))]]
    new_arr = [[el for el in array[0]], [el for el in array[1]], [el for el in array[2]], [el for el in array[3]]]

    # Создание pandas dataframe
    df = pd.DataFrame(data=new_arr, columns=list('12345678'))
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    # Стандартизация данных и переход к 2 числам
    pca = PCA(n_components=2)
    principal_comps = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_comps, columns=['1', '2'])
    return principal_df


def square_rooted(num):
    return round(sqrt(sum([a * a for a in num])), 3)


# Функция сравнения косинусного расстояния между 2 векторами
def get_cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)


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


# Функция для получения коэфицента похожести 2 изображений
def compare_images(img1, img2):
    image1 = get_img(img1)
    image2 = get_img(img2)

    encoded_img1 = d_encoder.predict(image1)
    encoded_img2 = d_encoder.predict(image2)

    arr = get_pca_metrics(encoded_img1).to_dict()
    arr2 = get_pca_metrics(encoded_img2).to_dict()

    comparer = 0
    # Среднее арифмитическое 4 метрик
    for i in range(4):
        first_vector = get_vector(arr, i)
        second_vector = get_vector(arr2, i)
        comparer += abs(get_cosine_similarity(first_vector, second_vector))
    return comparer / 4  # >= metrica


# Функция открытия изображения по ссылке или пути в файловой системе
def open_img(url_path):
    extractor = urlextract.URLExtract()
    urls = extractor.find_urls(url_path)

    if len(urls) == 0:
        img = Image.open(url_path)
    else:
        img = Image.open(urlopen(url_path)).convert("RGB")

    r, g, b = img.split()
    r = r.point(lambda i: i * 1.2)
    g = g.point(lambda i: i * 0.9)
    res_img = Image.merge('RGB', (r, g, b))

    return res_img.resize((32, 32))


# Функция для нахождения ключа по значению в словаре
def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k


# Функция для нахождения максимально похожего изображения из 2 массива
def get_picture_similarity(pic_arr):
    result_dict = []
    for url_dict in pic_arr:
        maximum = max(url_dict.values())
        best_sim = get_key(url_dict, maximum)
        result_dict += [best_sim + ":  --- " + str(maximum)]

    return result_dict


def get_similarities(img_arr1, img_arr2):
    container = []
    for url1 in img_arr1:
        img1 = open_img(url1)
        pred = {}
        for url2 in img_arr2:
            img2 = open_img(url2)
            pred[str(url1) + " - " + str(url2)] = compare_images(img1, img2)
        container.append(pred)
    return container


# Функция для сортировки словаря (прямая, обратная)
def sort_dict(dictionary, param):
    sorted_dict = {}
    sorted_values = sorted(dictionary.values())

    for i in sorted_values:
        for k in dictionary.keys():
            if dictionary[k] == i:
                sorted_dict[k] = dictionary[k]

    if param == "direct":
        return sorted_dict
    elif param == "reverse":
        copied_dict = {}
        for key in reversed(sorted_dict.copy()):
            copied_dict[key] = sorted_dict.get(key)
        return copied_dict


#------------------------------------------------------------------------------------------------------------
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

    html_text_img_of_all_pages1 = getting_data_from_site(first_url)
    html_text_img_of_all_pages2 = getting_data_from_site(second_url)

    # if len(html_text_img_of_all_pages1) < 2 or len(html_text_img_of_all_pages2) < 2:
    #     return render(request, "error.html")
    # else :

    html1, text1, img1 = html_text_img_of_all_pages1[0][0], html_text_img_of_all_pages1[0][1], \
                             html_text_img_of_all_pages1[0][2]
    html2, text2, img2 = html_text_img_of_all_pages2[0][0], html_text_img_of_all_pages2[0][1], \
                             html_text_img_of_all_pages2[0][2]



    texts = [text1, text2]
    if len(texts) <= 1 :
        sim_matrix_text = [["нет текста", "нет текста"], ["нет текста", "нет текста"]]
    else:
        sim_matrix_text = get_text_similarity(texts, method="cosine")

    res_html = [str(html1) , str(html2)]
    if len(res_html) <= 1:
        sim_matrix_html = [["нет хтмл", "нет хтмл"], ["нет хтмл", "нет хтмл"]]
    else:
        sim_matrix_html = get_text_similarity(res_html, method="cosine")

    res = get_similarities(img1, img2)
    res_img = get_picture_similarity(res)


    if len(res_img) <= 0:
        res_img = "Нет картинок на одном из сайтов"

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

    data = {"text_sim": sim_matrix_text[0][1], "html_sim" : sim_matrix_html[0][1], "parts": arr, "img": res_img}

    return render(request, "ez.html", context=data)



# https://github.com/
# https://openedu.ru/
# https://www.hellomonday.com/

