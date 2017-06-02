## Картинки для архитектуры и инструкции будут как сделаю дизайн(но текущая версия уже лежит)
# Ask me
### Ширин Никита, 152 группа ПМИ ФКН ВШЭ
## Цель проекта
Проект заключается в знакомстве с методами машинного обучения и реализации модели нейросети ***dynamic memory network (DMN)***  из статьи [Ask me anything](https://arxiv.org/pdf/1506.07285.pdf), которая позволяет отвечать на заданный ей вопрос с учетом предшествующего контекста. Эта задача является очень актуальной, так как модель, способная естественным образом отвечать на челоческие вопросы, будет хорошим интерфейсом для взаимодействия человека и машины.

В качестве примера использования модель обучается на датасете  [bAbI tasks](https://research.fb.com/downloads/babi/), который содержит 20 различных типов вопросов определенной структуры.
## Известные решения
Авторы статьи сравнивают свои результаты обучения модели на bAbi tasks с результатами других алгоритмов, использующих нейросети, среди которых: [AMV-RNN](https://nlp.stanford.edu/pubs/SocherHuvalManningNg_EMNLP2012.pdf), [MemNN](https://arxiv.org/pdf/1503.08895.pdf) и [CT-LSTM](https://arxiv.org/pdf/1503.00075.pdf). Среди всех рассмотренных моделей, DMN показывает лучший результат.
## Архитектура модели
Архитектура нейросети повторяет архитектуру, описанную в статье. ***Input module*** принимает на вход Glove-представления конктекста(каждое слово представляется n-мерным вектором, и вектора близких по значению слов близки в смысле расстояния) и выдает один последовательность векторов, характеризующую контекст. ***Question module*** делает то же самое, но на выходе получается один вектор - представление вопроса. Затем ***Memory module***: используя представление вопроса для акцентирования внимания мы делаем проход по выходу Input module и получаем векторное представление ответа, который затем в ***Answer module*** раскодируем в 3 слова (максимальная длина ответа в bAbI tasks). Скрытые представления вопроса и слов конктекста имеют размер 80, слова на вход подаются как 50-ти мерные вектора обученной модели Glove, длину контекста ограничиваю 300 символами, параметры можно изменять.
## Результаты
Модель из статьи, обученная на всех 20 типах вопросов одновременно, дает точность около 83%, что не очень высоко - учитывая, что модель, полученная в процессе работы над проектом, использующая в качестве Memory module простую GRU-сеть дает точность в 95% и намного проще обучается, но мой интерес заключался в реализации модели, предложенной авторами статьи.

Для визуализации работы модели была сделана web-страница, которая по контексту и вопросу выводит предсказанный ответ, а также содержит примеры.
## Выводы
Полученная за время выполнения проекта модель показывает, что достаточно простые типы вопросов с помощью новых алгоритмов, использующих нейросети, обрабатываются успешно, и мы можем считать данный тип задач решенным. Будущий интерес в этой области состоит в решении более сложных и прикладных задач, и внедрении решений в повседневную жизнь. 
## Инструкция по установке
Для обучения нейронной сети вам понадобятся пакеты:
1) Python 3.5
2) [Keras 2.0](https://keras.io/#installation), [Tensorflow 1.0](https://www.tensorflow.org/install/) - библиотеки для машинного обучения (вместе с зависимостями)
3) [Keras_tqdm](http://jupyter.readthedocs.io/en/latest/install.html) - альтернативная визуализация логов при обучении модели
4) Обученная модель [GloVe](http://nlp.stanford.edu/data/glove.6B.zip). Файл `glove.6B.50d.txt` должен быть в папке `ask-me-app`. Чтобы использовать другую версию, следуйте инструкциям ноутбука.
5) [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/install.html)
Для того, чтобы обучить модель без изменений, откройте `build_weights.ipynb` с помощью Jupyter Notebook и просто запустите все ячейки. Все ячейки содержат комментарии, и если вы захотите изменить параметры, это будет несложно.
В итоге получатся файлы: словарь вида индекс-слово `vocabulary.pkl` и веса нейросети`weights.h5`, используемые в следующей части.

Для запуска приложения-сервера понадобятся:
1) Keras 2.0, Tensorflow 1.0
2) [Flask](http://flask.pocoo.org/docs/0.12/installation/) - веб-сервер на python
3) Файлы `vocabulary.pkl` и `weights.h5`

Для запуска веб-сервера выполните:
~~~~
$python3 app.py
~~~~
В консоли вы увидите адрес созданной страницы, которую можно открыть в браузере. Flask позволяет запустить и видимый в интернете web-сервер, для этого используйте подходящий вам способ, который вы можете найти в инструкции к библиотеке.
