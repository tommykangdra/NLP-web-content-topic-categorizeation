# importing the library
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

import time
import re

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def preprocess_soup(text):
    bs = BeautifulSoup(text, 'lxml').get_text()
    return bs


def preprocess(input):
    # removing link
    pat1 = r'http?://[A-Za-z0-9./]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    combined_pat = r'|'.join((pat1, pat2))
    re1 = re.sub(combined_pat, '', input)

    # lower case
    text = re1.lower()

    # remove apostrophy
    remove_apo = text.translate({ord(c): "" for c in "'"})

    # remove special character
    removeSpecialChars = remove_apo.translate(
        {ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
    words = removeSpecialChars.split()

    # loop for preprocessing
    sentence_transform = []
    for word in words:
        word_transform = ""
        for character in word:
            if character.isalpha():
                word_transform += character
        if (word_transform not in stop_words) & (len(word_transform) > 1):
            if not(word_transform.isdigit()):
                word_lemmatize = lemmatizer.lemmatize(word_transform)
                sentence_transform.append(word_lemmatize)

    if len(sentence_transform) >= 100:
        output = ' '.join(sentence_transform[:100])
    else:
        output = ' '.join(sentence_transform)

    output = re.sub(' +', ' ', output)

    return output


conf = SparkConf().setAll([('spark.executor.memory', '16g')])
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession.builder.getOrCreate()

# start from here reading the parquet
time_init = time.time()
sdf = spark.read.parquet("data/raw/data2.parquet")
rdd = sdf.rdd.map(lambda x: (x[1], x[0]))

rdd_out = rdd.map(lambda x: (x[0], preprocess(preprocess_soup(x[1]))))
sdf_out = spark.createDataFrame(rdd_out, schema=['Id', 'OutputText'])

# Sentiment Analysis with Vader
sid = SentimentIntensityAnalyzer()
sentiment_rdd = rdd.map(lambda x: (
    x[0], sid.polarity_scores(x[1])['compound']))
sdf_vader = spark.createDataFrame(
    sentiment_rdd, schema=['Id', 'SentimentScore'])

sdf_out = sdf_out.join(sdf_vader, on='Id', how='inner')

df_out = sdf_out.toPandas()
df_out.to_csv('./data/preprocess/spark_preprocess.csv')

print('total_time:', time.time() - time_init)

# close spark session and context
spark.stop()
sc.stop()
