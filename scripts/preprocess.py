from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

conf = SparkConf().setAll([('spark.executor.memory', '16g')])
sc = SparkContext.getOrCreate(conf=conf)


# start from here reading the parquet
sdf = spark.read.parquet("data/raw/data2.parquet")
rdd = sdf.rdd.map(lambda x: x[0])

time_init = time.time()
sid = SentimentIntensityAnalyzer()
sentiment_rdd = rdd.map(lambda text: (
    text, sid.polarity_scores(text)['compound']))
sentiment = sentiment_rdd.collect()
print('total_time:', time.time() - time_init)

# pd.read_csv('../data/raw/train.csv')
