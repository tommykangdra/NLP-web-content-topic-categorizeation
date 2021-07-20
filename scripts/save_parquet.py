import pandas as pd
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

conf = SparkConf().setAll([('spark.executor.memory', '8g'),
                           ('spark.driver.memory', '8g')])
sc = SparkContext.getOrCreate(conf=conf)

# data
df = pd.read_csv('data/raw/train.csv')
string_array_2 = df.Text.astype('str').values

time_init = time.time()
rdd = sc.parallelize(string_array_2, 200)
rdd = rdd.map(lambda x: (x,))

# save as parquet
sdf = spark.createDataFrame(rdd, schema=['Text'])
sdf.createOrReplaceTempView("web_text")
sdf.write.mode("overwrite").parquet("data/raw/data2.parquet")
print(time.time() - time_init)

sc.stop()
