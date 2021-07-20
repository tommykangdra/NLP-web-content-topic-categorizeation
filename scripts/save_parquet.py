import pandas as pd
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

import time

conf = SparkConf().setAll([('spark.executor.memory', '8g'),
                           ('spark.driver.memory', '8g')])
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession.builder.getOrCreate()

# data
df = pd.read_csv('data/raw/train.csv')
string_array = df.Text.astype('str').values

time_init = time.time()

# rdd and spark dataframe
rdd = sc.parallelize(string_array, 200).zipWithIndex()
sdf = spark.createDataFrame(rdd, schema=['Text', 'Id'])

# write parquet in data raw
sdf.write.mode("overwrite").parquet("data/raw/data2.parquet")

print(time.time() - time_init)

sc.stop()
spark.stop()
