from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark_iforest.ml.iforest import *
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.functions import array
from pyspark.sql import SQLContext as sqlContext
from pyspark.ml.feature import VectorAssembler
import tempfile
import time


start = time.time()

file = '/Users/xue/Desktop/Farrago/Datasets/Netlogx Training Data/combined_csv.csv'

spark = SparkSession \
        .builder.master("local[*]") \
        .appName("IForestExample") \
        .getOrCreate()

data = spark.read.csv(file
                      , header=False
                      , inferSchema=True)
print('Number of Rows: {}'.format(data.count()))

columns_not_str = [item[0] for item in data.dtypes if not item[1].startswith('string')]
print(columns_not_str)

# convert columns to vectors
df = data.select(array(columns_not_str).alias('features'))
to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
df = df.select(to_vector("features").alias("features"))
df.show()

# convert value in columns to vector directly
# vecAssembler = VectorAssembler(inputCols=columns_not_str, outputCol="features")
# df = vecAssembler.transform(data)
# df.show()

# Init an IForest Object and Fit on a given data frame
iforest = IForest(maxDepth=10
                  , numTrees=100
                  , bootstrap=False
                  , approxQuantileRelativeError=0.01
                  #, contamination=0.05
                  , maxSamples=1000
                  , maxFeatures=20)

model = iforest.fit(df)

if model.hasSummary:

    summary = model.summary
    print('Number of anomalier: {}'.format(summary.numAnomalies))
else:
    print('Does not have summary')

transformed = model.transform(df)
transformed.show()

end = time.time()
print(end-start)