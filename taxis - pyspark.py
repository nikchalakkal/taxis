# Models on 11m records in Pyspark
#AWS setup
aws s3 cp s3://mlprojectmodeling/01-2016.csv ./

hdfs dfs -mkdir /user/data
hdfs dfs -put 01-2016.csv /user/data

#import data
pyspark --packages com.databricks:spark-csv_2.10:1.2.0

from pyspark.sql import SQLContext
from pyspark.sql.types import *
sqlContext = SQLContext(sc)

df = sqlContext.read.load('hdfs:///user/data/01-2016.csv',
format='com.databricks.spark.csv',
header = 'true',
inferSchema= 'true')

#counts
df.count()
#10906858
df.printSchema()

df.groupBy('VendorID').count().orderBy('count', ascending=False).show(50)
df.groupBy('passenger_count').count().orderBy('count', ascending=False).show(50)
df.groupBy('trip_distance').count().orderBy('count', ascending=False).show(50)


#specify schema
from pyspark.sql.functions import unix_timestamp

df=df.select(df['VendorID'],
unix_timestamp('tpep_pickup_datetime', 'yyyy-MM-dd HH:mm:ss').alias('pickup_tmstp'),
unix_timestamp('tpep_dropoff_datetime', 'yyyy-MM-dd HH:mm:ss').alias('dropoff_tmstp'),
df['passenger_count'],
df['trip_distance'],
df['pickup_longitude'],
df['pickup_latitude'],
df['RatecodeID'],
df['store_and_fwd_flag'],
df['dropoff_longitude'],
df['dropoff_latitude'],
df['payment_type'],
df['fare_amount'],
df['extra'],
df['mta_tax'],
df['tip_amount'],
df['tolls_amount'],
df['improvement_surcharge'],
df['total_amount']
)

#create duration variable
df = df.withColumn('duration', df['dropoff_tmstp']-df['pickup_tmstp'])

#filter out negative values and duration greater than 3 hours and total price less than $1000
df = df.filter(df.passenger_count >= 0)
df = df.filter(df.trip_distance >= 0)
df = df.filter(df.RatecodeID >= 0)
df = df.filter(df.fare_amount >= 0)
df = df.filter(df.mta_tax >= 0)
df = df.filter(df.tolls_amount >= 0)
df = df.filter(df.improvement_surcharge >= 0)
df = df.filter(df.total_amount >= 0)
df = df.filter(df.total_amount < 1000)
df = df.filter(df.duration >= 0)
df = df.filter(df.duration < 10800)
df = df.filter(df.duration > 60)

#create dataset to predict duration (features selected from stepwise)
df=df.select(df['trip_distance'],
df['RatecodeID'],
df['tolls_amount'],
df['mta_tax'],
df['payment_type'],
df['extra'],
df['passenger_count'],
df['duration']
)

#LinearRegression for trip duration
import pyspark.ml
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.linalg import Vectors

data = df.select("trip_distance","RatecodeID","tolls_amount","mta_tax","payment_type","extra","passenger_count","duration").rdd.map(lambda r: LabeledPoint(r[7],[r[0],r[1],r[2],r[3],r[4],r[5],r[6]])).toDF()
data.show()

from pyspark.ml.regression import LinearRegression
lr = LinearRegression()

from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.ml.linalg import VectorUDT as VectorUDTML
from pyspark.sql.functions import udf
as_ml = udf(lambda v: v.asML() if v is not None else None, VectorUDTML())

result = data.withColumn("features", as_ml("features"))

#split data and train model
split = result.randomSplit([0.7,0.3])
training = split[0]
test = split[1]

model = lr.fit(training, {lr.regParam:0.01}) #RMSE = 563.95

#apply to test
predictions = model.transform(test)

#evaluate RMSE
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName="rmse")
RMSE = evaluator.evaluate(predictions)
print("Model: Root Mean Squared Error = " + str(RMSE))

#AWS setup
aws s3 cp s3://mlprojectmodeling/01-2016.csv ./

hdfs dfs -mkdir /user/data
hdfs dfs -put 01-2016.csv /user/data

#import data
pyspark --packages com.databricks:spark-csv_2.10:1.2.0

from pyspark.sql import SQLContext
from pyspark.sql.types import *
sqlContext = SQLContext(sc)

df = sqlContext.read.load('hdfs:///user/data/01-2016.csv',
format='com.databricks.spark.csv',
header = 'true',
inferSchema= 'true')

#create duration variable
from pyspark.sql.functions import unix_timestamp

df=df.select(df['VendorID'],
unix_timestamp('tpep_pickup_datetime', 'yyyy-MM-dd HH:mm:ss').alias('pickup_tmstp'),
unix_timestamp('tpep_dropoff_datetime', 'yyyy-MM-dd HH:mm:ss').alias('dropoff_tmstp'),
df['passenger_count'],
df['trip_distance'],
df['pickup_longitude'],
df['pickup_latitude'],
df['RatecodeID'],
df['store_and_fwd_flag'],
df['dropoff_longitude'],
df['dropoff_latitude'],
df['payment_type'],
df['fare_amount'],
df['extra'],
df['mta_tax'],
df['tip_amount'],
df['tolls_amount'],
df['improvement_surcharge'],
df['total_amount']
)

df = df.withColumn('duration', df['dropoff_tmstp']-df['pickup_tmstp'])

#filter out negatives and duration greater than 2 hours and total price less than 500
df = df.filter(df.passenger_count >= 0)
df = df.filter(df.trip_distance >= 0)
df = df.filter(df.RatecodeID >= 0)
df = df.filter(df.fare_amount >= 0)
df = df.filter(df.mta_tax >= 0)
df = df.filter(df.tolls_amount >= 0)
df = df.filter(df.improvement_surcharge >= 0)
df = df.filter(df.total_amount >= 0)
df = df.filter(df.total_amount < 1000)
df = df.filter(df.duration >= 0)
df = df.filter(df.duration < 10800)
df = df.filter(df.duration > 60)

#create dataset to predict total price (features selected from stepwise)
df=df.select(df['trip_distance'],
df['RatecodeID'],
df['mta_tax'],
df['duration'],
df['tolls_amount'],
df['payment_type'],
df['extra'],
df['total_amount']
)

#LinearRegression for total price
import pyspark.ml
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.linalg import Vectors

data = df.select("trip_distance","RatecodeID","mta_tax","duration","tolls_amount","payment_type","extra","total_amount").rdd.map(lambda r: LabeledPoint(r[7],[r[0],r[1],r[2],r[3],r[4],r[5],r[6]])).toDF()
data.show()

from pyspark.ml.regression import LinearRegression
lr = LinearRegression()

from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.ml.linalg import VectorUDT as VectorUDTML
from pyspark.sql.functions import udf
as_ml = udf(lambda v: v.asML() if v is not None else None, VectorUDTML())

result = data.withColumn("features", as_ml("features"))

#split data and train model
split = result.randomSplit([0.7,0.3])
training = split[0]
test = split[1]

model = lr.fit(training, {lr.regParam:0.01}) #RMSE = 5.57

#apply model to test data
predictions = model.transform(test)

#evaluate
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName="rmse")
RMSE = evaluator.evaluate(predictions)
print("Model: Root Mean Squared Error = " + str(RMSE))
