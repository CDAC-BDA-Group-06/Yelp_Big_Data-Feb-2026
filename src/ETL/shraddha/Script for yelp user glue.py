import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql.functions import (
    col,
    trim,
    to_timestamp,
    when
)
from pyspark.sql.types import StringType

# -------------------------------------------------------
# Initialize Glue Job
# -------------------------------------------------------

args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# -------------------------------------------------------
# S3 Paths
# -------------------------------------------------------

INPUT_PATH = "s3://your-bucket/bronze/user.json"
OUTPUT_PATH = "s3://your-bucket/silver/user/"

# -------------------------------------------------------
# Read Bronze JSON
# -------------------------------------------------------

df = spark.read.json(INPUT_PATH)

print("Raw Record Count :", df.count())

# -------------------------------------------------------
# Remove Duplicate Records
# -------------------------------------------------------

df = df.dropDuplicates(["user_id"])

# -------------------------------------------------------
# Remove Missing Primary Key
# -------------------------------------------------------

df = df.filter(col("user_id").isNotNull())

# -------------------------------------------------------
# Trim Spaces from All String Columns
# -------------------------------------------------------

for field in df.schema.fields:
    if isinstance(field.dataType, StringType):
        df = df.withColumn(field.name, trim(col(field.name)))

# -------------------------------------------------------
# Convert Data Types
# -------------------------------------------------------

df = (
    df
    .withColumn(
        "yelping_since",
        to_timestamp(col("yelping_since"),
                     "yyyy-MM-dd HH:mm:ss")
    )
)

# Integer Columns

integer_columns = [
    "review_count",
    "useful",
    "funny",
    "cool",
    "fans",
    "compliment_hot",
    "compliment_more",
    "compliment_profile",
    "compliment_cute",
    "compliment_list",
    "compliment_note",
    "compliment_plain",
    "compliment_cool",
    "compliment_funny",
    "compliment_writer",
    "compliment_photos"
]

for c in integer_columns:
    if c in df.columns:
        df = df.withColumn(c, col(c).cast("int"))

# Double Columns

double_columns = [
    "average_stars"
]

for c in double_columns:
    if c in df.columns:
        df = df.withColumn(c, col(c).cast("double"))

# -------------------------------------------------------
# Handle Obvious Null Values
# -------------------------------------------------------

df = df.fillna({
    "review_count": 0,
    "useful": 0,
    "funny": 0,
    "cool": 0,
    "fans": 0,
    "average_stars": 0.0
})

# -------------------------------------------------------
# Flatten Nested Fields
# -------------------------------------------------------
# user.json has no nested structs
# Therefore no flattening required

# -------------------------------------------------------
# Write Silver Layer
# -------------------------------------------------------

(
    df.write
      .mode("overwrite")
      .option("compression", "snappy")
      .parquet(OUTPUT_PATH)
)

print("Silver Record Count :", df.count())

job.commit()