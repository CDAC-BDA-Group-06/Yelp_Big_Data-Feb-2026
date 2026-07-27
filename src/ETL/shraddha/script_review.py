import sys

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext

from pyspark.sql.functions import (
    col,
    trim,
    to_timestamp,
    current_timestamp
)

from pyspark.sql.types import (
    IntegerType,
    DoubleType
)

# -----------------------------------------------------
# Initialize Glue Job
# -----------------------------------------------------

args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

print("========== GLUE ETL JOB STARTED ==========")

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

DATABASE_NAME = "yelp_db"
TABLE_NAME = "yelp_academic_dataset_review_json"

OUTPUT_PATH = "s3://yelpdatasetbda/review_silver/"

# -----------------------------------------------------
# Read Data from Glue Data Catalog
# -----------------------------------------------------

print("Reading data from Glue Data Catalog...")

dynamic_frame = glueContext.create_dynamic_frame.from_catalog(
    database=DATABASE_NAME,
    table_name=TABLE_NAME
)

df = dynamic_frame.toDF()

initial_count = df.count()

print(f"Initial Record Count : {initial_count}")

# -----------------------------------------------------
# Remove Duplicate Reviews
# -----------------------------------------------------

df = df.dropDuplicates(["review_id"])

# -----------------------------------------------------
# Remove Records with Null Primary Keys
# -----------------------------------------------------

df = df.filter(
    col("review_id").isNotNull() &
    col("business_id").isNotNull() &
    col("user_id").isNotNull()
)

# -----------------------------------------------------
# Trim String Columns
# -----------------------------------------------------

string_columns = [
    "review_id",
    "business_id",
    "user_id",
    "text"
]

for column in string_columns:
    df = df.withColumn(column, trim(col(column)))

# -----------------------------------------------------
# Convert Data Types
# -----------------------------------------------------

df = df.withColumn(
    "stars",
    col("stars").cast(DoubleType())
)

df = df.withColumn(
    "useful",
    col("useful").cast(IntegerType())
)

df = df.withColumn(
    "funny",
    col("funny").cast(IntegerType())
)

df = df.withColumn(
    "cool",
    col("cool").cast(IntegerType())
)

df = df.withColumn(
    "date",
    to_timestamp(col("date"))
)

# -----------------------------------------------------
# Handle Null Values
# -----------------------------------------------------

df = df.fillna({
    "text": "",
    "useful": 0,
    "funny": 0,
    "cool": 0
})

# -----------------------------------------------------
# Add ETL Timestamp
# -----------------------------------------------------

df = df.withColumn(
    "etl_load_time",
    current_timestamp()
)

# -----------------------------------------------------
# Calculate Final Count
# -----------------------------------------------------

final_count = df.count()

print(f"Final Record Count   : {final_count}")
print(f"Records Removed      : {initial_count - final_count}")

# -----------------------------------------------------
# Write as Snappy Parquet
# -----------------------------------------------------

print("Writing Parquet files to S3...")

(
    df.write
      .mode("overwrite")
      .option("compression", "snappy")
      .parquet(OUTPUT_PATH)
)

print("Parquet files successfully written.")

# -----------------------------------------------------
# Commit Glue Job
# -----------------------------------------------------

job.commit()

print("========== GLUE ETL JOB COMPLETED ==========")