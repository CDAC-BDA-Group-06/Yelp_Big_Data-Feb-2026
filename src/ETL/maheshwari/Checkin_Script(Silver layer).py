
# **Script:-**
# User Checkin Script:-
"""
AWS Glue ETL Job

Yelp Checkin Dataset
Bronze (Glue Data Catalog) -> Silver (Parquet + Snappy)

Source:
Database : yelp_db
Table    : yelp_academic_dataset_checkin_json

Destination:
s3://yelpdatasetvita/silver_layer/checkin/
"""

import re
import sys

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext

from pyspark.sql.functions import (
    col,
    trim,
    when,
    current_timestamp
)

from pyspark.sql.types import StringType

# ---------------------------------------------------------------------
# Initialize Glue Job
# ---------------------------------------------------------------------

args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATABASE_NAME = "yelp_db"
TABLE_NAME = "yelp_academic_dataset_checkin_json"

SILVER_PATH = "s3://yelpdatasetvita/silver_layer/checkin/"

PRIMARY_KEY = "business_id"

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

def log(message):
    print(f"[CHECKIN_ETL] {message}")

# ---------------------------------------------------------------------
# Standardize Column Names
# ---------------------------------------------------------------------

def standardize_column_name(column_name):

    column_name = column_name.strip()
    column_name = column_name.replace(" ", "_")
    column_name = column_name.replace("-", "_")
    column_name = re.sub(r"[^a-zA-Z0-9_]", "", column_name)
    column_name = re.sub(r"_+", "_", column_name)

    return column_name.lower()

# ---------------------------------------------------------------------
# Read Bronze Table from Glue Catalog
# ---------------------------------------------------------------------

log("Reading Bronze table from Glue Data Catalog...")

dynamic_frame = glueContext.create_dynamic_frame.from_catalog(
    database=DATABASE_NAME,
    table_name=TABLE_NAME
)

checkin_df = dynamic_frame.toDF()

# Cache because count() is used multiple times
checkin_df = checkin_df.cache()

log(f"Rows Read : {checkin_df.count():,}")
log(f"Columns   : {len(checkin_df.columns)}")

# ---------------------------------------------------------------------
# Standardize Column Names
# ---------------------------------------------------------------------

checkin_df = checkin_df.toDF(
    *[
        standardize_column_name(c)
        for c in checkin_df.columns
    ]
)

# ---------------------------------------------------------------------
# Trim String Columns
# ---------------------------------------------------------------------

string_columns = [
    field.name
    for field in checkin_df.schema.fields
    if isinstance(field.dataType, StringType)
]

for column_name in string_columns:

    checkin_df = checkin_df.withColumn(
        column_name,
        trim(col(column_name))
    )

# ---------------------------------------------------------------------
# Empty String -> NULL
# ---------------------------------------------------------------------

for column_name in string_columns:

    checkin_df = checkin_df.withColumn(
        column_name,
        when(
            trim(col(column_name)) == "",
            None
        ).otherwise(col(column_name))
    )

# ---------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------

null_pk_count = checkin_df.filter(
    col(PRIMARY_KEY).isNull()
).count()

duplicate_pk_count = (
    checkin_df
    .groupBy(PRIMARY_KEY)
    .count()
    .filter(col("count") > 1)
    .count()
)

log(f"Null Primary Keys      : {null_pk_count}")
log(f"Duplicate Primary Keys : {duplicate_pk_count}")

# ---------------------------------------------------------------------
# Remove Null / Blank Primary Keys
# ---------------------------------------------------------------------

rows_before = checkin_df.count()

checkin_df = checkin_df.filter(
    col(PRIMARY_KEY).isNotNull()
)

checkin_df = checkin_df.filter(
    trim(col(PRIMARY_KEY)) != ""
)

rows_after = checkin_df.count()

log(f"Removed {rows_before - rows_after} rows with null/blank business_id")

# ---------------------------------------------------------------------
# Remove Duplicate Records
# ---------------------------------------------------------------------

rows_before = checkin_df.count()

checkin_df = checkin_df.dropDuplicates(
    [PRIMARY_KEY]
)

rows_after = checkin_df.count()

log(f"Removed {rows_before - rows_after} duplicate rows")

# ---------------------------------------------------------------------
# Add ETL Timestamp
# ---------------------------------------------------------------------

checkin_df = checkin_df.withColumn(
    "etl_processed_timestamp",
    current_timestamp()
)

# ---------------------------------------------------------------------
# Reduce Number of Output Files
# ---------------------------------------------------------------------

checkin_df = checkin_df.coalesce(1)

# ---------------------------------------------------------------------
# Write Silver Layer
# ---------------------------------------------------------------------

final_count = checkin_df.count()

log(f"Writing {final_count:,} rows to Silver Layer...")

checkin_df.write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(SILVER_PATH)

log("Silver Layer written successfully.")

# ---------------------------------------------------------------------
# Commit Job
# ---------------------------------------------------------------------

job.commit()

log("Glue Job Completed Successfully.")
