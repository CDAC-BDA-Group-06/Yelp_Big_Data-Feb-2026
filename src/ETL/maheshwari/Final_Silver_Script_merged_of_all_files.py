"""
AWS Glue ETL Job — Consolidated Bronze -> Silver

Yelp Academic Dataset
Processes: Business, User, Checkin, Review, Tip
Source   : Glue Data Catalog (database "yelp_db")
Target   : s3://yelpdatasetvita/silver_layer/<dataset>/  (Parquet + Snappy)

This job runs all five dataset pipelines sequentially inside a single
Glue job run. Each dataset keeps its own transformation logic exactly
as it existed in its original standalone script — nothing was
generalized away or force-fit into a one-size-fits-all cast map.
Only genuinely shared plumbing (Glue/Spark init, logging, struct
flattening, column-name standardization, trim / blank-to-null) is
factored into shared utilities.

NOTE ON THE TIP DATASET:
No source script for Tip was provided. Its configuration below is
built from the public Yelp Academic Dataset Tip schema
(user_id, business_id, text, date, compliment_count) and mirrors the
trim / cast / timestamp / dedupe conventions used across the other
four scripts. Replace TIP_ETL's logic if your actual source differs.
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
    to_timestamp,
    to_date,
    date_format,
    current_timestamp,
    coalesce,
    lit,
    round as spark_round,
)

from pyspark.sql.types import StructType, StringType


# =======================================================================
# Glue / Spark Bootstrap  (shared across all datasets)
# =======================================================================

args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

DATABASE_NAME = "yelp_db"
S3_SILVER_ROOT = "s3://yelpdatasetvita/silver_layer"


# =======================================================================
# Shared Utilities
# =======================================================================

def log(dataset_tag, message):
    """Uniform structured logging across all dataset pipelines."""
    print(f"[{dataset_tag}_ETL] {message}")


def standardize_column_name(name):
    """
    Lowercases, strips, and replaces spaces/hyphens/illegal characters
    with underscores. Used by Business and Checkin (their original
    scripts both defined this identically).
    """
    name = name.strip()
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)
    name = re.sub(r"_+", "_", name)
    return name.lower()


def flatten_df(df):
    """
    Recursively flattens nested struct columns into
    parent_child-named top-level columns. Used only by Business.
    """
    while True:
        struct_cols = [
            field.name
            for field in df.schema.fields
            if isinstance(field.dataType, StructType)
        ]
        if len(struct_cols) == 0:
            break
        for struct_col in struct_cols:
            expanded_cols = [
                col(f"{struct_col}.{field.name}").alias(f"{struct_col}_{field.name}")
                for field in df.schema[struct_col].dataType.fields
            ]
            df = df.select(
                *[c for c in df.columns if c != struct_col],
                *expanded_cols,
            )
    return df


def trim_string_columns(df, columns=None):
    """
    Trims whitespace on the given string columns, or on every
    StringType column if none are specified.
    """
    string_cols = columns or [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, StringType)
    ]
    for c in string_cols:
        df = df.withColumn(c, trim(col(c)))
    return df


def blank_strings_to_null(df, columns):
    """Converts empty/whitespace-only strings to NULL for given columns."""
    for c in columns:
        df = df.withColumn(c, when(trim(col(c)) == "", None).otherwise(col(c)))
    return df


def read_bronze(tag, table_name):
    log(tag, f"Reading Bronze table '{table_name}' from Glue Data Catalog...")
    dynamic_frame = glueContext.create_dynamic_frame.from_catalog(
        database=DATABASE_NAME,
        table_name=table_name,
    )
    df = dynamic_frame.toDF()
    log(tag, f"Rows Read : {df.count():,}")
    log(tag, f"Columns   : {len(df.columns)}")
    return df


def write_silver(tag, df, output_path, coalesce_to_one=False):
    if coalesce_to_one:
        df = df.coalesce(1)
    final_count = df.count()
    log(tag, f"Writing {final_count:,} rows to Silver Layer at {output_path}")
    df.write \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(output_path)
    log(tag, "Silver Layer written successfully.")


# =======================================================================
# 1. BUSINESS
# =======================================================================

def process_business():
    tag = "BUSINESS"
    table_name = "yelp_academic_dataset_business_json"
    output_path = f"{S3_SILVER_ROOT}/business/"
    primary_key = "business_id"

    df = read_bronze(tag, table_name)

    df = flatten_df(df)
    log(tag, f"Columns after flattening : {len(df.columns)}")

    df = df.toDF(*[standardize_column_name(c) for c in df.columns])

    string_cols = [
        field.name for field in df.schema.fields
        if isinstance(field.dataType, StringType)
    ]
    df = trim_string_columns(df, string_cols)
    df = blank_strings_to_null(df, string_cols)

    # --- Validation report ---
    null_pk_count = df.filter(col(primary_key).isNull()).count()
    duplicate_pk_count = (
        df.groupBy(primary_key).count().filter("count > 1").count()
    )
    invalid_stars_count = (
        df.filter((col("stars") < 1) | (col("stars") > 5)).count()
        if "stars" in df.columns else 0
    )
    negative_review_count = (
        df.filter(col("review_count") < 0).count()
        if "review_count" in df.columns else 0
    )
    invalid_latitude = (
        df.filter((col("latitude") < -90) | (col("latitude") > 90)).count()
        if "latitude" in df.columns else 0
    )
    invalid_longitude = (
        df.filter((col("longitude") < -180) | (col("longitude") > 180)).count()
        if "longitude" in df.columns else 0
    )

    log(tag, f"Null Primary Keys        : {null_pk_count}")
    log(tag, f"Duplicate Primary Keys   : {duplicate_pk_count}")
    log(tag, f"Invalid Stars            : {invalid_stars_count}")
    log(tag, f"Negative Review Count    : {negative_review_count}")
    log(tag, f"Invalid Latitude         : {invalid_latitude}")
    log(tag, f"Invalid Longitude        : {invalid_longitude}")

    # --- Remove null/blank PK ---
    before = df.count()
    df = df.filter(col(primary_key).isNotNull())
    df = df.filter(trim(col(primary_key)) != "")
    after = df.count()
    log(tag, f"Removed {before - after} rows having null/blank primary key")

    # --- Remove duplicates ---
    before = df.count()
    df = df.dropDuplicates([primary_key])
    after = df.count()
    log(tag, f"Removed {before - after} duplicate rows")

    # --- Cast types ---
    cast_map = {
        "stars": "double",
        "review_count": "int",
        "latitude": "double",
        "longitude": "double",
        "is_open": "int",
    }
    for column_name, data_type in cast_map.items():
        if column_name in df.columns:
            df = df.withColumn(column_name, col(column_name).cast(data_type))

    df = df.withColumn("etl_processed_timestamp", current_timestamp())

    write_silver(tag, df, output_path, coalesce_to_one=False)


# =======================================================================
# 2. USER
# =======================================================================

def process_user():
    tag = "USER"
    table_name = "yelp_academic_dataset_user_json"
    output_path = f"{S3_SILVER_ROOT}/user/"
    primary_key = "user_id"

    df = read_bronze(tag, table_name)
    df = df.cache()

    # --- Remove duplicates first (matches original order) ---
    before = df.count()
    df = df.dropDuplicates([primary_key])
    after = df.count()
    log(tag, f"Removed {before - after} duplicate records")

    # --- Remove missing PK ---
    before = df.count()
    df = df.filter(col(primary_key).isNotNull())
    df = df.filter(trim(col(primary_key)) != "")
    after = df.count()
    log(tag, f"Removed {before - after} null/blank primary keys")

    # --- Trim all string columns ---
    df = trim_string_columns(df)

    # --- Type conversions ---
    if "yelping_since" in df.columns:
        df = df.withColumn(
            "yelping_since",
            to_timestamp(col("yelping_since"), "yyyy-MM-dd HH:mm:ss"),
        )

    integer_columns = [
        "review_count", "useful", "funny", "cool", "fans",
        "compliment_hot", "compliment_more", "compliment_profile",
        "compliment_cute", "compliment_list", "compliment_note",
        "compliment_plain", "compliment_cool", "compliment_funny",
        "compliment_writer", "compliment_photos",
    ]
    for column_name in integer_columns:
        if column_name in df.columns:
            df = df.withColumn(column_name, col(column_name).cast("int"))

    if "average_stars" in df.columns:
        df = df.withColumn("average_stars", col("average_stars").cast("double"))

    # --- Fill missing values ---
    fill_values = {
        "review_count": 0, "useful": 0, "funny": 0,
        "cool": 0, "fans": 0, "average_stars": 0.0,
    }
    df = df.fillna(fill_values)

    df = df.withColumn("etl_processed_timestamp", current_timestamp())

    write_silver(tag, df, output_path, coalesce_to_one=True)


# =======================================================================
# 3. CHECKIN
# =======================================================================

def process_checkin():
    tag = "CHECKIN"
    table_name = "yelp_academic_dataset_checkin_json"
    output_path = f"{S3_SILVER_ROOT}/checkin/"
    primary_key = "business_id"

    df = read_bronze(tag, table_name)
    df = df.cache()

    df = df.toDF(*[standardize_column_name(c) for c in df.columns])

    string_columns = [
        field.name for field in df.schema.fields
        if isinstance(field.dataType, StringType)
    ]
    df = trim_string_columns(df, string_columns)
    df = blank_strings_to_null(df, string_columns)

    null_pk_count = df.filter(col(primary_key).isNull()).count()
    duplicate_pk_count = (
        df.groupBy(primary_key).count().filter(col("count") > 1).count()
    )
    log(tag, f"Null Primary Keys      : {null_pk_count}")
    log(tag, f"Duplicate Primary Keys : {duplicate_pk_count}")

    rows_before = df.count()
    df = df.filter(col(primary_key).isNotNull())
    df = df.filter(trim(col(primary_key)) != "")
    rows_after = df.count()
    log(tag, f"Removed {rows_before - rows_after} rows with null/blank business_id")

    rows_before = df.count()
    df = df.dropDuplicates([primary_key])
    rows_after = df.count()
    log(tag, f"Removed {rows_before - rows_after} duplicate rows")

    df = df.withColumn("etl_processed_timestamp", current_timestamp())

    write_silver(tag, df, output_path, coalesce_to_one=True)


# =======================================================================
# 4. REVIEW
# =======================================================================

def process_review():
    tag = "REVIEW"
    table_name = "yelp_academic_dataset_review_json"
    output_path = f"{S3_SILVER_ROOT}/review/"

    df = read_bronze(tag, table_name)
    initial_count = df.count()
    log(tag, f"Initial Record Count : {initial_count}")

    # --- Remove duplicate reviews ---
    df = df.dropDuplicates(["review_id"])

    # --- Remove null primary/foreign keys ---
    df = df.filter(
        col("review_id").isNotNull()
        & col("business_id").isNotNull()
        & col("user_id").isNotNull()
    )

    # --- Trim specific string columns ---
    string_columns = ["review_id", "business_id", "user_id", "text"]
    df = trim_string_columns(df, string_columns)

    # --- Type conversions ---
    df = (
        df
        .withColumn("date", col("date").cast("timestamp"))
        .withColumn("time_stamp", date_format(col("date"), "HH:mm:ss"))
        .withColumn("date", to_date(col("date")))
        .withColumn("stars", col("stars").cast("double"))
    )

    for c in ["useful", "funny", "cool"]:
        df = df.withColumn(c, coalesce(col(c).cast("int"), lit(0)))

    df = df.fillna({"text": ""})

    df = df.withColumn("etl_load_time", current_timestamp())

    final_count = df.count()
    log(tag, f"Final Record Count   : {final_count}")
    log(tag, f"Records Removed      : {initial_count - final_count}")

    # --- Weighted score ---
    df = df.withColumn(
        "weighted_score",
        spark_round(
            (
                0.7 * (col("stars") / 5.0)
                + 0.3 * ((col("useful") + col("funny") + col("cool")) / 30.0)
            ) * 10
        ).cast("int"),
    )

    write_silver(tag, df, output_path, coalesce_to_one=False)


# =======================================================================
# 5. TIP
#    NOTE: No source script was provided for Tip. This follows the
#    standard public Yelp Tip schema and mirrors the trim / cast /
#    timestamp / dedupe conventions used in the four confirmed
#    scripts above. Adjust to your actual source if it differs.
# =======================================================================

def process_tip():
    tag = "TIP"
    table_name = "yelp_academic_dataset_tip_json"
    output_path = f"{S3_SILVER_ROOT}/tip/"

    df = read_bronze(tag, table_name)
    initial_count = df.count()
    log(tag, f"Initial Record Count : {initial_count}")

    # --- Remove records missing required foreign keys ---
    df = df.filter(col("business_id").isNotNull() & col("user_id").isNotNull())

    # --- Trim string columns ---
    string_columns = [
        field.name for field in df.schema.fields
        if isinstance(field.dataType, StringType)
    ]
    df = trim_string_columns(df, string_columns)
    df = blank_strings_to_null(df, string_columns)

    # --- Remove exact duplicate tips (same user, business, text, date) ---
    dedupe_cols = [c for c in ["user_id", "business_id", "text", "date"] if c in df.columns]
    before = df.count()
    df = df.dropDuplicates(dedupe_cols)
    after = df.count()
    log(tag, f"Removed {before - after} duplicate tip rows")

    # --- Type conversions ---
    if "date" in df.columns:
        df = df.withColumn("date", col("date").cast("timestamp"))

    if "compliment_count" in df.columns:
        df = df.withColumn(
            "compliment_count",
            coalesce(col("compliment_count").cast("int"), lit(0)),
        )

    if "text" in df.columns:
        df = df.fillna({"text": ""})

    df = df.withColumn("etl_processed_timestamp", current_timestamp())

    final_count = df.count()
    log(tag, f"Final Record Count   : {final_count}")
    log(tag, f"Records Removed      : {initial_count - final_count}")

    write_silver(tag, df, output_path, coalesce_to_one=True)


# =======================================================================
# Orchestration — run every dataset pipeline in this single Glue job
# =======================================================================

PIPELINES = [
    ("BUSINESS", process_business),
    ("USER", process_user),
    ("CHECKIN", process_checkin),
    ("REVIEW", process_review),
    ("TIP", process_tip),
]

failures = []

for name, pipeline_fn in PIPELINES:
    print(f"\n========== STARTING {name} PIPELINE ==========")
    try:
        pipeline_fn()
        print(f"========== {name} PIPELINE COMPLETED ==========\n")
    except Exception as e:
        print(f"!!!!!!!!!! {name} PIPELINE FAILED: {e} !!!!!!!!!!")
        failures.append(name)

if failures:
    print(f"\nCompleted with failures in: {', '.join(failures)}")
else:
    print("\nAll five dataset pipelines completed successfully.")

# =======================================================================
# Commit Job
# =======================================================================

job.commit()

print("========== GLUE CONSOLIDATED BRONZE -> SILVER JOB FINISHED ==========")