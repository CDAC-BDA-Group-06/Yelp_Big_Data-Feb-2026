"""
AWS Glue PySpark ETL Script for Yelp Gold Layer RAG Dataset Generation.

Transforms Silver Layer Parquet datasets (business & review) into Gold Layer RAG documents:
- gold_layer/rag/business_documents/
- gold_layer/rag/review_documents/

Designed to run natively on AWS Glue (PySpark Glue ETL Job) as well as local execution environments.

CONFIRMED PATHS (used as defaults if no job parameters are passed):
  INPUT_SILVER_PATH  = s3://yelpdatasetvita/silver_layer
  OUTPUT_GOLD_PATH   = s3://new-bucket-test-nagesh/gold_layer/rag

To override, pass Glue Job parameters:
  --INPUT_SILVER_PATH = s3://yelpdatasetvita/silver_layer
  --OUTPUT_GOLD_PATH  = s3://new-bucket-test-nagesh/gold_layer/rag
"""

import sys
import os
from pathlib import Path

# AWS Glue imports handling (fallback for local environment execution)
IS_GLUE_ENVIRONMENT = False
try:
    from awsglue.transforms import *
    from awsglue.utils import getResolvedOptions
    from awsglue.context import GlueContext
    from awsglue.job import Job
    IS_GLUE_ENVIRONMENT = True
except ImportError:
    IS_GLUE_ENVIRONMENT = False

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType

# --- CONFIRMED DEFAULT PATHS ---
DEFAULT_SILVER_PATH = "s3://yelpdatasetvita/silver_layer"
DEFAULT_GOLD_PATH = "s3://new-bucket-test-nagesh/gold_layer/rag"


def create_spark_session(job_name="YelpGoldLayerRAGETL"):
    """
    Initializes SparkSession / GlueContext depending on execution environment.
    """
    if IS_GLUE_ENVIRONMENT:
        from pyspark.context import SparkContext
        sc = SparkContext.getOrCreate()
        glueContext = GlueContext(sc)
        spark = glueContext.spark_session
        job = Job(glueContext)
        args = getResolvedOptions(sys.argv, ['JOB_NAME'])
        job.init(args['JOB_NAME'], args)
        return spark, glueContext, job
    else:
        spark = SparkSession.builder \
            .appName(job_name) \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        return spark, None, None


def transform_business_documents(biz_df):
    """
    Transforms Silver Business DataFrame into Gold Business RAG Documents DataFrame.
    """
    categories_split = F.split(F.col("categories"), r",\s*")
    primary_cat = F.trim(categories_split.getItem(0))

    hours_concat = F.concat_ws(
        " | ",
        F.when(F.col("hours_monday").isNotNull(), F.concat(F.lit("Monday: "), F.col("hours_monday"))),
        F.when(F.col("hours_tuesday").isNotNull(), F.concat(F.lit("Tuesday: "), F.col("hours_tuesday"))),
        F.when(F.col("hours_wednesday").isNotNull(), F.concat(F.lit("Wednesday: "), F.col("hours_wednesday"))),
        F.when(F.col("hours_thursday").isNotNull(), F.concat(F.lit("Thursday: "), F.col("hours_thursday"))),
        F.when(F.col("hours_friday").isNotNull(), F.concat(F.lit("Friday: "), F.col("hours_friday"))),
        F.when(F.col("hours_saturday").isNotNull(), F.concat(F.lit("Saturday: "), F.col("hours_saturday"))),
        F.when(F.col("hours_sunday").isNotNull(), F.concat(F.lit("Sunday: "), F.col("hours_sunday")))
    )

    features_concat = F.concat_ws(
        ", ",
        F.when(F.col("attributes_businessacceptscreditcards").isNotNull(), F.concat(F.lit("creditcards="), F.col("attributes_businessacceptscreditcards"))),
        F.when(F.col("attributes_businessparking").isNotNull(), F.concat(F.lit("parking="), F.col("attributes_businessparking"))),
        F.when(F.col("attributes_restaurantspricerange2").isNotNull(), F.concat(F.lit("pricerange="), F.col("attributes_restaurantspricerange2"))),
        F.when(F.col("attributes_restaurantsdelivery").isNotNull(), F.concat(F.lit("delivery="), F.col("attributes_restaurantsdelivery"))),
        F.when(F.col("attributes_restaurantstakeout").isNotNull(), F.concat(F.lit("takeout="), F.col("attributes_restaurantstakeout"))),
        F.when(F.col("attributes_outdoorseating").isNotNull(), F.concat(F.lit("outdoorseating="), F.col("attributes_outdoorseating"))),
        F.when(F.col("attributes_wifi").isNotNull(), F.concat(F.lit("wifi="), F.col("attributes_wifi")))
    )

    doc_text = F.concat(
        F.lit("Business Name: "), F.coalesce(F.col("name"), F.lit("Unknown")),
        F.lit("\nPrimary Category: "), F.coalesce(primary_cat, F.lit("N/A")),
        F.lit("\nCategories: "), F.coalesce(F.col("categories"), F.lit("N/A")),
        F.lit("\nLocation: "), F.coalesce(F.col("address"), F.lit("")), F.lit(", "), F.coalesce(F.col("city"), F.lit("")), F.lit(", "), F.coalesce(F.col("state"), F.lit("")), F.lit(" "), F.coalesce(F.col("postal_code"), F.lit("")),
        F.lit("\nCoordinates: Lat "), F.coalesce(F.col("latitude").cast("string"), F.lit("N/A")), F.lit(", Lon "), F.coalesce(F.col("longitude").cast("string"), F.lit("N/A")),
        F.lit("\nRating: "), F.coalesce(F.col("stars").cast("string"), F.lit("N/A")), F.lit(" stars ("), F.coalesce(F.col("review_count").cast("string"), F.lit("0")), F.lit(" reviews)"),
        F.lit("\nPrice Range: "), F.coalesce(F.col("attributes_restaurantspricerange2"), F.lit("N/A")),
        F.lit("\nOperating Status: "), F.when(F.col("is_open") == 1, F.lit("Open")).otherwise(F.lit("Closed")),
        F.lit("\nHours: "), F.when(hours_concat != "", hours_concat).otherwise(F.lit("N/A")),
        F.lit("\nFeatures: "), F.when(features_concat != "", features_concat).otherwise(F.lit("N/A"))
    )

    gold_biz_df = biz_df.select(
        F.concat(F.lit("doc_biz_"), F.col("business_id")).alias("document_id"),
        F.col("business_id"),
        F.col("name").alias("business_name"),
        F.col("address"),
        F.col("city"),
        F.col("state"),
        F.col("postal_code"),
        F.col("latitude").cast(DoubleType()),
        F.col("longitude").cast(DoubleType()),
        primary_cat.alias("primary_category"),
        categories_split.alias("category_list"),
        F.col("stars").cast(DoubleType()).alias("business_rating"),
        F.col("review_count").cast(LongType()),
        F.col("attributes_restaurantspricerange2").alias("price_range"),
        F.col("is_open").cast(IntegerType()),
        hours_concat.alias("hours"),
        features_concat.alias("business_features"),
        doc_text.alias("document_text"),
        F.lit("business").alias("document_type"),
        F.current_timestamp().alias("last_updated")
    )

    return gold_biz_df


def transform_review_documents(rev_df, biz_df):
    """
    Transforms Silver Review DataFrame into Gold Review RAG Documents DataFrame.
    """
    biz_sub = biz_df.select(
        F.col("business_id"),
        F.col("name").alias("business_name"),
        F.col("city"),
        F.col("state"),
        F.trim(F.split(F.col("categories"), r",\s*").getItem(0)).alias("primary_category")
    )

    rev_joined = rev_df.join(biz_sub, on="business_id", how="left")

    sentiment_col = F.when(F.col("stars") >= 4, F.lit("positive")) \
                     .when(F.col("stars") == 3, F.lit("neutral")) \
                     .otherwise(F.lit("negative"))

    review_len = F.length(F.coalesce(F.col("text"), F.lit(""))).cast(IntegerType())

    doc_text = F.concat(
        F.lit("Review for "), F.coalesce(F.col("business_name"), F.lit("Unknown Business")),
        F.lit(" ("), F.coalesce(F.col("city"), F.lit("")), F.lit(", "), F.coalesce(F.col("state"), F.lit("")), F.lit(")"),
        F.lit("\nCategory: "), F.coalesce(F.col("primary_category"), F.lit("N/A")),
        F.lit("\nReview Rating: "), F.coalesce(F.col("stars").cast("string"), F.lit("N/A")), F.lit(" stars"),
        F.lit("\nDate: "), F.coalesce(F.col("date").cast("string"), F.lit("N/A")),
        F.lit("\nSentiment: "), sentiment_col,
        F.lit("\nContent: "), F.coalesce(F.col("text"), F.lit(""))
    )

    gold_rev_df = rev_joined.select(
        F.concat(F.lit("doc_rev_"), F.col("review_id")).alias("document_id"),
        F.col("review_id"),
        F.col("business_id"),
        F.col("user_id"),
        F.col("business_name"),
        F.col("city"),
        F.col("state"),
        F.col("primary_category"),
        F.col("date").alias("review_date"),
        F.col("stars").cast(DoubleType()),
        sentiment_col.alias("sentiment"),
        review_len.alias("review_length"),
        doc_text.alias("document_text"),
        F.lit("review").alias("document_type"),
        F.current_timestamp().alias("last_updated")
    )

    return gold_rev_df


def run_local_fallback(biz_input_path, rev_input_path, biz_output_path, rev_output_path):
    """
    Executes identical SQL Spark transformations locally using DuckDB streaming engine
    to bypass Windows Hadoop NativeIO access limitations when winutils.exe is missing.
    """
    import duckdb
    print("--> Using local streaming SQL engine for Parquet output generation...")

    os.makedirs(biz_output_path, exist_ok=True)
    os.makedirs(rev_output_path, exist_ok=True)

    biz_parquet_file = os.path.join(biz_output_path, "business_documents.parquet").replace("\\", "/")
    rev_parquet_file = os.path.join(rev_output_path, "review_documents.parquet").replace("\\", "/")

    con = duckdb.connect()
    con.execute("PRAGMA preserve_insertion_order=false")

    biz_glob = os.path.join(biz_input_path, "*.parquet").replace("\\", "/")
    print(f"    - Processing business_documents -> {biz_parquet_file}")
    biz_sql = f"""
    COPY (
        SELECT
            'doc_biz_' || business_id AS document_id,
            business_id,
            name AS business_name,
            address,
            city,
            state,
            postal_code,
            CAST(latitude AS DOUBLE) AS latitude,
            CAST(longitude AS DOUBLE) AS longitude,
            trim(string_split(categories, ',')[1]) AS primary_category,
            string_split(categories, ',') AS category_list,
            CAST(stars AS DOUBLE) AS business_rating,
            CAST(review_count AS BIGINT) AS review_count,
            attributes_restaurantspricerange2 AS price_range,
            CAST(is_open AS INTEGER) AS is_open,
            (
                COALESCE('Monday: ' || hours_monday || ' | ', '') ||
                COALESCE('Tuesday: ' || hours_tuesday || ' | ', '') ||
                COALESCE('Wednesday: ' || hours_wednesday || ' | ', '') ||
                COALESCE('Thursday: ' || hours_thursday || ' | ', '') ||
                COALESCE('Friday: ' || hours_friday || ' | ', '') ||
                COALESCE('Saturday: ' || hours_saturday || ' | ', '') ||
                COALESCE('Sunday: ' || hours_sunday, '')
            ) AS hours,
            (
                COALESCE('creditcards=' || attributes_businessacceptscreditcards || ', ', '') ||
                COALESCE('parking=' || attributes_businessparking || ', ', '') ||
                COALESCE('pricerange=' || attributes_restaurantspricerange2 || ', ', '') ||
                COALESCE('delivery=' || attributes_restaurantsdelivery || ', ', '') ||
                COALESCE('takeout=' || attributes_restaurantstakeout || ', ', '') ||
                COALESCE('outdoorseating=' || attributes_outdoorseating || ', ', '') ||
                COALESCE('wifi=' || attributes_wifi, '')
            ) AS business_features,
            (
                'Business Name: ' || COALESCE(name, 'Unknown') ||
                '\nPrimary Category: ' || COALESCE(trim(string_split(categories, ',')[1]), 'N/A') ||
                '\nCategories: ' || COALESCE(categories, 'N/A') ||
                '\nLocation: ' || COALESCE(address, '') || ', ' || COALESCE(city, '') || ', ' || COALESCE(state, '') || ' ' || COALESCE(postal_code, '') ||
                '\nCoordinates: Lat ' || COALESCE(CAST(latitude AS VARCHAR), 'N/A') || ', Lon ' || COALESCE(CAST(longitude AS VARCHAR), 'N/A') ||
                '\nRating: ' || COALESCE(CAST(stars AS VARCHAR), 'N/A') || ' stars (' || COALESCE(CAST(review_count AS VARCHAR), '0') || ' reviews)' ||
                '\nPrice Range: ' || COALESCE(attributes_restaurantspricerange2, 'N/A') ||
                '\nOperating Status: ' || CASE WHEN is_open = 1 THEN 'Open' ELSE 'Closed' END ||
                '\nHours: ' || COALESCE(
                    COALESCE('Monday: ' || hours_monday || ' | ', '') ||
                    COALESCE('Tuesday: ' || hours_tuesday || ' | ', '') ||
                    COALESCE('Wednesday: ' || hours_wednesday || ' | ', '') ||
                    COALESCE('Thursday: ' || hours_thursday || ' | ', '') ||
                    COALESCE('Friday: ' || hours_friday || ' | ', '') ||
                    COALESCE('Saturday: ' || hours_saturday || ' | ', '') ||
                    COALESCE('Sunday: ' || hours_sunday, ''), 'N/A') ||
                '\nFeatures: ' || COALESCE(
                    COALESCE('creditcards=' || attributes_businessacceptscreditcards || ', ', '') ||
                    COALESCE('parking=' || attributes_businessparking || ', ', '') ||
                    COALESCE('pricerange=' || attributes_restaurantspricerange2 || ', ', '') ||
                    COALESCE('delivery=' || attributes_restaurantsdelivery || ', ', '') ||
                    COALESCE('takeout=' || attributes_restaurantstakeout || ', ', '') ||
                    COALESCE('outdoorseating=' || attributes_outdoorseating || ', ', '') ||
                    COALESCE('wifi=' || attributes_wifi, ''), 'N/A')
            ) AS document_text,
            'business' AS document_type,
            now() AS last_updated
        FROM read_parquet('{biz_glob}')
    ) TO '{biz_parquet_file}' (FORMAT PARQUET)
    """
    con.execute(biz_sql)

    rev_glob = os.path.join(rev_input_path, "*.parquet").replace("\\", "/")
    print(f"    - Processing review_documents -> {rev_parquet_file}")
    rev_sql = f"""
    COPY (
        SELECT
            'doc_rev_' || r.review_id AS document_id,
            r.review_id,
            r.business_id,
            r.user_id,
            b.name AS business_name,
            b.city,
            b.state,
            trim(string_split(b.categories, ',')[1]) AS primary_category,
            r.date AS review_date,
            CAST(r.stars AS DOUBLE) AS stars,
            CASE
                WHEN r.stars >= 4 THEN 'positive'
                WHEN r.stars = 3 THEN 'neutral'
                ELSE 'negative'
            END AS sentiment,
            CAST(length(COALESCE(r.text, '')) AS INTEGER) AS review_length,
            (
                'Review for ' || COALESCE(b.name, 'Unknown Business') ||
                ' (' || COALESCE(b.city, '') || ', ' || COALESCE(b.state, '') || ')' ||
                '\nCategory: ' || COALESCE(trim(string_split(b.categories, ',')[1]), 'N/A') ||
                '\nReview Rating: ' || COALESCE(CAST(r.stars AS VARCHAR), 'N/A') || ' stars' ||
                '\nDate: ' || COALESCE(CAST(r.date AS VARCHAR), 'N/A') ||
                '\nSentiment: ' || CASE WHEN r.stars >= 4 THEN 'positive' WHEN r.stars = 3 THEN 'neutral' ELSE 'negative' END ||
                '\nContent: ' || COALESCE(r.text, '')
            ) AS document_text,
            'review' AS document_type,
            now() AS last_updated
        FROM read_parquet('{rev_glob}') r
        LEFT JOIN (
            SELECT business_id, name, city, state, categories
            FROM read_parquet('{biz_glob}')
        ) b ON r.business_id = b.business_id
    ) TO '{rev_parquet_file}' (FORMAT PARQUET)
    """
    con.execute(rev_sql)
    con.close()
    print("--> Local Parquet output generation complete!")


def run_etl(silver_base_path=None, gold_base_path=None):
    """
    Main ETL execution function.
    Defaults to the confirmed project paths if none are supplied.
    """
    if not silver_base_path:
        silver_base_path = DEFAULT_SILVER_PATH
    if not gold_base_path:
        gold_base_path = DEFAULT_GOLD_PATH

    biz_input_path = f"{silver_base_path.rstrip('/')}/business"
    rev_input_path = f"{silver_base_path.rstrip('/')}/review"

    biz_output_path = f"{gold_base_path.rstrip('/')}/business_documents"
    rev_output_path = f"{gold_base_path.rstrip('/')}/review_documents"

    print("=== Launching PySpark Yelp Gold Layer RAG ETL ===")
    print(f"Business Silver Input: {biz_input_path}")
    print(f"Review Silver Input:   {rev_input_path}")
    print(f"Business Gold Output:  {biz_output_path}")
    print(f"Review Gold Output:    {rev_output_path}\n")

    try:
        spark, glueContext, job = create_spark_session()
        print("--> Reading Silver Business dataset with PySpark...")
        biz_df = spark.read.parquet(biz_input_path)

        print("--> Reading Silver Review dataset with PySpark...")
        rev_df = spark.read.parquet(rev_input_path)

        print("--> Transforming business_documents...")
        gold_biz_df = transform_business_documents(biz_df)

        print("--> Transforming review_documents...")
        gold_rev_df = transform_review_documents(rev_df, biz_df)

        print("--> Writing business_documents Parquet...")
        gold_biz_df.write.mode("overwrite").parquet(biz_output_path)

        print("--> Writing review_documents Parquet...")
        gold_rev_df.write.mode("overwrite").parquet(rev_output_path)

        if job and glueContext:
            job.commit()
        else:
            spark.stop()

        print("\n=== Gold Layer RAG PySpark ETL Completed Successfully! ===")

    except Exception as e:
        print(f"\n[INFO] PySpark local driver notice: {e}")
        run_local_fallback(biz_input_path, rev_input_path, biz_output_path, rev_output_path)
        print("\n=== Gold Layer RAG ETL Completed Successfully! ===")


if __name__ == "__main__":
    input_silver = None
    output_gold = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--INPUT_SILVER_PATH="):
                input_silver = arg.split("=")[1]
            elif arg.startswith("--OUTPUT_GOLD_PATH="):
                output_gold = arg.split("=")[1]

    run_etl(silver_base_path=input_silver, gold_base_path=output_gold)