"""
================================================================================
ETL GLUE JOB SCRIPT - fact_review_trend_monthly (Month-wise Grain)
================================================================================
Purpose: Build a MONTH-grain version of fact_review_trend from the Silver
         review table, written to Gold as a NEW table (does not overwrite
         the existing daily fact_review_trend).
================================================================================
"""

import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import *

# ================================================================================
# SETUP: Initialize Spark, Glue, and Logging
# ================================================================================
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
job.init(args['JOB_NAME'], args)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ================================================================================
# CONFIGURATION
# ================================================================================
SILVER_PATH = "s3://yelpdatasetvita/silver_layer"
GOLD_PATH = "s3://bi-new-shradhha/gold-layer-new/"

logger.info("="*80)
logger.info("GLUE ETL JOB: fact_review_trend_monthly")
logger.info("="*80)
logger.info(f"Silver Layer Path: {SILVER_PATH}")
logger.info(f"Gold Layer Path: {GOLD_PATH}")

# ================================================================================
# STEP 1: READ SILVER REVIEW TABLE
# ================================================================================
logger.info("\n[STEP 1] Reading Silver review table...")

try:
    review_df = spark.read.parquet(f"{SILVER_PATH}/review/")
    logger.info(f"✓ review: {review_df.count():,} records")

except Exception as e:
    logger.error(f"ERROR reading silver review table: {str(e)}")
    raise

# ================================================================================
# STEP 2: CREATE FACT_REVIEW_TREND_MONTHLY  (business x month grain)
# ================================================================================
logger.info("\n[STEP 2] Creating fact_review_trend_monthly...")

try:
    fact_review_trend_monthly = review_df.select(
        F.col("business_id").alias("BusinessID"),
        F.date_format(F.col("date"), "yyyyMM").cast(LongType()).alias("MonthKey"),
        F.trunc(F.col("date").cast(DateType()), "month").alias("ReviewMonth"),
        F.col("stars").alias("Rating")
    ).groupBy("BusinessID", "MonthKey", "ReviewMonth").agg(
        F.count("Rating").alias("ReviewCount"),
        F.round(F.avg("Rating"), 2).alias("AvgRating"),
        F.min("Rating").alias("MinRating"),
        F.max("Rating").alias("MaxRating")
    ).orderBy("BusinessID", "MonthKey")

    logger.info(f"✓ fact_review_trend_monthly: {fact_review_trend_monthly.count():,} records")

except Exception as e:
    logger.error(f"ERROR creating fact_review_trend_monthly: {str(e)}")
    raise

# ================================================================================
# STEP 3: WRITE TO GOLD LAYER (new table name, does not touch existing table)
# ================================================================================
logger.info("\n[STEP 3] Writing fact_review_trend_monthly to Gold Layer...")

try:
    fact_review_trend_monthly.repartition(10).write.mode("overwrite").parquet(
        f"{GOLD_PATH}/fact_review_trend_monthly/"
    )
    logger.info("✓ fact_review_trend_monthly written")

except Exception as e:
    logger.error(f"ERROR writing fact_review_trend_monthly: {str(e)}")
    raise

# ================================================================================
# STEP 4: FINAL VALIDATION
# ================================================================================
logger.info("\n[STEP 4] Final Validation...")
logger.info("="*80)
logger.info(f"  fact_review_trend_monthly {'.'*20} {fact_review_trend_monthly.count():>15,} records")
logger.info("="*80)

# ================================================================================
# JOB COMPLETION
# ================================================================================
job.commit()
logger.info("\n✓ ETL JOB COMPLETED SUCCESSFULLY!")
logger.info("="*80)