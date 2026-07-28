# ==============================================================================
# Yelp Gold Layer - Business Intelligence ETL
# AWS Glue 5.0 | PySpark
#
# Source  : s3://yelpdatasetvita/silver_layer/
# Target  : s3://yelpdatasetvita/gold_layer/bi/
#
# Purpose:
# Create dashboard-ready Gold datasets for Athena, Power BI and QuickSight.
#
# Input Datasets
# ---------------
# business/
# review/
# user/
# checkin/
# tip/
#
# Output Datasets
# ----------------
# dim_business/
# fact_reviews/
# fact_checkins/
# dim_date/
# business_metrics/
# ==============================================================================

import sys
import logging

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.sql.types import *

# ------------------------------------------------------------------------------
# Job Initialization
# ------------------------------------------------------------------------------

args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# ------------------------------------------------------------------------------
# Spark Configurations
# ------------------------------------------------------------------------------

spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("YelpGoldBI")

# ------------------------------------------------------------------------------
# S3 Paths
# ------------------------------------------------------------------------------

S3_SILVER_ROOT = "s3://yelpdatasetvita/silver_layer"
S3_GOLD_ROOT = "s3://yelpdatasetvita/gold_layer"

# ------------------------------------------------------------------------------
# Read Dataset
# ------------------------------------------------------------------------------

def read_silver(dataset_name: str, folder: str) -> DataFrame:

    path = f"{S3_SILVER_ROOT}/{folder}"

    logger.info(f"Reading {dataset_name}")
    logger.info(path)

    df = spark.read.parquet(path)

 #   logger.info(f"{dataset_name} Rows : {df.count()}")

    return df


# ------------------------------------------------------------------------------
# Write Dataset
# ------------------------------------------------------------------------------

def write_gold(dataset_name: str,
               df: DataFrame,
               folder: str,
               partition_by=None):

    output_path = f"{S3_GOLD_ROOT}/bi/{folder}"

    logger.info(f"Writing {dataset_name}")

    writer = (
        df.write
        .mode("overwrite")
        .format("parquet")
        .option("compression", "snappy")
    )

    if partition_by:
        writer.partitionBy(*partition_by).save(output_path)
    else:
        writer.save(output_path)

    logger.info(f"Completed : {output_path}")


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def add_review_length(df: DataFrame) -> DataFrame:

    return (
        df
        .withColumn(
            "review_length",
            F.length(F.col("text"))
        )
        .withColumn(
            "word_count",
            F.size(F.split(F.col("text"), " "))
        )
    )


def standardize_categories(df: DataFrame) -> DataFrame:

    return df.withColumn(
        "primary_category",
        F.trim(
            F.split(
                F.col("categories"),
                ","
            )[0]
        )
    )


def add_date_columns(df: DataFrame) -> DataFrame:

    return (
        df
        .withColumn("review_date", F.to_date("date"))
        .withColumn("review_year", F.year("review_date"))
        .withColumn("review_month", F.month("review_date"))
        .withColumn("review_day", F.dayofmonth("review_date"))
        .withColumn("review_week", F.weekofyear("review_date"))
        .withColumn("review_quarter", F.quarter("review_date"))
        .withColumn(
            "review_day_name",
            F.date_format("review_date", "EEEE")
        )
        .withColumn(
            "review_month_name",
            F.date_format("review_date", "MMMM")
        )
    )


# ------------------------------------------------------------------------------
# Data Quality Checks
# ------------------------------------------------------------------------------

def validate_primary_key(df: DataFrame,
                         column_name: str,
                         dataset: str):

    duplicate_count = (
        df.groupBy(column_name)
          .count()
          .filter("count>1")
          .count()
    )

    logger.info(
        f"{dataset} Duplicate {column_name}: {duplicate_count}"
    )


def validate_nulls(df: DataFrame,
                   dataset: str):

    logger.info(f"Null Summary : {dataset}")

    expr = [
        F.count(
            F.when(F.col(c).isNull(), c)
        ).alias(c)
        for c in df.columns
    ]

    df.select(expr).show(truncate=False)


# ------------------------------------------------------------------------------
# Read All Silver Tables
# ------------------------------------------------------------------------------

logger.info("Loading Silver Layer")

business_df = read_silver("Business", "business")
review_df = read_silver("Review", "review")
user_df = read_silver("User", "user")
checkin_df = read_silver("Checkin", "checkin")


# ------------------------------------------------------------------------------
# Basic Data Quality
# ------------------------------------------------------------------------------

validate_primary_key(
    business_df,
    "business_id",
    "Business"
)

validate_primary_key(
    review_df,
    "review_id",
    "Review"
)

validate_primary_key(
    user_df,
    "user_id",
    "User"
)

validate_nulls(business_df, "Business")
validate_nulls(review_df, "Review")

# ------------------------------------------------------------------------------
# Initial Feature Engineering
# ------------------------------------------------------------------------------

business_df = standardize_categories(business_df)

review_df = add_review_length(review_df)

review_df = add_date_columns(review_df)

logger.info("Silver Layer Loaded Successfully")

# ==============================================================================
# BUILD DIMENSION : BUSINESS
# ==============================================================================

def build_dim_business(business_df: DataFrame) -> DataFrame:
    """
    Creates the Business Dimension table for BI dashboards.
    One record per business.
    """

    logger.info("Building dim_business")

    dim_business = (
        business_df
        .select(
            "business_id",
            "name",
            "address",
            "city",
            "state",
            "postal_code",
            "latitude",
            "longitude",
            "stars",
            "review_count",
            "is_open",
            "categories",
            "primary_category",
            "attributes_restaurantspricerange2",
            "attributes_wifi",
            "attributes_outdoorseating",
            "attributes_restaurantsreservations",
            "attributes_goodforkids",
            "attributes_restaurantsattire",
            "attributes_alcohol",
            "attributes_noiselevel",
            "hours_monday",
            "hours_tuesday",
            "hours_wednesday",
            "hours_thursday",
            "hours_friday",
            "hours_saturday",
            "hours_sunday"
        )

        .withColumn(
            "business_status",
            F.when(F.col("is_open") == 1, "Open")
             .otherwise("Closed")
        )

        .withColumn(
            "customer_satisfaction_score",
            F.round(F.col("stars") * 20, 2)
        )

        .dropDuplicates(["business_id"])
    )



    return dim_business


# ==============================================================================
# BUILD FACT TABLE : REVIEWS
# ==============================================================================

def build_fact_reviews(review_df: DataFrame) -> DataFrame:
    """
    Creates Review Fact table.
    One row = One Review.
    """

    logger.info("Building fact_reviews")

    fact_reviews = (
        review_df
        .select(
            "review_id",
            "business_id",
            "user_id",
            "date",
            "stars",
            "useful",
            "funny",
            "cool",
            "weighted_score",
            "review_length",
            "word_count",
            "review_year",
            "review_month",
            "review_day",
            "review_week",
            "review_quarter",
            "review_day_name",
            "review_month_name"
        )

        .dropDuplicates(["review_id"])
    )

 #   logger.info(
 #       f"fact_reviews Rows : {fact_reviews.count()}"
 #   )

    return fact_reviews


# ==============================================================================
# CREATE GOLD DATASETS
# ==============================================================================

logger.info("Creating Business Dimension")

dim_business_df = build_dim_business(
    business_df
)

logger.info("Creating Review Fact")

fact_reviews_df = build_fact_reviews(
    review_df
)


# ==============================================================================
# WRITE GOLD TABLES
# ==============================================================================

write_gold(
    "DIM_BUSINESS",
    dim_business_df,
    "dim_business"
)

write_gold(
    "FACT_REVIEWS",
    fact_reviews_df,
    "fact_reviews",
    partition_by=["review_year"]
)

# ==============================================================================
# BUILD FACT TABLE : CHECKINS
# ==============================================================================

def build_fact_checkins(checkin_df: DataFrame) -> DataFrame:
    """
    Creates Check-in Fact table.

    One row per business.
    Calculates total check-ins and derives useful BI columns.
    """

    logger.info("Building fact_checkins")

    fact_checkins = (
        checkin_df

        .withColumn(
            "checkin_array",
            F.split(F.col("date"), ",")
        )

        .withColumn(
            "checkin_count",
            F.size("checkin_array")
        )

        .select(
            "business_id",
            "checkin_count"
        )
    )

    
    return fact_checkins


# ==============================================================================
# BUILD DATE DIMENSION
# ==============================================================================

def build_dim_date(review_df: DataFrame) -> DataFrame:
    """
    Creates Date Dimension.

    One row per calendar date.
    """

    logger.info("Building dim_date")

    dim_date = (

        review_df

        .select("review_date")

        .distinct()

        .withColumnRenamed(
            "review_date",
            "date"
        )

        .withColumn(
            "year",
            F.year("date")
        )

        .withColumn(
            "quarter",
            F.quarter("date")
        )

        .withColumn(
            "month",
            F.month("date")
        )

        .withColumn(
            "month_name",
            F.date_format("date", "MMMM")
        )

        .withColumn(
            "week",
            F.weekofyear("date")
        )

        .withColumn(
            "day",
            F.dayofmonth("date")
        )

        .withColumn(
            "day_name",
            F.date_format("date", "EEEE")
        )

        .withColumn(
            "weekend_flag",
            F.when(
                F.dayofweek("date").isin([1,7]),
                "Weekend"
            ).otherwise("Weekday")
        )

        .orderBy("date")

    )


    return dim_date


# ==============================================================================
# CREATE GOLD DATASETS
# ==============================================================================

logger.info("Creating Check-in Fact")

fact_checkins_df = build_fact_checkins(
    checkin_df
)

logger.info("Creating Date Dimension")

dim_date_df = build_dim_date(
    review_df
)


# ==============================================================================
# WRITE GOLD TABLES
# ==============================================================================

write_gold(
    "FACT_CHECKINS",
    fact_checkins_df,
    "fact_checkins"
)

write_gold(
    "DIM_DATE",
    dim_date_df,
    "dim_date",
    partition_by=["year"]
)


# ==============================================================================
# BUILD BUSINESS METRICS
# ==============================================================================

def build_business_metrics(
    business_df: DataFrame,
    review_df: DataFrame,
    fact_checkins_df: DataFrame
) -> DataFrame:
    """
    Build Business Metrics table for BI dashboards.

    One row per business.
    """

    logger.info("Building business_metrics")

    # ----------------------------------------------------------
    # Review Aggregation
    # ----------------------------------------------------------

    review_metrics = (
        review_df
        .groupBy("business_id")
        .agg(
            F.count("review_id").alias("total_reviews"),
            F.avg("stars").alias("average_rating"),
            F.avg("review_length").alias("average_review_length"),
            F.sum("useful").alias("total_useful"),
            F.sum("funny").alias("total_funny"),
            F.sum("cool").alias("total_cool")
        )
    )

    # ----------------------------------------------------------
    # Join Business + Reviews + Check-ins
    # ----------------------------------------------------------

    metrics = (
        business_df
        .join(
            review_metrics,
            "business_id",
            "left"
        )
        .join(
            fact_checkins_df,
            "business_id",
            "left"
        )
    )

    # ----------------------------------------------------------
    # Replace NULL values
    # ----------------------------------------------------------

    metrics = metrics.fillna({
        "total_reviews": 0,
        "average_rating": 0,
        "average_review_length": 0,
        "checkin_count": 0,
        "total_useful": 0,
        "total_funny": 0,
        "total_cool": 0
    })

    # ----------------------------------------------------------
    # Rating Score
    # ----------------------------------------------------------

    metrics = metrics.withColumn(
        "rating_score",
        F.round(
            (F.col("average_rating") / 5.0) * 100,
            2
        )
    )

    # ----------------------------------------------------------
    # Review Score
    # ----------------------------------------------------------

    max_reviews = max(
    metrics.agg(F.max("total_reviews")).collect()[0][0] or 0,
    1
)

    metrics = metrics.withColumn(
        "review_score",
        F.round(
            (F.col("total_reviews") / F.lit(max_reviews)) * 100,
            2
        )
    )

    # ----------------------------------------------------------
    # Check-in Score
    # ----------------------------------------------------------

    max_checkins = max(
    metrics.agg(F.max("checkin_count")).collect()[0][0] or 0,
    1
)

    metrics = metrics.withColumn(
        "checkin_score",
        F.round(
            (F.col("checkin_count") / F.lit(max_checkins)) * 100,
            2
        )
    )

    # ----------------------------------------------------------
    # Business Health Score
    # ----------------------------------------------------------

    metrics = metrics.withColumn(

        "business_health_score",

        F.round(

            (
                F.col("rating_score") * 0.50 +
                F.col("review_score") * 0.30 +
                F.col("checkin_score") * 0.20

            ),

            2

        )

    )

    # ----------------------------------------------------------
    # Customer Engagement Score
    # ----------------------------------------------------------

    metrics = metrics.withColumn(

        "customer_engagement_score",

        F.round(

            (
                F.col("review_score") * 0.60 +
                F.col("checkin_score") * 0.40

            ),

            2

        )

    )

    # ----------------------------------------------------------
    # Health Status
    # ----------------------------------------------------------

    metrics = metrics.withColumn(

        "health_status",

        F.when(
            F.col("business_health_score") >= 80,
            "Excellent"
        )

        .when(
            F.col("business_health_score") >= 60,
            "Good"
        )

        .when(
            F.col("business_health_score") >= 40,
            "Average"
        )

        .otherwise(
            "Needs Improvement"
        )

    )

    # ----------------------------------------------------------
    # Engagement Status
    # ----------------------------------------------------------

    metrics = metrics.withColumn(

        "engagement_status",

        F.when(
            F.col("customer_engagement_score") >= 80,
            "High"
        )

        .when(
            F.col("customer_engagement_score") >= 60,
            "Moderate"
        )

        .when(
            F.col("customer_engagement_score") >= 40,
            "Low"
        )

        .otherwise(
            "Very Low"
        )

    )

    # ----------------------------------------------------------
    # Final Columns
    # ----------------------------------------------------------

    metrics = metrics.select(

        "business_id",

        "name",

        "city",

        "state",

        "primary_category",

        "average_rating",

        "total_reviews",

        "checkin_count",

        "average_review_length",

        "rating_score",

        "review_score",

        "checkin_score",

        "business_health_score",

        "customer_engagement_score",

        "health_status",

        "engagement_status"

    )

   

    return metrics


# ==============================================================================
# CREATE BUSINESS METRICS
# ==============================================================================

logger.info("Creating Business Metrics")

business_metrics_df = build_business_metrics(
    business_df,
    review_df,
    fact_checkins_df
)


# ==============================================================================
# WRITE GOLD DATASET
# ==============================================================================

write_gold(

    "BUSINESS_METRICS",

    business_metrics_df,

    "business_metrics"

)

logger.info("All BI Gold datasets created successfully.")

job.commit()