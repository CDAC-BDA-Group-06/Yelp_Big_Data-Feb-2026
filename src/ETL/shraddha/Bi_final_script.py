"""
================================================================================
FINAL ETL GLUE JOB SCRIPT - Silver to Gold Layer Transformation
Ready to Copy & Paste into AWS Glue Job
================================================================================
Purpose: Transform Yelp data from Silver Layer to Gold Layer (Star Schema)
Created: 2026-07-30
Database: yelp_db (Yelp Open Dataset - Business, Review, User, Checkin)
================================================================================
"""

import sys
import math
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
# CONFIGURATION - ADJUST THESE PATHS AS NEEDED
# ================================================================================
SILVER_PATH = "s3://yelpdatasetvita/silver_layer"
GOLD_PATH = "s3://bi-new-shradhha/gold_layer"

logger.info("="*80)
logger.info("GLUE ETL JOB: Silver to Gold Layer Transformation")
logger.info("="*80)
logger.info(f"Silver Layer Path: {SILVER_PATH}")
logger.info(f"Gold Layer Path: {GOLD_PATH}")

# Precomputed constants for log-scale scoring (must be plain Python floats,
# not F.log1p(10000) — F.log1p() requires a Column/str argument, not an int)
LOG1P_10000 = math.log1p(10000)
LOG1P_50000 = math.log1p(50000)

# ================================================================================
# STEP 1: READ SILVER LAYER TABLES
# ================================================================================
logger.info("\n[STEP 1] Reading Silver Layer Tables...")

try:
    business_df = spark.read.parquet(f"{SILVER_PATH}/business/")
    logger.info(f"✓ business: {business_df.count():,} records")
    
    review_df = spark.read.parquet(f"{SILVER_PATH}/review/")
    logger.info(f"✓ review: {review_df.count():,} records")
    
    user_df = spark.read.parquet(f"{SILVER_PATH}/user/")
    logger.info(f"✓ user: {user_df.count():,} records")
    
    checkin_df = spark.read.parquet(f"{SILVER_PATH}/checkin/")
    logger.info(f"✓ checkin: {checkin_df.count():,} records")
    
except Exception as e:
    logger.error(f"ERROR reading silver tables: {str(e)}")
    raise

# ================================================================================
# STEP 2: CREATE DIM_DATE
# ================================================================================
logger.info("\n[STEP 2] Creating dim_date...")

try:
    # Extract all dates from review and checkin
    review_dates = review_df.select(F.col("date")).distinct()
    checkin_dates = checkin_df.select(F.col("date")).distinct()
    
    # Combine and create date dimension
    all_dates = review_dates.union(checkin_dates).distinct()
    all_dates = all_dates.filter(F.col("date").isNotNull())
    
    # Convert to proper date format
    dim_date = all_dates.select(
        F.col("date").cast(DateType()).alias("Date")
    ).distinct()
    
    # Add date components
    dim_date = dim_date.select(
        F.date_format(F.col("Date"), "yyyyMMdd").cast(LongType()).alias("DateKey"),
        F.col("Date"),
        F.month(F.col("Date")).alias("Month"),
        F.date_format(F.col("Date"), "MMMM").alias("MonthName"),
        F.quarter(F.col("Date")).alias("Quarter"),
        F.year(F.col("Date")).alias("Year"),
        F.dayofweek(F.col("Date")).alias("DayOfWeek"),
        F.dayofmonth(F.col("Date")).alias("DayOfMonth"),
        F.weekofyear(F.col("Date")).alias("WeekOfYear")
    ).orderBy("DateKey")
    
    logger.info(f"✓ dim_date: {dim_date.count():,} records")
    
except Exception as e:
    logger.error(f"ERROR creating dim_date: {str(e)}")
    raise

# ================================================================================
# STEP 3: CREATE DIM_BUSINESS
# ================================================================================
logger.info("\n[STEP 3] Creating dim_business...")

try:
    dim_business = business_df.select(
        F.col("business_id").alias("BusinessID"),
        F.col("name").alias("BusinessName"),
        F.col("city").alias("City"),
        F.col("state").alias("State"),
        F.col("address").alias("Address"),
        F.col("postal_code").alias("PostalCode"),
        F.col("latitude").alias("Latitude"),
        F.col("longitude").alias("Longitude"),
        F.split(F.col("categories"), ",")[0].alias("PrimaryCategory"),
        F.col("categories").alias("Categories")
    ).distinct()
    
    logger.info(f"✓ dim_business: {dim_business.count():,} records")
    
except Exception as e:
    logger.error(f"ERROR creating dim_business: {str(e)}")
    raise

# ================================================================================
# STEP 4: CREATE FACT_BUSINESS (MAIN FACT TABLE)
# ================================================================================
logger.info("\n[STEP 4] Creating fact_business (Main KPI Table)...")

try:
    # Aggregate review metrics
    review_metrics = review_df.groupBy("business_id").agg(
        F.count("review_id").alias("ReviewCount"),
        F.round(F.avg("stars"), 2).alias("AvgRating"),
        F.round(F.avg(F.col("useful") + F.col("funny") + F.col("cool")), 2).alias("AvgReviewEngagement"),
        F.round(F.avg(F.length("text")), 2).alias("AvgReviewLength")
    )
    
    # Aggregate checkin metrics
    checkin_metrics = checkin_df.groupBy("business_id").agg(
        F.count("*").alias("CheckinCount")
    )
    
    # Create fact_business
    fact_business = business_df.select("business_id").distinct()
    fact_business = fact_business.join(review_metrics, "business_id", "left")
    fact_business = fact_business.join(checkin_metrics, "business_id", "left")
    
    # Fill nulls
    fact_business = fact_business.fillna(0)
    
    # Calculate scores
    fact_business = fact_business.select(
        F.col("business_id").alias("BusinessID"),
        F.col("ReviewCount").cast(IntegerType()),
        F.col("AvgRating").cast(DoubleType()),
        F.col("CheckinCount").cast(IntegerType()),
        F.col("AvgReviewEngagement").cast(DoubleType()),
        F.col("AvgReviewLength").cast(DoubleType()),
        # Rating Score: 0-100
        F.round((F.col("AvgRating") / 5) * 100, 2).alias("RatingScore"),
        # Review Score: 0-100 (log scale)
        F.round(F.least(F.log1p(F.col("ReviewCount")) / LOG1P_10000 * 100, F.lit(100)), 2).alias("ReviewScore"),
        # Checkin Score: 0-100 (log scale)
        F.round(F.least(F.log1p(F.col("CheckinCount")) / LOG1P_50000 * 100, F.lit(100)), 2).alias("CheckinScore"),
        # Business Health Score (weighted)
        F.round(
            (F.round((F.col("AvgRating") / 5) * 100, 2) * 0.4 +
             F.round(F.least(F.log1p(F.col("ReviewCount")) / LOG1P_10000 * 100, F.lit(100)), 2) * 0.35 +
             F.round(F.least(F.log1p(F.col("CheckinCount")) / LOG1P_50000 * 100, F.lit(100)), 2) * 0.25),
            2
        ).alias("BusinessHealthScore"),
        # Customer Engagement Score
        F.round(
            (F.round(F.least(F.log1p(F.col("ReviewCount")) / LOG1P_10000 * 100, F.lit(100)), 2) * 0.6 +
             F.round(F.least(F.log1p(F.col("CheckinCount")) / LOG1P_50000 * 100, F.lit(100)), 2) * 0.4),
            2
        ).alias("CustomerEngagementScore"),
        # Health Status
        F.when(F.round((F.col("AvgRating") / 5) * 100, 2) >= 75, "Excellent")
         .when(F.round((F.col("AvgRating") / 5) * 100, 2) >= 60, "Good")
         .when(F.round((F.col("AvgRating") / 5) * 100, 2) >= 45, "Average")
         .otherwise("Poor").alias("HealthStatus"),
        # Engagement Status
        F.when(
            F.round(
                (F.round(F.least(F.log1p(F.col("ReviewCount")) / LOG1P_10000 * 100, F.lit(100)), 2) * 0.6 +
                 F.round(F.least(F.log1p(F.col("CheckinCount")) / LOG1P_50000 * 100, F.lit(100)), 2) * 0.4),
                2
            ) >= 70, "High"
        ).when(
            F.round(
                (F.round(F.least(F.log1p(F.col("ReviewCount")) / LOG1P_10000 * 100, F.lit(100)), 2) * 0.6 +
                 F.round(F.least(F.log1p(F.col("CheckinCount")) / LOG1P_50000 * 100, F.lit(100)), 2) * 0.4),
                2
            ) >= 40, "Medium"
        ).otherwise("Low").alias("EngagementStatus"),
        F.current_timestamp().alias("LoadTimestamp")
    )
    
    logger.info(f"✓ fact_business: {fact_business.count():,} records")
    
except Exception as e:
    logger.error(f"ERROR creating fact_business: {str(e)}")
    raise

# ================================================================================
# STEP 5: CREATE FACT_REVIEW_TREND
# ================================================================================
logger.info("\n[STEP 5] Creating fact_review_trend...")

try:
    fact_review_trend = review_df.select(
        F.col("business_id").alias("BusinessID"),
        F.col("date").cast(DateType()).alias("ReviewDate"),
        F.date_format(F.col("date"), "yyyyMMdd").cast(LongType()).alias("DateKey"),
        F.col("stars").alias("Rating")
    ).groupBy("BusinessID", "DateKey", "ReviewDate").agg(
        F.count("Rating").alias("ReviewCount"),
        F.round(F.avg("Rating"), 2).alias("AvgRating"),
        F.min("Rating").alias("MinRating"),
        F.max("Rating").alias("MaxRating")
    ).orderBy("BusinessID", "DateKey")
    
    logger.info(f"✓ fact_review_trend: {fact_review_trend.count():,} records")
    
except Exception as e:
    logger.error(f"ERROR creating fact_review_trend: {str(e)}")
    raise

# ================================================================================
# STEP 6: CREATE FACT_RATING_DISTRIBUTION
# ================================================================================
logger.info("\n[STEP 6] Creating fact_rating_distribution...")

try:
    fact_rating_distribution = review_df.select(
        F.col("business_id").alias("BusinessID"),
        F.col("stars").alias("RatingValue")
    ).groupBy("BusinessID", "RatingValue").agg(
        F.count("*").alias("ReviewCount")
    ).orderBy("BusinessID", "RatingValue")
    
    logger.info(f"✓ fact_rating_distribution: {fact_rating_distribution.count():,} records")
    
except Exception as e:
    logger.error(f"ERROR creating fact_rating_distribution: {str(e)}")
    raise

# ================================================================================
# STEP 7: CREATE DIM_BUSINESS_HOURS
# ================================================================================
logger.info("\n[STEP 7] Creating dim_business_hours...")

try:
    # Extract hours columns
    hours_cols = [col for col in business_df.columns if col.startswith("hours_")]
    
    if hours_cols:
        business_hours_list = []
        for row in business_df.select("business_id", *hours_cols).collect():
            business_id = row["business_id"]
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            for i, day_name in enumerate(days, 1):
                col_name = f"hours_{day_name.lower()}"
                if col_name in hours_cols:
                    hours_str = row[col_name]
                    if hours_str and hours_str != "-":
                        try:
                            times = hours_str.split("-")
                            open_time = times[0].strip() if len(times) > 0 else None
                            close_time = times[1].strip() if len(times) > 1 else None
                            business_hours_list.append((business_id, i, day_name, open_time, close_time))
                        except:
                            pass
        
        if business_hours_list:
            dim_business_hours = spark.createDataFrame(
                business_hours_list,
                ["BusinessID", "DayOfWeekNum", "DayOfWeek", "OpenTime", "CloseTime"]
            )
        else:
            dim_business_hours = spark.createDataFrame(
                [],
                "BusinessID string, DayOfWeekNum int, DayOfWeek string, OpenTime string, CloseTime string"
            )
    else:
        dim_business_hours = spark.createDataFrame(
            [],
            "BusinessID string, DayOfWeekNum int, DayOfWeek string, OpenTime string, CloseTime string"
        )
    
    logger.info(f"✓ dim_business_hours: {dim_business_hours.count():,} records")
    
except Exception as e:
    logger.error(f"ERROR creating dim_business_hours: {str(e)}")
    dim_business_hours = spark.createDataFrame(
        [],
        "BusinessID string, DayOfWeekNum int, DayOfWeek string, OpenTime string, CloseTime string"
    )

# ================================================================================
# STEP 8: CREATE FACT_CHECKIN_DAY
# ================================================================================
logger.info("\n[STEP 8] Creating fact_checkin_day...")

try:
    fact_checkin_day = checkin_df.select(
        F.col("business_id").alias("BusinessID"),
        F.col("date").cast(DateType()).alias("CheckinDate"),
        F.dayofweek(F.col("date")).alias("DayOfWeek"),
        F.date_format(F.col("date"), "yyyyMMdd").cast(LongType()).alias("DateKey")
    ).groupBy("BusinessID", "DateKey", "CheckinDate", "DayOfWeek").agg(
        F.count("*").alias("CheckinCount")
    ).orderBy("BusinessID", "DateKey")
    
    logger.info(f"✓ fact_checkin_day: {fact_checkin_day.count():,} records")
    
except Exception as e:
    logger.error(f"ERROR creating fact_checkin_day: {str(e)}")
    raise

# ================================================================================
# STEP 9: CREATE FACT_CHECKIN_HOUR
# ================================================================================
logger.info("\n[STEP 9] Creating fact_checkin_hour...")

try:
    fact_checkin_hour = checkin_df.select(
        F.col("business_id").alias("BusinessID"),
        F.col("date").cast(DateType()).alias("CheckinDate"),
        F.hour(F.col("date")).alias("HourOfDay"),
        F.date_format(F.col("date"), "yyyyMMdd").cast(LongType()).alias("DateKey")
    ).groupBy("BusinessID", "DateKey", "CheckinDate", "HourOfDay").agg(
        F.count("*").alias("CheckinCount")
    ).orderBy("BusinessID", "DateKey", "HourOfDay")
    
    logger.info(f"✓ fact_checkin_hour: {fact_checkin_hour.count():,} records")
    
except Exception as e:
    logger.error(f"ERROR creating fact_checkin_hour: {str(e)}")
    raise

# ================================================================================
# STEP 10: WRITE TO GOLD LAYER
# ================================================================================
logger.info("\n[STEP 10] Writing Gold Layer Tables to S3...")

try:
    # Write all tables
    dim_date.repartition(1).write.mode("overwrite").parquet(f"{GOLD_PATH}/dim_date/")
    logger.info("✓ dim_date written")
    
    dim_business.repartition(10).write.mode("overwrite").parquet(f"{GOLD_PATH}/dim_business/")
    logger.info("✓ dim_business written")
    
    fact_business.repartition(10).write.mode("overwrite").parquet(f"{GOLD_PATH}/fact_business/")
    logger.info("✓ fact_business written")
    
    fact_review_trend.repartition(20).write.mode("overwrite").parquet(f"{GOLD_PATH}/fact_review_trend/")
    logger.info("✓ fact_review_trend written")
    
    fact_rating_distribution.repartition(10).write.mode("overwrite").parquet(f"{GOLD_PATH}/fact_rating_distribution/")
    logger.info("✓ fact_rating_distribution written")
    
    dim_business_hours.repartition(5).write.mode("overwrite").parquet(f"{GOLD_PATH}/dim_business_hours/")
    logger.info("✓ dim_business_hours written")
    
    fact_checkin_day.repartition(20).write.mode("overwrite").parquet(f"{GOLD_PATH}/fact_checkin_day/")
    logger.info("✓ fact_checkin_day written")
    
    fact_checkin_hour.repartition(20).write.mode("overwrite").parquet(f"{GOLD_PATH}/fact_checkin_hour/")
    logger.info("✓ fact_checkin_hour written")
    
except Exception as e:
    logger.error(f"ERROR writing to gold layer: {str(e)}")
    raise

# ================================================================================
# STEP 11: FINAL VALIDATION & SUMMARY
# ================================================================================
logger.info("\n[STEP 11] Final Validation...")

try:
    summary_data = [
        ("dim_date", dim_date.count()),
        ("dim_business", dim_business.count()),
        ("fact_business", fact_business.count()),
        ("fact_review_trend", fact_review_trend.count()),
        ("fact_rating_distribution", fact_rating_distribution.count()),
        ("dim_business_hours", dim_business_hours.count()),
        ("fact_checkin_day", fact_checkin_day.count()),
        ("fact_checkin_hour", fact_checkin_hour.count()),
    ]
    
    logger.info("\n" + "="*80)
    logger.info("GOLD LAYER SUMMARY - ALL TABLES CREATED SUCCESSFULLY")
    logger.info("="*80)
    for table_name, count in summary_data:
        logger.info(f"  {table_name:.<40} {count:>15,} records")
    logger.info("="*80)
    
except Exception as e:
    logger.error(f"ERROR in validation: {str(e)}")
    raise

# ================================================================================
# JOB COMPLETION
# ================================================================================
job.commit()
logger.info("\n✓ ETL JOB COMPLETED SUCCESSFULLY!")
logger.info("="*80)
