"""
AWS Glue PySpark job: Yelp Business Bronze -> Silver

Required job parameters:
  --JOB_NAME
  --BRONZE_PATH   e.g. s3://my-bucket/bronze/yelp/yelp_academic_dataset_business.json
  --SILVER_PATH   e.g. s3://my-bucket/silver/yelp/business/

Optional job parameters:
  --WRITE_MODE          overwrite|append   (default: overwrite)
  --PIPELINE_VERSION    (default: 2.1-glue)
  --OUTPUT_PARTITIONS   positive integer; omit to keep Spark's partitioning

Recommended runtime: AWS Glue 5.0 or 5.1.
No third-party Python libraries are required.
"""

import sys
import json
import time
from typing import Dict, List, Optional

import boto3

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T


def optional_arg(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read an optional Glue argument without causing getResolvedOptions to fail."""
    flag = f"--{name}"
    if flag not in sys.argv:
        return default
    index = sys.argv.index(flag)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Argument {flag} was supplied without a value.")
    return sys.argv[index + 1]


required_args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "BRONZE_PATH", "SILVER_PATH"],
)

JOB_NAME = required_args["JOB_NAME"]
BRONZE_PATH = required_args["BRONZE_PATH"].rstrip("/")
SILVER_PATH = required_args["SILVER_PATH"].rstrip("/") + "/"
WRITE_MODE = (optional_arg("WRITE_MODE", "overwrite") or "overwrite").lower()
PIPELINE_VERSION = optional_arg("PIPELINE_VERSION", "2.1-glue") or "2.1-glue"
OUTPUT_PARTITIONS_RAW = optional_arg("OUTPUT_PARTITIONS")

# Optional Glue Data Catalog crawler settings.
CRAWLER_NAME = optional_arg("CRAWLER_NAME")
DATABASE_NAME = optional_arg("DATABASE_NAME", "yelp_silver_db") or "yelp_silver_db"
GLUE_ROLE_ARN = optional_arg("GLUE_ROLE_ARN")
TABLE_PREFIX = optional_arg("TABLE_PREFIX", "yelp_") or ""
RUN_CRAWLER = (optional_arg("RUN_CRAWLER", "false") or "false").lower() in {"true", "1", "yes", "y"}
WAIT_FOR_CRAWLER = (optional_arg("WAIT_FOR_CRAWLER", "true") or "true").lower() in {"true", "1", "yes", "y"}

if not BRONZE_PATH.startswith("s3://"):
    raise ValueError("BRONZE_PATH must be an s3:// URI.")
if not SILVER_PATH.startswith("s3://"):
    raise ValueError("SILVER_PATH must be an s3:// URI.")
if WRITE_MODE not in {"overwrite", "append"}:
    raise ValueError("WRITE_MODE must be either 'overwrite' or 'append'.")

OUTPUT_PARTITIONS: Optional[int] = None
if OUTPUT_PARTITIONS_RAW is not None:
    try:
        OUTPUT_PARTITIONS = int(OUTPUT_PARTITIONS_RAW)
    except ValueError as exc:
        raise ValueError("OUTPUT_PARTITIONS must be a positive integer.") from exc
    if OUTPUT_PARTITIONS <= 0:
        raise ValueError("OUTPUT_PARTITIONS must be a positive integer.")

sc = SparkContext.getOrCreate()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(JOB_NAME, required_args)

spark.conf.set("spark.sql.parquet.compression.codec", "snappy")
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

SOURCE_DATASET = "yelp_academic_dataset_business"
SOURCE_LAYER = "bronze"

EXPECTED_BUSINESS_COLUMNS = [
    "business_id", "name", "address", "city", "state", "postal_code",
    "latitude", "longitude", "stars", "review_count", "is_open",
    "attributes", "categories", "hours",
]

MANDATORY_BUSINESS_COLUMNS = [
    "business_id", "name", "latitude", "longitude",
    "stars", "review_count", "is_open",
]

DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

ATTRIBUTE_BOOLEAN_MAP: Dict[str, str] = {
    "attributes_accepts_insurance": "AcceptsInsurance",
    "attributes_by_appointment_only": "ByAppointmentOnly",
    "attributes_bike_parking": "BikeParking",
    "attributes_business_accepts_bitcoin": "BusinessAcceptsBitcoin",
    "attributes_business_accepts_credit_cards": "BusinessAcceptsCreditCards",
    "attributes_caters": "Caters",
    "attributes_coat_check": "CoatCheck",
    "attributes_dogs_allowed": "DogsAllowed",
    "attributes_drive_thru": "DriveThru",
    "attributes_good_for_dancing": "GoodForDancing",
    "attributes_good_for_kids": "GoodForKids",
    "attributes_happy_hour": "HappyHour",
    "attributes_has_tv": "HasTV",
    "attributes_open_24_hours": "Open24Hours",
    "attributes_outdoor_seating": "OutdoorSeating",
    "attributes_restaurants_counter_service": "RestaurantsCounterService",
    "attributes_restaurants_delivery": "RestaurantsDelivery",
    "attributes_restaurants_good_for_groups": "RestaurantsGoodForGroups",
    "attributes_restaurants_reservations": "RestaurantsReservations",
    "attributes_restaurants_table_service": "RestaurantsTableService",
    "attributes_restaurants_takeout": "RestaurantsTakeOut",
    "attributes_wheelchair_accessible": "WheelchairAccessible",
}

ATTRIBUTE_CATEGORICAL_MAP: Dict[str, str] = {
    "attributes_ages_allowed": "AgesAllowed",
    "attributes_alcohol": "Alcohol",
    "attributes_byob": "BYOB",
    "attributes_byob_corkage": "BYOBCorkage",
    "attributes_corkage": "Corkage",
    "attributes_noise_level": "NoiseLevel",
    "attributes_restaurants_attire": "RestaurantsAttire",
    "attributes_restaurants_price_range": "RestaurantsPriceRange2",
    "attributes_smoking": "Smoking",
    "attributes_wifi": "WiFi",
}

ATTRIBUTE_COMPLEX_MAP: Dict[str, str] = {
    "attributes_ambience_raw": "Ambience",
    "attributes_best_nights_raw": "BestNights",
    "attributes_business_parking_raw": "BusinessParking",
    "attributes_dietary_restrictions_raw": "DietaryRestrictions",
    "attributes_good_for_meal_raw": "GoodForMeal",
    "attributes_hair_specializes_in_raw": "HairSpecializesIn",
    "attributes_music_raw": "Music",
}


def normalize_text(column_expression):
    value = F.trim(column_expression.cast("string"))
    return F.when(
        value.isNull() | F.lower(value).isin("", "none", "null", "nan"),
        F.lit(None).cast("string"),
    ).otherwise(value)


def normalize_yelp_string(column_expression):
    value = normalize_text(column_expression)
    value = F.regexp_replace(value, r"^[uU](['\"])", "$1")
    value = F.regexp_replace(value, r"^['\"]|['\"]$", "")
    return F.when(
        value.isNull() | F.lower(value).isin("", "none", "null", "nan"),
        F.lit(None).cast("string"),
    ).otherwise(value)


def normalize_yelp_boolean(column_expression):
    value = F.lower(normalize_yelp_string(column_expression))
    return (
        F.when(value.isin("true", "1", "yes", "y"), F.lit(True))
        .when(value.isin("false", "0", "no", "n"), F.lit(False))
        .otherwise(F.lit(None).cast("boolean"))
    )


def stable_hash_value(df: DataFrame, column_name: str):
    """Create a stable string representation for scalar or complex columns."""
    data_type = df.schema[column_name].dataType
    column = F.col(column_name)
    if isinstance(data_type, (T.StructType, T.ArrayType, T.MapType)):
        return F.to_json(column)
    return column.cast("string")


def create_update_and_run_crawler(
    crawler_name: str,
    database_name: str,
    role_arn: str,
    s3_path: str,
    table_prefix: str = "",
    wait_for_completion: bool = True,
) -> None:
    """Create/update and optionally wait for a Glue crawler after Silver write."""
    glue = boto3.client("glue")

    try:
        glue.get_database(Name=database_name)
        print(f"Glue database already exists: {database_name}")
    except glue.exceptions.EntityNotFoundException:
        glue.create_database(
            DatabaseInput={
                "Name": database_name,
                "Description": "Silver-layer catalog for Yelp analytics",
            }
        )
        print(f"Created Glue database: {database_name}")

    targets = {"S3Targets": [{"Path": s3_path.rstrip("/") + "/"}]}
    schema_change_policy = {
        "UpdateBehavior": "UPDATE_IN_DATABASE",
        "DeleteBehavior": "LOG",
    }
    recrawl_policy = {"RecrawlBehavior": "CRAWL_EVERYTHING"}
    configuration = json.dumps({
        "Version": 1.0,
        "Grouping": {"TableGroupingPolicy": "CombineCompatibleSchemas"},
        "CrawlerOutput": {
            "Partitions": {"AddOrUpdateBehavior": "InheritFromTable"}
        },
    })

    try:
        glue.get_crawler(Name=crawler_name)
        glue.update_crawler(
            Name=crawler_name,
            Role=role_arn,
            DatabaseName=database_name,
            Targets=targets,
            TablePrefix=table_prefix,
            SchemaChangePolicy=schema_change_policy,
            RecrawlPolicy=recrawl_policy,
            Configuration=configuration,
        )
        print(f"Updated Glue crawler: {crawler_name}")
    except glue.exceptions.EntityNotFoundException:
        glue.create_crawler(
            Name=crawler_name,
            Role=role_arn,
            DatabaseName=database_name,
            Description="Catalog Yelp Business Silver Parquet output",
            Targets=targets,
            TablePrefix=table_prefix,
            SchemaChangePolicy=schema_change_policy,
            RecrawlPolicy=recrawl_policy,
            Configuration=configuration,
        )
        print(f"Created Glue crawler: {crawler_name}")

    try:
        glue.start_crawler(Name=crawler_name)
        print(f"Started Glue crawler: {crawler_name}")
    except glue.exceptions.CrawlerRunningException:
        print(f"Glue crawler is already running: {crawler_name}")

    if not wait_for_completion:
        return

    while True:
        crawler = glue.get_crawler(Name=crawler_name)["Crawler"]
        state = crawler["State"]
        print(f"Crawler state: {state}")
        if state == "READY":
            last_crawl = crawler.get("LastCrawl", {})
            status = last_crawl.get("Status")
            print(f"Last crawl status: {status}")
            if status == "FAILED":
                raise RuntimeError(last_crawl.get("ErrorMessage", "Glue crawler failed."))
            break
        time.sleep(15)


print(f"Starting Glue job: {JOB_NAME}")
print(f"Spark version: {spark.version}")
print(f"Bronze path: {BRONZE_PATH}")
print(f"Silver path: {SILVER_PATH}")

# Yelp JSON is newline-delimited JSON, not one multi-line JSON document.
df_business_bronze = (
    spark.read
    .option("multiLine", "false")
    .option("mode", "PERMISSIVE")
    .json(BRONZE_PATH)
)

if "_corrupt_record" in df_business_bronze.columns:
    corrupt_count = df_business_bronze.filter(F.col("_corrupt_record").isNotNull()).count()
    if corrupt_count > 0:
        raise ValueError(f"Input contains {corrupt_count} corrupt JSON record(s).")
    df_business_bronze = df_business_bronze.drop("_corrupt_record")

bronze_row_count = df_business_bronze.count()
if bronze_row_count == 0:
    raise ValueError("Bronze input contains zero rows. The job will not overwrite Silver.")

actual_columns = set(df_business_bronze.columns)
missing_expected_columns = sorted(set(EXPECTED_BUSINESS_COLUMNS) - actual_columns)
unexpected_columns = sorted(actual_columns - set(EXPECTED_BUSINESS_COLUMNS))
missing_mandatory_columns = sorted(set(MANDATORY_BUSINESS_COLUMNS) - actual_columns)

print(f"Bronze rows: {bronze_row_count:,}")
print(f"Missing expected columns: {missing_expected_columns}")
print(f"Unexpected source columns: {unexpected_columns}")

if missing_mandatory_columns:
    raise ValueError(f"Mandatory source columns are missing: {missing_mandatory_columns}")

# Add optional scalar columns when absent. Nested columns are handled conditionally.
df_business_silver = df_business_bronze
for optional_string_column in ["address", "city", "state", "postal_code", "categories"]:
    if optional_string_column not in df_business_silver.columns:
        df_business_silver = df_business_silver.withColumn(
            optional_string_column, F.lit(None).cast("string")
        )

# Standardize core columns.
df_business_silver = (
    df_business_silver
    .withColumn("business_id", normalize_text(F.col("business_id")))
    .withColumn("name", normalize_text(F.col("name")))
    .withColumn("address", normalize_text(F.col("address")))
    .withColumn("city", normalize_text(F.col("city")))
    .withColumn("state", F.upper(normalize_text(F.col("state"))))
    .withColumn("postal_code", normalize_text(F.col("postal_code")))
    .withColumn("latitude", F.col("latitude").cast(T.DoubleType()))
    .withColumn("longitude", F.col("longitude").cast(T.DoubleType()))
    .withColumn("stars", F.col("stars").cast(T.DoubleType()))
    .withColumn("review_count", F.col("review_count").cast(T.LongType()))
    .withColumn("is_open", F.col("is_open").cast(T.IntegerType()))
)

# Metadata and source hash. to_json is used for nested values for Glue/Spark stability.
hash_input_columns = [
    F.coalesce(stable_hash_value(df_business_silver, column_name), F.lit("∅"))
    for column_name in EXPECTED_BUSINESS_COLUMNS
    if column_name in df_business_silver.columns
]

df_business_silver = (
    df_business_silver
    .withColumn("source_record_hash", F.sha2(F.concat_ws("||", *hash_input_columns), 256))
    .withColumn("source_dataset", F.lit(SOURCE_DATASET))
    .withColumn("source_layer", F.lit(SOURCE_LAYER))
    .withColumn("pipeline_version", F.lit(PIPELINE_VERSION))
    .withColumn("silver_processed_at", F.current_timestamp())
    .withColumn("silver_processed_date", F.current_date())
)

# Duplicate ID flags without removing rows.
business_id_counts = (
    df_business_silver
    .filter(F.col("business_id").isNotNull())
    .groupBy("business_id")
    .agg(F.count(F.lit(1)).cast("long").alias("business_id_record_count"))
)

df_business_silver = (
    df_business_silver
    .join(business_id_counts, on="business_id", how="left")
    .withColumn(
        "business_id_record_count",
        F.coalesce(F.col("business_id_record_count"), F.lit(0).cast("long")),
    )
    .withColumn("_dq_missing_business_id", F.col("business_id").isNull())
    .withColumn(
        "_dq_duplicate_business_id",
        F.col("business_id").isNotNull() & (F.col("business_id_record_count") > 1),
    )
)

# Categories: retain original string and add clean array/count.
raw_categories_array = F.transform(
    F.split(F.coalesce(F.col("categories"), F.lit("")), ","),
    lambda value: F.trim(value),
)
clean_categories_array = F.filter(
    raw_categories_array,
    lambda value: value.isNotNull() & (value != ""),
)

df_business_silver = (
    df_business_silver
    .withColumn("categories_array", clean_categories_array)
    .withColumn("categories_count", F.size(F.col("categories_array")))
    .withColumn("has_categories", F.col("categories_count") > 0)
)

# Flatten selected attributes safely, even if the source attributes column is absent.
attributes_type = (
    df_business_silver.schema["attributes"].dataType
    if "attributes" in df_business_silver.columns
    else None
)
available_attribute_fields = (
    {field.name for field in attributes_type.fields}
    if isinstance(attributes_type, T.StructType)
    else set()
)

for output_column, nested_field in ATTRIBUTE_BOOLEAN_MAP.items():
    expression = (
        normalize_yelp_boolean(F.col(f"attributes.`{nested_field}`"))
        if nested_field in available_attribute_fields
        else F.lit(None).cast("boolean")
    )
    df_business_silver = df_business_silver.withColumn(output_column, expression)

for output_column, nested_field in ATTRIBUTE_CATEGORICAL_MAP.items():
    expression = (
        F.lower(normalize_yelp_string(F.col(f"attributes.`{nested_field}`")))
        if nested_field in available_attribute_fields
        else F.lit(None).cast("string")
    )
    df_business_silver = df_business_silver.withColumn(output_column, expression)

for output_column, nested_field in ATTRIBUTE_COMPLEX_MAP.items():
    expression = (
        normalize_yelp_string(F.col(f"attributes.`{nested_field}`"))
        if nested_field in available_attribute_fields
        else F.lit(None).cast("string")
    )
    df_business_silver = df_business_silver.withColumn(output_column, expression)

has_attributes_expression = (
    F.col("attributes").isNotNull()
    if "attributes" in df_business_silver.columns
    else F.lit(False)
)
df_business_silver = df_business_silver.withColumn(
    "has_attributes", has_attributes_expression
)

# Flatten hours safely.
hours_type = (
    df_business_silver.schema["hours"].dataType
    if "hours" in df_business_silver.columns
    else None
)
available_hour_fields = (
    {field.name for field in hours_type.fields}
    if isinstance(hours_type, T.StructType)
    else set()
)

for day in DAYS:
    output_column = f"hours_{day.lower()}"
    expression = (
        normalize_text(F.col(f"hours.`{day}`"))
        if day in available_hour_fields
        else F.lit(None).cast("string")
    )
    df_business_silver = df_business_silver.withColumn(output_column, expression)

hours_columns = [f"hours_{day.lower()}" for day in DAYS]
open_days_count_expression = sum(
    [F.when(F.col(column_name).isNotNull(), 1).otherwise(0) for column_name in hours_columns],
    F.lit(0),
)
open_24_hours_days_expression = sum(
    [
        F.when(
            F.regexp_replace(F.col(column_name), r"\s+", "") == "0:0-0:0",
            1,
        ).otherwise(0)
        for column_name in hours_columns
    ],
    F.lit(0),
)

df_business_silver = (
    df_business_silver
    .withColumn("open_days_count", open_days_count_expression)
    .withColumn("has_hours", F.col("open_days_count") > 0)
    .withColumn("open_24_hours_days_count", open_24_hours_days_expression)
    .withColumn("has_any_24_hour_day", F.col("open_24_hours_days_count") > 0)
    .withColumn("is_open_24_hours_every_day", F.col("open_24_hours_days_count") == 7)
)

# Analytical bands.
df_business_silver = (
    df_business_silver
    .withColumn(
        "rating_band",
        F.when(F.col("stars").isNull(), "unknown")
        .when(F.col("stars") < 2.5, "low")
        .when(F.col("stars") < 4.0, "medium")
        .otherwise("high"),
    )
    .withColumn(
        "review_volume_band",
        F.when(F.col("review_count").isNull(), "unknown")
        .when(F.col("review_count") < 10, "very_low")
        .when(F.col("review_count") < 50, "low")
        .when(F.col("review_count") < 200, "medium")
        .when(F.col("review_count") < 1000, "high")
        .otherwise("very_high"),
    )
)

# Data-quality flags are summarized into dq_issue_count and dq_status.
df_business_silver = (
    df_business_silver
    .withColumn("_dq_missing_name", F.col("name").isNull())
    .withColumn("_dq_missing_city", F.col("city").isNull())
    .withColumn("_dq_missing_state", F.col("state").isNull())
    .withColumn("_dq_missing_postal_code", F.col("postal_code").isNull())
    .withColumn(
        "_dq_missing_coordinates",
        F.col("latitude").isNull() | F.col("longitude").isNull(),
    )
    .withColumn(
        "_dq_invalid_latitude",
        F.col("latitude").isNotNull() & ~F.col("latitude").between(-90.0, 90.0),
    )
    .withColumn(
        "_dq_invalid_longitude",
        F.col("longitude").isNotNull() & ~F.col("longitude").between(-180.0, 180.0),
    )
    .withColumn(
        "_dq_invalid_stars",
        F.col("stars").isNotNull() & ~F.col("stars").between(1.0, 5.0),
    )
    .withColumn(
        "_dq_invalid_review_count",
        F.col("review_count").isNotNull() & (F.col("review_count") < 0),
    )
    .withColumn(
        "_dq_invalid_is_open",
        F.col("is_open").isNotNull() & ~F.col("is_open").isin(0, 1),
    )
    .withColumn("_dq_missing_categories", ~F.col("has_categories"))
    .withColumn("_dq_missing_attributes", ~F.col("has_attributes"))
    .withColumn("_dq_missing_hours", ~F.col("has_hours"))
)

internal_dq_columns = [
    column_name for column_name in df_business_silver.columns
    if column_name.startswith("_dq_")
]
dq_issue_count_expression = sum(
    [F.when(F.coalesce(F.col(column_name), F.lit(False)), 1).otherwise(0)
     for column_name in internal_dq_columns],
    F.lit(0),
)

df_business_silver = (
    df_business_silver
    .withColumn("dq_issue_count", dq_issue_count_expression)
    .withColumn(
        "dq_status",
        F.when(F.col("dq_issue_count") == 0, "valid")
        .when(F.col("dq_issue_count") <= 2, "warning")
        .otherwise("review_required"),
    )
)

# Remove flattened attribute fields that are null in every source row.
flattened_attribute_columns = [
    column_name for column_name in df_business_silver.columns
    if column_name.startswith("attributes_")
]
if flattened_attribute_columns:
    non_null_counts = (
        df_business_silver
        .agg(*[
            F.sum(F.when(F.col(column_name).isNotNull(), 1).otherwise(0)).alias(column_name)
            for column_name in flattened_attribute_columns
        ])
        .first()
        .asDict()
    )
    all_null_attribute_columns = [
        column_name for column_name, count_value in non_null_counts.items()
        if int(count_value or 0) == 0
    ]
    if all_null_attribute_columns:
        df_business_silver = df_business_silver.drop(*all_null_attribute_columns)
else:
    all_null_attribute_columns = []

print(f"Dropped all-null attribute columns: {all_null_attribute_columns}")

core_business_columns = [
    "business_id", "name", "address", "city", "state", "postal_code",
    "latitude", "longitude", "stars", "rating_band", "review_count",
    "review_volume_band", "is_open",
]
category_columns = [
    "categories", "categories_array", "categories_count", "has_categories",
]
final_attribute_columns = sorted([
    column_name for column_name in df_business_silver.columns
    if column_name.startswith("attributes_")
])
hours_output_columns = [
    "hours_monday", "hours_tuesday", "hours_wednesday", "hours_thursday",
    "hours_friday", "hours_saturday", "hours_sunday", "has_hours",
    "open_days_count", "open_24_hours_days_count", "has_any_24_hour_day",
    "is_open_24_hours_every_day",
]
quality_columns = ["business_id_record_count", "dq_issue_count", "dq_status"]
metadata_columns = [
    "source_record_hash", "source_dataset", "source_layer", "pipeline_version",
    "silver_processed_at", "silver_processed_date",
]

final_column_order: List[str] = (
    core_business_columns
    + category_columns
    + ["has_attributes"]
    + final_attribute_columns
    + hours_output_columns
    + quality_columns
    + metadata_columns
)
final_column_order = [
    column_name for column_name in final_column_order
    if column_name in df_business_silver.columns
]

df_business_silver_final = df_business_silver.select(*final_column_order)

final_silver_row_count = df_business_silver_final.count()
if final_silver_row_count != bronze_row_count:
    raise ValueError(
        "Row-count reconciliation failed: "
        f"Bronze={bronze_row_count}, Silver={final_silver_row_count}"
    )

print(f"Final Silver columns: {len(df_business_silver_final.columns)}")
df_business_silver_final.printSchema()
df_business_silver_final.groupBy("dq_status").count().orderBy("dq_status").show(
    truncate=False
)

output_df = (
    df_business_silver_final.coalesce(OUTPUT_PARTITIONS)
    if OUTPUT_PARTITIONS is not None
    else df_business_silver_final
)

(
    output_df.write
    .mode(WRITE_MODE)
    .format("parquet")
    .option("compression", "snappy")
    .save(SILVER_PATH)
)

# Reload persisted output before committing the Glue job.
df_verified = spark.read.parquet(SILVER_PATH)
verified_row_count = df_verified.count()
if WRITE_MODE == "overwrite" and verified_row_count != bronze_row_count:
    raise ValueError(
        "Persisted Silver verification failed: "
        f"Expected={bronze_row_count}, Actual={verified_row_count}"
    )

unexpected_persisted_columns = sorted(
    {"attributes", "hours"}.intersection(df_verified.columns)
)
if unexpected_persisted_columns:
    raise ValueError(
        f"Unexpected nested columns persisted: {unexpected_persisted_columns}"
    )

print(f"Verified persisted rows: {verified_row_count:,}")
print(f"Verified persisted columns: {len(df_verified.columns)}")
print(f"Silver output written successfully to: {SILVER_PATH}")

if RUN_CRAWLER:
    if not CRAWLER_NAME:
        raise ValueError("CRAWLER_NAME is required when RUN_CRAWLER=true.")
    if not GLUE_ROLE_ARN:
        raise ValueError("GLUE_ROLE_ARN is required when RUN_CRAWLER=true.")

    create_update_and_run_crawler(
        crawler_name=CRAWLER_NAME,
        database_name=DATABASE_NAME,
        role_arn=GLUE_ROLE_ARN,
        s3_path=SILVER_PATH,
        table_prefix=TABLE_PREFIX,
        wait_for_completion=WAIT_FOR_CRAWLER,
    )

job.commit()
