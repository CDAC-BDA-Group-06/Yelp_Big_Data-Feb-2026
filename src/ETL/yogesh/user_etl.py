import sys
import logging

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext

from pyspark.sql.functions import (
    col,
    trim,
    to_timestamp,
    regexp_replace
)

from pyspark.sql.types import (
    StringType,
    IntegerType,
    DoubleType
)

# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Initialize Glue Job
# ------------------------------------------------------------

args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ------------------------------------------------------------
# S3 Paths
# ------------------------------------------------------------

INPUT_PATH = "s3://layerbronze/yelp_dataset/yelp_academic_dataset_user.json"
OUTPUT_PATH = "s3://silveruser/silver/"

try:

    # --------------------------------------------------------
    # Read JSON
    # --------------------------------------------------------

    logger.info("Reading Bronze JSON Data...")

    df = spark.read.json(INPUT_PATH)

    raw_count = df.count()

    logger.info(f"Raw Records : {raw_count}")

    # --------------------------------------------------------
    # Remove Completely Empty Rows
    # --------------------------------------------------------

    logger.info("Removing Empty Rows...")

    df = df.dropna(how="all")

    # --------------------------------------------------------
    # Remove Duplicate Users
    # --------------------------------------------------------

    logger.info("Removing Duplicate user_id...")

    df = df.dropDuplicates(["user_id"])

    # --------------------------------------------------------
    # Remove Null Primary Key
    # --------------------------------------------------------

    logger.info("Removing Null user_id...")

    df = df.filter(col("user_id").isNotNull())

    # --------------------------------------------------------
    # Trim Spaces From Every String Column
    # --------------------------------------------------------

    logger.info("Trimming String Columns...")

    for field in df.schema.fields:

        if isinstance(field.dataType, StringType):

            df = df.withColumn(
                field.name,
                trim(col(field.name))
            )

    # --------------------------------------------------------
    # Clean Friends Column
    # --------------------------------------------------------

    if "friends" in df.columns:

        logger.info("Cleaning Friends Column...")

        df = df.withColumn(
            "friends",
            regexp_replace(col("friends"), "\\s+", "")
        )

    # --------------------------------------------------------
    # Convert Timestamp
    # --------------------------------------------------------

    logger.info("Converting Timestamp...")

    df = df.withColumn(
        "yelping_since",
        to_timestamp(
            col("yelping_since"),
            "yyyy-MM-dd HH:mm:ss"
        )
    )

    # Remove Invalid Dates

    df = df.filter(
        col("yelping_since").isNotNull()
    )

    # --------------------------------------------------------
    # Integer Columns
    # --------------------------------------------------------

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

    logger.info("Casting Integer Columns...")

    for c in integer_columns:

        if c in df.columns:

            df = df.withColumn(
                c,
                col(c).cast(IntegerType())
            )

    # --------------------------------------------------------
    # Double Columns
    # --------------------------------------------------------

    logger.info("Casting Double Columns...")

    if "average_stars" in df.columns:

        df = df.withColumn(
            "average_stars",
            col("average_stars").cast(DoubleType())
        )

    # --------------------------------------------------------
    # Fill Null Values
    # --------------------------------------------------------

    logger.info("Replacing Null Values...")

    fill_dictionary = {

        "review_count":0,
        "useful":0,
        "funny":0,
        "cool":0,
        "fans":0,
        "average_stars":0.0,

        "compliment_hot":0,
        "compliment_more":0,
        "compliment_profile":0,
        "compliment_cute":0,
        "compliment_list":0,
        "compliment_note":0,
        "compliment_plain":0,
        "compliment_cool":0,
        "compliment_funny":0,
        "compliment_writer":0,
        "compliment_photos":0

    }

    df = df.fillna(fill_dictionary)

    # --------------------------------------------------------
    # Repartition
    # --------------------------------------------------------

    logger.info("Repartitioning Data...")

    df = df.repartition(4)

    # --------------------------------------------------------
    # Final Count
    # --------------------------------------------------------

    final_count = df.count()

    logger.info(f"Silver Records : {final_count}")

    # --------------------------------------------------------
    # Write Parquet
    # --------------------------------------------------------

    logger.info("Writing Parquet Files...")

    (
        df.write
        .mode("overwrite")
        .option("compression","snappy")
        .parquet(OUTPUT_PATH)
    )

    logger.info("Parquet Files Written Successfully.")

    job.commit()

except Exception as e:

    logger.error(f"Glue Job Failed : {str(e)}")

    raise