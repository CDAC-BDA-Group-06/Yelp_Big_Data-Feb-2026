# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("Yelp Sentiment Analysis") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Spark Version:", spark.version)


# ============================================================
# 2. READ DATA FROM S3
# ============================================================

file_path = (
    "s3://yelpdatasetvita/gold_layer/ml/"
    "sentiment_features/"
    "part-00000-c3f61153-e06b-4f01-88ea-69e9a133a765-c000.snappy.parquet"
)

df = spark.read.parquet(file_path)


# ============================================================
# 3. CHECK DATA
# ============================================================

df.printSchema()

print("Number of Rows:", df.count())

print("Columns:")
print(df.columns)

df.show(10, truncate=False)


# ============================================================
# 4. SELECT REQUIRED COLUMNS
# ============================================================

df = df.select(
    col("review_text"),
    col("sentiment_label"),
    col("stars")
)


# ============================================================
# 5. REMOVE NULL VALUES
# ============================================================

df = df.dropna(
    subset=[
        "review_text",
        "sentiment_label"
    ]
)

print("Rows after removing nulls:", df.count())


# ============================================================
# 6. CLEAN REVIEW TEXT
# ============================================================

df = df.withColumn(
    "clean_text",
    lower(col("review_text"))
)

# Remove URLs
df = df.withColumn(
    "clean_text",
    regexp_replace(
        col("clean_text"),
        r"http\S+|www\S+",
        ""
    )
)

# Remove HTML tags
df = df.withColumn(
    "clean_text",
    regexp_replace(
        col("clean_text"),
        r"<[^>]*>",
        ""
    )
)

# Keep only letters and spaces
df = df.withColumn(
    "clean_text",
    regexp_replace(
        col("clean_text"),
        r"[^a-zA-Z\s]",
        ""
    )
)

# Remove extra spaces
df = df.withColumn(
    "clean_text",
    trim(col("clean_text"))
)


print("Cleaned Text:")
df.select(
    "review_text",
    "clean_text"
).show(5, truncate=False)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Training Records:", train_df.count())
print("Testing Records :", test_df.count())


# ============================================================
# 8. TEXT PREPROCESSING PIPELINE
# ============================================================

label_indexer = StringIndexer(
    inputCol="sentiment_label",
    outputCol="label",
    handleInvalid="skip"
)

tokenizer = Tokenizer(
    inputCol="clean_text",
    outputCol="words"
)

stopword_remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)

hashing_tf = HashingTF(
    inputCol="filtered_words",
    outputCol="rawFeatures",
    numFeatures=20000
)

idf = IDF(
    inputCol="rawFeatures",
    outputCol="features"
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    regParam=0.0
)


# ============================================================
# 9. CREATE PIPELINE
# ============================================================

pipeline = Pipeline(
    stages=[
        label_indexer,
        tokenizer,
        stopword_remover,
        hashing_tf,
        idf,
        lr
    ]
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("Training sentiment model...")

pipeline_model = pipeline.fit(train_df)

print("Model training completed.")


# ============================================================
# 11. MAKE PREDICTIONS
# ============================================================

predictions = pipeline_model.transform(test_df)


# ============================================================
# 12. DISPLAY PREDICTIONS
# ============================================================

predictions.select(
    "clean_text",
    "sentiment_label",
    "label",
    "prediction",
    "probability"
).show(
    10,
    truncate=False
)


# ============================================================
# 13. ACCURACY
# ============================================================

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = accuracy_evaluator.evaluate(predictions)

print("Accuracy:", accuracy)


# ============================================================
# 14. F1 SCORE
# ============================================================

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

f1_score = f1_evaluator.evaluate(predictions)

print("F1 Score:", f1_score)


# ============================================================
# 15. SAVE MODEL TO S3
# ============================================================

model_path = (
    "s3://yelpdatasetvita/"
    "gold_layer/sentiment_analysis_Model/"
)

pipeline_model.write() \
    .overwrite() \
    .save(model_path)


# ============================================================
# 16. SAVE PREDICTIONS TO S3
# ============================================================

output_path = (
    "s3://yelpdatasetvita/"
    "gold_layer/sentiment_analysis_output/"
)

predictions.write \
    .mode("overwrite") \
    .parquet(output_path)


# ============================================================
# 17. STOP SPARK
# ============================================================

spark.stop()
