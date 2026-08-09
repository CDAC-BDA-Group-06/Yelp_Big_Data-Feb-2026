#!/usr/bin/env python3
"""Yelp content-based recommender: all training and inference code in one file.

Run with spark-submit and one command:

    spark-submit Yelp_Content_Recommender_All_In_One.py check
    spark-submit Yelp_Content_Recommender_All_In_One.py train --model-version v2
    spark-submit Yelp_Content_Recommender_All_In_One.py validate --model-version v2
    spark-submit Yelp_Content_Recommender_All_In_One.py recommend \
        --model-version v2 --categories "pizza,italian" --state PA --top-k 10

This single script replaces the separate Python jobs and support modules from
Yelp_Content_Recommender_EMR_Scripts.zip. No --py-files package is required.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise RuntimeError(
        "NumPy is missing. Install it cluster-wide on EMR before running "
        "this spark-submit job."
    ) from error

from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, Normalizer
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window


# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_S3_BUCKET = 'yelpdataset-project'
DEFAULT_MODEL_VERSION = 'v1'
DEFAULT_OUTPUT_PREFIX = 'gold_layer/ml/content_based_recommender_model'
COLUMN_ALIASES = {'business_id': ['business_id', 'businessid'], 'name': ['name', 'business_name'], 'categories': ['categories', 'category'], 'city': ['city'], 'state': ['state'], 'latitude': ['latitude', 'lat'], 'longitude': ['longitude', 'lon', 'lng'], 'stars': ['stars', 'business_avg_rating', 'avg_rating'], 'review_count': ['review_count', 'business_review_count'], 'is_open': ['is_open', 'open_status'], 'attributes_restaurantspricerange2': ['attributes_restaurantspricerange2', 'attributes_restaurants_price_range2', 'restaurantspricerange2', 'price_range'], 'attributes_wifi': ['attributes_wifi', 'wifi', 'has_wifi'], 'attributes_outdoorseating': ['attributes_outdoorseating', 'attributes_outdoor_seating', 'outdoorseating', 'outdoor_seating']}
REQUIRED_INPUT_COLUMNS = {'business_id', 'categories', 'city', 'state', 'stars', 'review_count', 'is_open', 'attributes_restaurantspricerange2', 'attributes_wifi', 'attributes_outdoorseating'}
OPTIONAL_COLUMN_TYPES = {'name': 'string', 'latitude': 'double', 'longitude': 'double'}
BUSINESS_SERVING_COLUMNS = ['business_id', 'name', 'categories', 'category_tokens', 'city', 'city_clean', 'state', 'state_clean', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'is_open_clean', 'price_range_numeric', 'wifi_clean', 'outdoor_seating_clean', 'category_vector_normalized']
REQUIRED_SERVING_COLUMNS = {'business_id', 'categories', 'category_tokens', 'city', 'state', 'stars', 'review_count', 'is_open_clean', 'price_range_numeric', 'wifi_clean', 'outdoor_seating_clean', 'category_vector_normalized'}
INITIAL_WEIGHTS = {'category': 0.5, 'attributes': 0.25, 'rating': 0.15, 'review_confidence': 0.1}
WEIGHT_GRID = [{'category': 0.55, 'attributes': 0.2, 'rating': 0.15, 'review_confidence': 0.1}, {'category': 0.5, 'attributes': 0.25, 'rating': 0.15, 'review_confidence': 0.1}, {'category': 0.6, 'attributes': 0.15, 'rating': 0.15, 'review_confidence': 0.1}, {'category': 0.5, 'attributes': 0.2, 'rating': 0.2, 'review_confidence': 0.1}, {'category': 0.45, 'attributes': 0.3, 'rating': 0.15, 'review_confidence': 0.1}, {'category': 0.5, 'attributes': 0.25, 'rating': 0.1, 'review_confidence': 0.15}]
DEFAULT_USER_QUERY = {'categories': ['pizza', 'italian'], 'location': {'city': None, 'state': None}, 'attributes': {'price_range': 2.0, 'wifi': 'free', 'outdoor_seating': True}, 'minimum_rating': 4.0, 'top_k': 10, 'exclude_business_ids': []}

@dataclass(frozen=True)
class EvaluationConfig:
    random_seed: int = 42
    minimum_rating: float = 4.0
    minimum_relevant_businesses: int = 5
    maximum_relevant_businesses: int = 500
    maximum_queries_per_type: int = 20
    top_k: int = 10
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

@dataclass(frozen=True)
class ModelPaths:
    bucket: str
    model_version: str
    output_prefix: str
    output_root: str
    business_vectors: str
    validation_results: str
    test_metrics: str
    sample_recommendations: str
    model_configuration: str
    model_manifest: str
    category_vectorizer: str
    normalizer: str

def build_model_paths(bucket: str=DEFAULT_S3_BUCKET, model_version: str=DEFAULT_MODEL_VERSION, output_prefix: str=DEFAULT_OUTPUT_PREFIX) -> ModelPaths:
    normalized_prefix = output_prefix.strip('/')
    output_root = f's3://{bucket}/{normalized_prefix}/{model_version}/'
    artifacts_root = output_root + 'model_artifacts/'
    return ModelPaths(bucket=bucket, model_version=model_version, output_prefix=normalized_prefix, output_root=output_root, business_vectors=output_root + 'business_feature_vectors/', validation_results=output_root + 'evaluation/validation_results/', test_metrics=output_root + 'evaluation/test_metrics/', sample_recommendations=output_root + 'sample_recommendations/', model_configuration=output_root + 'model_configuration/', model_manifest=output_root + 'model_manifest/', category_vectorizer=artifacts_root + 'category_vectorizer_model/', normalizer=artifacts_root + 'normalizer/')

def default_business_input_candidates(bucket: str) -> List[str]:
    return [f's3://{bucket}/gold_layer/ml/content_based_filtering/content.parquet', f's3://{bucket}/gold_layer/ml/content_based_filtering/content/', f's3://{bucket}/gold_layer/ml/content_based_filtering/', f's3://{bucket}/gold_layer/ml/content_based_data/']

def validate_weights(weights: Dict[str, float]) -> None:
    expected = {'category', 'attributes', 'rating', 'review_confidence'}
    missing = expected - set(weights)
    if missing:
        raise ValueError(f'Missing scoring weights: {sorted(missing)}')
    total = sum((float(weights[name]) for name in expected))
    if abs(total - 1.0) > 1e-09:
        raise ValueError(f'Scoring weights must sum to 1.0; received {total}.')

# ==============================================================================
# S3 and Hadoop filesystem helpers
# ==============================================================================

def path_exists(spark, path: str) -> bool:
    j_path = spark._jvm.org.apache.hadoop.fs.Path(path)
    file_system = j_path.getFileSystem(spark._jsc.hadoopConfiguration())
    return bool(file_system.exists(j_path))

def delete_path_if_exists(spark, path: str) -> None:
    j_path = spark._jvm.org.apache.hadoop.fs.Path(path)
    file_system = j_path.getFileSystem(spark._jsc.hadoopConfiguration())
    if file_system.exists(j_path):
        deleted = bool(file_system.delete(j_path, True))
        if not deleted and file_system.exists(j_path):
            raise PermissionError(f'Could not delete existing path: {path}')

def write_json_document(spark, document: Dict[str, Any], output_path: str) -> None:
    payload = json.dumps(document, sort_keys=True)
    spark.createDataFrame([(payload,)], ['value']).coalesce(1).write.mode('overwrite').text(output_path)

def read_json_document(spark, input_path: str) -> Dict[str, Any]:
    row = spark.read.text(input_path).first()
    if row is None or row['value'] is None:
        raise ValueError(f'No JSON document found at {input_path}')
    return json.loads(row['value'])

def resolve_first_existing_path(spark, candidates: Iterable[str]) -> str:
    inspected = []
    for candidate in candidates:
        inspected.append(candidate)
        try:
            if path_exists(spark, candidate):
                return candidate
        except Exception as error:
            print(f'Could not inspect {candidate}: {error}')
    raise FileNotFoundError('None of the configured input paths exists. Inspected: ' + ', '.join(inspected))

def prepare_output_root(spark, output_root: str, overwrite: bool) -> None:
    if not output_root.startswith('s3://'):
        raise ValueError(f'Output root must be an S3 path: {output_root}')
    if path_exists(spark, output_root):
        if not overwrite:
            raise FileExistsError(f'Model output already exists: {output_root}. Use a new --model-version or pass --overwrite.')
        print('Deleting existing model version:', output_root)
        delete_path_if_exists(spark, output_root)

def save_ml_artifact(writer_factory: Callable[[], Any], output_path: str, overwrite: bool) -> None:
    writer = writer_factory()
    if overwrite:
        writer = writer.overwrite()
    writer.save(output_path)

def save_parquet(dataframe, output_path: str, overwrite: bool, partitions: Optional[int]=None) -> None:
    output_df = dataframe
    if partitions is not None:
        output_df = output_df.repartition(int(partitions))
    mode = 'overwrite' if overwrite else 'errorifexists'
    output_df.write.mode(mode).option('compression', 'snappy').parquet(output_path)

def verify_artifact_paths(spark, artifact_paths: Dict[str, str]) -> None:
    missing = {name: path for name, path in artifact_paths.items() if not path_exists(spark, path)}
    if missing:
        raise FileNotFoundError(f'Missing saved artifacts: {missing}')

# ==============================================================================
# Business data preparation
# ==============================================================================

def canonicalize_columns(dataframe, aliases: Dict[str, Iterable[str]]=COLUMN_ALIASES):
    result = dataframe
    current_lookup = {column_name.lower(): column_name for column_name in result.columns}
    for canonical_name, candidate_names in aliases.items():
        if canonical_name in result.columns:
            continue
        matched_name = None
        for candidate_name in candidate_names:
            actual_name = current_lookup.get(candidate_name.lower())
            if actual_name is not None:
                matched_name = actual_name
                break
        if matched_name is not None and matched_name != canonical_name:
            result = result.withColumnRenamed(matched_name, canonical_name)
            current_lookup[canonical_name.lower()] = canonical_name
    return result

def preflight_business_data(dataframe):
    business_df = canonicalize_columns(dataframe)
    missing_input_columns = sorted(REQUIRED_INPUT_COLUMNS - set(business_df.columns))
    if missing_input_columns:
        raise ValueError('Missing required input columns: ' + ', '.join(missing_input_columns))
    for optional_column, data_type in OPTIONAL_COLUMN_TYPES.items():
        if optional_column not in business_df.columns:
            business_df = business_df.withColumn(optional_column, F.lit(None).cast(data_type))
    business_row_count = business_df.count()
    if business_row_count == 0:
        raise ValueError('The business input dataset contains zero rows.')
    null_id_count = business_df.filter(F.col('business_id').isNull()).limit(1).count()
    if null_id_count > 0:
        raise ValueError('The business input dataset contains a null business_id.')
    duplicate_id_count = business_df.groupBy('business_id').count().filter(F.col('count') > 1).limit(1).count()
    if duplicate_id_count > 0:
        raise ValueError('The business input dataset contains duplicate business_id values.')
    return (business_df, business_row_count)

def handle_feature_nulls(business_df):
    business_df = business_df.fillna({'categories': 'Unknown', 'city': 'Unknown', 'state': 'Unknown', 'attributes_restaurantspricerange2': 'Unknown', 'attributes_wifi': 'Unknown', 'attributes_outdoorseating': 'Unknown'})
    return business_df.withColumn('stars', F.coalesce(F.col('stars').cast('double'), F.lit(0.0))).withColumn('review_count', F.greatest(F.coalesce(F.col('review_count').cast('long'), F.lit(0)), F.lit(0)))

def add_category_tokens(business_df):
    business_df = business_df.withColumn('category_tokens', F.array_distinct(F.array_remove(F.transform(F.split(F.lower(F.col('categories')), ','), lambda category: F.trim(category)), '')))
    return business_df.withColumn('category_tokens', F.when(F.size(F.col('category_tokens')) > 0, F.col('category_tokens')).otherwise(F.array(F.lit('unknown'))))

def fit_category_models(business_df):
    vectorizer = CountVectorizer(inputCol='category_tokens', outputCol='category_vector', binary=True, minDF=2.0)
    vectorizer_model = vectorizer.fit(business_df)
    if len(vectorizer_model.vocabulary) == 0:
        raise ValueError('The fitted category vocabulary is empty.')
    transformed_df = vectorizer_model.transform(business_df)
    normalizer = Normalizer(inputCol='category_vector', outputCol='category_vector_normalized', p=2.0)
    transformed_df = normalizer.transform(transformed_df)
    return (transformed_df, vectorizer_model, normalizer)

def clean_business_attributes(business_df):
    price_text = F.lower(F.trim(F.coalesce(F.col('attributes_restaurantspricerange2').cast('string'), F.lit('unknown'))))
    wifi_text = F.lower(F.trim(F.coalesce(F.col('attributes_wifi').cast('string'), F.lit('unknown'))))
    wifi_text = F.regexp_replace(F.regexp_replace(wifi_text, "u'", ''), "'", '')
    outdoor_text = F.lower(F.trim(F.coalesce(F.col('attributes_outdoorseating').cast('string'), F.lit('unknown'))))
    outdoor_text = F.regexp_replace(F.regexp_replace(outdoor_text, "u'", ''), "'", '')
    open_text = F.lower(F.trim(F.col('is_open').cast('string')))
    return business_df.withColumn('price_range_numeric', F.when(F.regexp_extract(price_text, '([1-4])', 1) != '', F.regexp_extract(price_text, '([1-4])', 1).cast('double')).otherwise(F.lit(None).cast('double'))).withColumn('wifi_clean', F.when(wifi_text.contains('free'), F.lit('free')).when(wifi_text.contains('paid'), F.lit('paid')).when(wifi_text.isin('no', 'none', 'false', '0'), F.lit('no')).otherwise(F.lit('unknown'))).withColumn('outdoor_seating_clean', F.when(outdoor_text.isin('true', '1', 'yes'), F.lit('true')).when(outdoor_text.isin('false', '0', 'no'), F.lit('false')).otherwise(F.lit('unknown'))).withColumn('is_open_clean', F.when(open_text.isin('1', 'true', 'yes', 'open'), F.lit(1)).otherwise(F.lit(0))).withColumn('city_clean', F.lower(F.trim(F.col('city')))).withColumn('state_clean', F.upper(F.trim(F.col('state'))))

def build_serving_dataframe(business_df, expected_row_count: int):
    missing_columns = sorted(set(BUSINESS_SERVING_COLUMNS) - set(business_df.columns))
    if missing_columns:
        raise ValueError('Cannot create serving data. Missing columns: ' + ', '.join(missing_columns))
    serving_df = business_df.select(*BUSINESS_SERVING_COLUMNS).cache()
    serving_row_count = serving_df.count()
    if serving_row_count != expected_row_count:
        raise ValueError('Serving row count does not match the validated input row count.')
    missing_required = sorted(REQUIRED_SERVING_COLUMNS - set(serving_df.columns))
    if missing_required:
        raise ValueError('Serving data is missing required columns: ' + ', '.join(missing_required))
    if serving_df.filter(F.col('business_id').isNull()).limit(1).count() > 0:
        raise ValueError('Serving data contains a null business_id.')
    return (serving_df, serving_row_count)

def prepare_business_features(raw_business_df):
    business_df, business_row_count = preflight_business_data(raw_business_df)
    business_df = handle_feature_nulls(business_df)
    business_df = add_category_tokens(business_df)
    business_df, vectorizer_model, normalizer = fit_category_models(business_df)
    business_df = clean_business_attributes(business_df)
    serving_df, serving_row_count = build_serving_dataframe(business_df, business_row_count)
    return {'business_df': business_df, 'business_serving_df': serving_df, 'business_row_count': business_row_count, 'serving_row_count': serving_row_count, 'category_vectorizer_model': vectorizer_model, 'normalizer': normalizer}

# ==============================================================================
# Recommendation engine
# ==============================================================================

def clean_text(value: Any, uppercase: bool=False) -> Optional[str]:
    if value is None:
        return None
    cleaned_value = str(value).strip()
    if cleaned_value == '':
        return None
    return cleaned_value.upper() if uppercase else cleaned_value.lower()

def clean_boolean_attribute(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 'true' if value else 'false'
    cleaned_value = str(value).strip().lower()
    true_values = {'true', 'yes', '1', 'available'}
    false_values = {'false', 'no', '0', 'not available'}
    if cleaned_value in true_values:
        return 'true'
    if cleaned_value in false_values:
        return 'false'
    return cleaned_value

def clean_user_query(user_query: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(user_query, dict):
        raise TypeError('The user query must be a dictionary.')
    cleaned_categories = []
    for category in user_query.get('categories', []):
        cleaned_category = clean_text(category)
        if cleaned_category is not None:
            cleaned_categories.append(cleaned_category)
    cleaned_categories = list(dict.fromkeys(cleaned_categories))
    location = user_query.get('location', {}) or {}
    cleaned_location = {'city': clean_text(location.get('city')), 'state': clean_text(location.get('state'), uppercase=True)}
    cleaned_attributes = {}
    for attribute_name, attribute_value in (user_query.get('attributes', {}) or {}).items():
        cleaned_name = clean_text(attribute_name)
        if cleaned_name is None:
            continue
        if cleaned_name == 'price_range':
            if attribute_value is not None:
                price_value = float(attribute_value)
                if price_value not in {1.0, 2.0, 3.0, 4.0}:
                    raise ValueError('price_range must be 1, 2, 3, or 4.')
                cleaned_attributes[cleaned_name] = price_value
        elif cleaned_name == 'outdoor_seating':
            cleaned_value = clean_boolean_attribute(attribute_value)
            if cleaned_value not in {None, 'true', 'false'}:
                raise ValueError('outdoor_seating must be true/false or yes/no.')
            if cleaned_value is not None:
                cleaned_attributes[cleaned_name] = cleaned_value
        elif cleaned_name == 'wifi':
            cleaned_value = clean_text(attribute_value)
            if cleaned_value not in {None, 'free', 'paid', 'no'}:
                raise ValueError('wifi must be free, paid, or no.')
            if cleaned_value is not None:
                cleaned_attributes[cleaned_name] = cleaned_value
        else:
            raise ValueError(f"Unsupported attribute '{attribute_name}'. Supported attributes: price_range, wifi, outdoor_seating.")
    minimum_rating = user_query.get('minimum_rating')
    if minimum_rating is not None:
        minimum_rating = float(minimum_rating)
        if minimum_rating < 1.0 or minimum_rating > 5.0:
            raise ValueError('minimum_rating must be between 1.0 and 5.0.')
    top_k = int(user_query.get('top_k', 10))
    if top_k <= 0 or top_k > 100:
        raise ValueError('top_k must be between 1 and 100.')
    exclude_business_ids = [str(value).strip() for value in user_query.get('exclude_business_ids', []) if value is not None and str(value).strip()]
    return {'categories': cleaned_categories, 'location': cleaned_location, 'attributes': cleaned_attributes, 'minimum_rating': minimum_rating, 'top_k': top_k, 'exclude_business_ids': list(dict.fromkeys(exclude_business_ids))}

@F.udf(returnType=DoubleType())
def vector_dot_product(left_vector, right_vector):
    if left_vector is None or right_vector is None:
        return 0.0
    return float(left_vector.dot(right_vector))

def recommend_businesses(user_query, business_features_df, vectorizer_model, normalizer_model, weights):
    """Return a lazy Spark DataFrame containing ranked recommendations."""
    validate_weights(weights)
    cleaned_query = clean_user_query(user_query)
    spark = business_features_df.sparkSession
    query_schema = StructType([StructField('category_tokens', ArrayType(StringType()), nullable=False)])
    query_category_df = spark.createDataFrame([(cleaned_query['categories'],)], schema=query_schema)
    query_category_df = vectorizer_model.transform(query_category_df)
    query_category_df = normalizer_model.transform(query_category_df)
    query_vector = query_category_df.select('category_vector_normalized').first()['category_vector_normalized']
    query_vector_df = spark.createDataFrame([(query_vector,)], StructType([StructField('query_category_vector', VectorUDT(), nullable=False)]))
    candidates_df = business_features_df.filter(F.col('is_open_clean') == 1)
    city_value = cleaned_query['location']['city']
    state_value = cleaned_query['location']['state']
    minimum_rating = cleaned_query['minimum_rating']
    if city_value is not None:
        candidates_df = candidates_df.filter(F.col('city_clean') == city_value)
    if state_value is not None:
        candidates_df = candidates_df.filter(F.col('state_clean') == state_value)
    if minimum_rating is not None:
        candidates_df = candidates_df.filter(F.col('stars') >= minimum_rating)
    excluded_ids = cleaned_query['exclude_business_ids']
    if excluded_ids:
        candidates_df = candidates_df.filter(~F.col('business_id').isin(excluded_ids))
    requested_attributes = cleaned_query['attributes']
    attribute_match_expressions = []
    if 'price_range' in requested_attributes:
        attribute_match_expressions.append(F.when(F.col('price_range_numeric') == F.lit(requested_attributes['price_range']), F.lit(1)).otherwise(F.lit(0)))
    if 'wifi' in requested_attributes:
        attribute_match_expressions.append(F.when(F.col('wifi_clean') == F.lit(requested_attributes['wifi']), F.lit(1)).otherwise(F.lit(0)))
    if 'outdoor_seating' in requested_attributes:
        attribute_match_expressions.append(F.when(F.col('outdoor_seating_clean') == F.lit(requested_attributes['outdoor_seating']), F.lit(1)).otherwise(F.lit(0)))
    requested_attribute_count = len(attribute_match_expressions)
    matched_attribute_count_expression = F.lit(0)
    for expression in attribute_match_expressions:
        matched_attribute_count_expression = matched_attribute_count_expression + expression
    max_review_log = business_features_df.select(F.max(F.log1p(F.col('review_count'))).alias('max_review_log')).first()['max_review_log'] or 1.0
    scored_df = candidates_df.crossJoin(F.broadcast(query_vector_df)).withColumn('category_match_score', vector_dot_product(F.col('category_vector_normalized'), F.col('query_category_vector'))).withColumn('requested_attribute_count', F.lit(requested_attribute_count)).withColumn('matched_attribute_count', matched_attribute_count_expression).withColumn('attribute_match_score', F.when(F.col('requested_attribute_count') > 0, F.col('matched_attribute_count') / F.col('requested_attribute_count')).otherwise(F.lit(0.0))).withColumn('all_requested_attributes_matched', F.when((F.col('requested_attribute_count') == 0) | (F.col('matched_attribute_count') == F.col('requested_attribute_count')), F.lit(1)).otherwise(F.lit(0))).withColumn('rating_score', F.col('stars') / F.lit(5.0)).withColumn('review_confidence_score', F.log1p(F.col('review_count')) / F.lit(float(max_review_log))).withColumn('final_score', F.col('category_match_score') * F.lit(float(weights['category'])) + F.col('attribute_match_score') * F.lit(float(weights['attributes'])) + F.col('rating_score') * F.lit(float(weights['rating'])) + F.col('review_confidence_score') * F.lit(float(weights['review_confidence']))).withColumn('attribute_match_type', F.when(F.col('requested_attribute_count') == 0, F.lit('No attributes requested')).when(F.col('all_requested_attributes_matched') == 1, F.lit('Exact attribute match')).when(F.col('matched_attribute_count') > 0, F.lit('Partial attribute match')).otherwise(F.lit('No attribute match')))
    ranking_window = Window.orderBy(F.col('all_requested_attributes_matched').desc(), F.col('final_score').desc(), F.col('stars').desc(), F.col('review_count').desc(), F.col('business_id').asc())
    return scored_df.withColumn('recommendation_rank', F.row_number().over(ranking_window)).filter(F.col('recommendation_rank') <= F.lit(cleaned_query['top_k'])).withColumn('category_match_percentage', F.round(F.col('category_match_score') * 100, 2)).withColumn('attribute_match_percentage', F.round(F.col('attribute_match_score') * 100, 2)).withColumn('final_score_percentage', F.round(F.col('final_score') * 100, 2)).orderBy('recommendation_rank')

# ==============================================================================
# Offline evaluation and tuning
# ==============================================================================

def _limit_query_type(dataframe, query_type, order_columns, config):
    return dataframe.filter((F.col('relevant_business_count') >= config.minimum_relevant_businesses) & (F.col('relevant_business_count') <= config.maximum_relevant_businesses)).orderBy(*order_columns).limit(config.maximum_queries_per_type).withColumn('query_type', F.lit(query_type))

def create_evaluation_queries(business_serving_df, config: EvaluationConfig):
    eligible_business_df = business_serving_df.filter(F.col('is_open_clean') == 1).filter(F.col('stars') >= F.lit(config.minimum_rating)).filter(F.size(F.col('category_tokens')) > 0)
    exploded_category_df = eligible_business_df.select('business_id', 'state_clean', 'price_range_numeric', 'wifi_clean', 'outdoor_seating_clean', F.explode('category_tokens').alias('query_category')).filter(~F.col('query_category').isin('unknown', 'restaurants'))
    category_state_queries_df = _limit_query_type(exploded_category_df.groupBy('query_category', 'state_clean').agg(F.countDistinct('business_id').alias('relevant_business_count')).withColumn('query_price_range', F.lit(None).cast('double')).withColumn('query_wifi', F.lit(None).cast('string')).withColumn('query_outdoor_seating', F.lit(None).cast('string')), 'category_state', [F.desc('relevant_business_count'), F.asc('state_clean'), F.asc('query_category')], config)
    category_price_queries_df = _limit_query_type(exploded_category_df.filter(F.col('price_range_numeric').isNotNull()).groupBy('query_category', 'state_clean', 'price_range_numeric').agg(F.countDistinct('business_id').alias('relevant_business_count')).withColumnRenamed('price_range_numeric', 'query_price_range').withColumn('query_wifi', F.lit(None).cast('string')).withColumn('query_outdoor_seating', F.lit(None).cast('string')), 'category_state_price', [F.desc('relevant_business_count'), F.asc('state_clean'), F.asc('query_category')], config)
    category_wifi_queries_df = _limit_query_type(exploded_category_df.filter(~F.col('wifi_clean').isin('unknown')).groupBy('query_category', 'state_clean', 'wifi_clean').agg(F.countDistinct('business_id').alias('relevant_business_count')).withColumnRenamed('wifi_clean', 'query_wifi').withColumn('query_price_range', F.lit(None).cast('double')).withColumn('query_outdoor_seating', F.lit(None).cast('string')), 'category_state_wifi', [F.desc('relevant_business_count'), F.asc('state_clean'), F.asc('query_category')], config)
    category_outdoor_queries_df = _limit_query_type(exploded_category_df.filter(~F.col('outdoor_seating_clean').isin('unknown')).groupBy('query_category', 'state_clean', 'outdoor_seating_clean').agg(F.countDistinct('business_id').alias('relevant_business_count')).withColumnRenamed('outdoor_seating_clean', 'query_outdoor_seating').withColumn('query_price_range', F.lit(None).cast('double')).withColumn('query_wifi', F.lit(None).cast('string')), 'category_state_outdoor', [F.desc('relevant_business_count'), F.asc('state_clean'), F.asc('query_category')], config)
    query_columns = ['query_type', 'query_category', 'state_clean', 'query_price_range', 'query_wifi', 'query_outdoor_seating', 'relevant_business_count']
    evaluation_queries_df = category_state_queries_df.select(*query_columns).unionByName(category_price_queries_df.select(*query_columns)).unionByName(category_wifi_queries_df.select(*query_columns)).unionByName(category_outdoor_queries_df.select(*query_columns)).withColumn('query_id', F.sha2(F.concat_ws('|', F.col('query_type'), F.col('query_category'), F.col('state_clean'), F.coalesce(F.col('query_price_range').cast('string'), F.lit('')), F.coalesce(F.col('query_wifi'), F.lit('')), F.coalesce(F.col('query_outdoor_seating'), F.lit(''))), 256)).dropDuplicates(['query_id']).cache()
    query_count = evaluation_queries_df.count()
    if query_count < 10:
        raise ValueError(f'Only {query_count} evaluation queries were created. Reduce minimum_relevant_businesses or inspect the input data.')
    return (evaluation_queries_df, query_count)

def split_evaluation_queries(evaluation_queries_df, evaluation_query_count: int, config: EvaluationConfig):
    ratio_total = config.train_ratio + config.validation_ratio + config.test_ratio
    if abs(ratio_total - 1.0) > 1e-09:
        raise ValueError('Train, validation, and test ratios must sum to 1.0.')
    split_window = Window.orderBy(F.rand(config.random_seed), F.col('query_id'))
    ranked_queries_df = evaluation_queries_df.withColumn('split_rank', F.row_number().over(split_window))
    train_end = max(1, int(evaluation_query_count * config.train_ratio))
    validation_end = max(train_end + 1, int(evaluation_query_count * (config.train_ratio + config.validation_ratio)))
    validation_end = min(validation_end, evaluation_query_count - 1)
    split_queries_df = ranked_queries_df.withColumn('dataset_split', F.when(F.col('split_rank') <= train_end, F.lit('train')).when(F.col('split_rank') <= validation_end, F.lit('validation')).otherwise(F.lit('test'))).cache()
    split_counts = {row['dataset_split']: row['count'] for row in split_queries_df.groupBy('dataset_split').count().collect()}
    for split_name in ('train', 'validation', 'test'):
        if split_counts.get(split_name, 0) == 0:
            raise ValueError(f'{split_name.title()} query split is empty.')
    return (split_queries_df, split_counts)

def build_evaluation_candidates(business_serving_df, split_queries_df, vectorizer_model, normalizer_model, config: EvaluationConfig):
    query_vector_input_df = split_queries_df.withColumn('category_tokens', F.array(F.col('query_category')))
    query_vector_df = vectorizer_model.transform(query_vector_input_df)
    query_vector_df = normalizer_model.transform(query_vector_df)
    query_vector_df = query_vector_df.withColumnRenamed('category_vector_normalized', 'query_category_vector_normalized')
    max_review_log = business_serving_df.select(F.max(F.log1p(F.col('review_count'))).alias('max_review_log')).first()['max_review_log'] or 1.0
    query_side_df = query_vector_df.select('query_id', 'query_type', 'dataset_split', 'query_category', 'state_clean', 'query_price_range', 'query_wifi', 'query_outdoor_seating', 'query_category_vector_normalized')
    business_side_df = business_serving_df.select('business_id', 'category_tokens', F.col('state_clean').alias('business_state_clean'), 'stars', 'review_count', 'is_open_clean', 'price_range_numeric', 'wifi_clean', 'outdoor_seating_clean', 'category_vector_normalized')
    evaluation_candidates_df = business_side_df.filter(F.col('is_open_clean') == 1).join(F.broadcast(query_side_df), F.col('business_state_clean') == F.col('state_clean'), 'inner').withColumn('category_match_score', vector_dot_product(F.col('category_vector_normalized'), F.col('query_category_vector_normalized'))).withColumn('requested_attribute_count', F.col('query_price_range').isNotNull().cast('int') + F.col('query_wifi').isNotNull().cast('int') + F.col('query_outdoor_seating').isNotNull().cast('int')).withColumn('matched_attribute_count', F.when(F.col('query_price_range').isNotNull() & (F.col('price_range_numeric') == F.col('query_price_range')), F.lit(1)).otherwise(F.lit(0)) + F.when(F.col('query_wifi').isNotNull() & (F.col('wifi_clean') == F.col('query_wifi')), F.lit(1)).otherwise(F.lit(0)) + F.when(F.col('query_outdoor_seating').isNotNull() & (F.col('outdoor_seating_clean') == F.col('query_outdoor_seating')), F.lit(1)).otherwise(F.lit(0))).withColumn('attribute_match_score', F.when(F.col('requested_attribute_count') > 0, F.col('matched_attribute_count') / F.col('requested_attribute_count')).otherwise(F.lit(0.0))).withColumn('rating_score', F.col('stars') / F.lit(5.0)).withColumn('review_confidence_score', F.log1p(F.col('review_count')) / F.lit(float(max_review_log))).withColumn('is_relevant', (F.expr('array_contains(category_tokens, query_category)') & (F.col('stars') >= F.lit(config.minimum_rating)) & (F.col('query_price_range').isNull() | (F.col('price_range_numeric') == F.col('query_price_range'))) & (F.col('query_wifi').isNull() | (F.col('wifi_clean') == F.col('query_wifi'))) & (F.col('query_outdoor_seating').isNull() | (F.col('outdoor_seating_clean') == F.col('query_outdoor_seating')))).cast('int')).select('query_id', 'query_type', 'dataset_split', 'business_id', 'stars', 'review_count', 'category_match_score', 'attribute_match_score', 'rating_score', 'review_confidence_score', 'is_relevant').cache()
    candidate_count = evaluation_candidates_df.count()
    if candidate_count == 0:
        raise ValueError('No evaluation candidates were created.')
    return (evaluation_candidates_df, candidate_count)

def score_and_evaluate(candidate_df, weights, top_k: int):
    validate_weights(weights)
    scored_df = candidate_df.withColumn('final_score', F.col('category_match_score') * F.lit(float(weights['category'])) + F.col('attribute_match_score') * F.lit(float(weights['attributes'])) + F.col('rating_score') * F.lit(float(weights['rating'])) + F.col('review_confidence_score') * F.lit(float(weights['review_confidence'])))
    ranking_window = Window.partitionBy('query_id').orderBy(F.col('final_score').desc(), F.col('stars').desc(), F.col('review_count').desc(), F.col('business_id').asc())
    top_k_df = scored_df.withColumn('recommendation_rank', F.row_number().over(ranking_window)).filter(F.col('recommendation_rank') <= top_k).cache()
    relevant_counts_df = candidate_df.filter(F.col('is_relevant') == 1).groupBy('query_id', 'query_type').agg(F.countDistinct('business_id').alias('relevant_count'))
    hit_summary_df = top_k_df.groupBy('query_id', 'query_type').agg(F.sum('is_relevant').cast('double').alias('hit_count'), F.min(F.when(F.col('is_relevant') == 1, F.col('recommendation_rank'))).alias('first_relevant_rank'), F.sum(F.when(F.col('is_relevant') == 1, F.lit(1.0) / (F.log(F.col('recommendation_rank') + F.lit(1.0)) / F.log(F.lit(2.0)))).otherwise(F.lit(0.0))).alias('dcg_at_k'))

    @F.udf(returnType=DoubleType())
    def ideal_dcg_at_k(relevant_count):
        if relevant_count is None or relevant_count <= 0:
            return 0.0
        ideal_hits = min(int(relevant_count), int(top_k))
        return float(sum((1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))))
    query_metrics_df = relevant_counts_df.join(hit_summary_df, ['query_id', 'query_type'], 'left').fillna({'hit_count': 0.0, 'dcg_at_k': 0.0}).withColumn('precision_at_k', F.col('hit_count') / F.lit(float(top_k))).withColumn('recall_at_k', F.when(F.col('relevant_count') > 0, F.col('hit_count') / F.col('relevant_count')).otherwise(F.lit(0.0))).withColumn('hit_rate_at_k', F.when(F.col('hit_count') > 0, F.lit(1.0)).otherwise(F.lit(0.0))).withColumn('mrr_at_k', F.when(F.col('first_relevant_rank').isNotNull(), F.lit(1.0) / F.col('first_relevant_rank')).otherwise(F.lit(0.0))).withColumn('idcg_at_k', ideal_dcg_at_k(F.col('relevant_count'))).withColumn('ndcg_at_k', F.when(F.col('idcg_at_k') > 0, F.col('dcg_at_k') / F.col('idcg_at_k')).otherwise(F.lit(0.0)))
    overall_metrics_df = query_metrics_df.agg(F.round(F.avg('precision_at_k'), 4).alias('precision_at_k'), F.round(F.avg('recall_at_k'), 4).alias('recall_at_k'), F.round(F.avg('hit_rate_at_k'), 4).alias('hit_rate_at_k'), F.round(F.avg('mrr_at_k'), 4).alias('mrr_at_k'), F.round(F.avg('ndcg_at_k'), 4).alias('ndcg_at_k'), F.countDistinct('query_id').alias('query_count'))
    return (top_k_df, query_metrics_df, overall_metrics_df)

def tune_weights(evaluation_candidates_df, weight_grid, config: EvaluationConfig):
    validation_candidate_df = evaluation_candidates_df.filter(F.col('dataset_split') == 'validation').cache()
    validation_rows = []
    for grid_index, weight_set in enumerate(weight_grid, start=1):
        validation_top_k_df, _, validation_metrics_df = score_and_evaluate(validation_candidate_df, weight_set, config.top_k)
        metric_row = validation_metrics_df.first()
        validation_top_k_df.unpersist()
        if metric_row is None or metric_row['query_count'] == 0:
            raise ValueError('Validation evaluation returned no query metrics.')
        validation_rows.append({'grid_index': grid_index, 'category_weight': float(weight_set['category']), 'attribute_weight': float(weight_set['attributes']), 'rating_weight': float(weight_set['rating']), 'review_weight': float(weight_set['review_confidence']), 'precision_at_k': float(metric_row['precision_at_k']), 'recall_at_k': float(metric_row['recall_at_k']), 'hit_rate_at_k': float(metric_row['hit_rate_at_k']), 'mrr_at_k': float(metric_row['mrr_at_k']), 'ndcg_at_k': float(metric_row['ndcg_at_k']), 'validation_queries': int(metric_row['query_count'])})
    spark = evaluation_candidates_df.sparkSession
    weight_results_df = spark.createDataFrame(validation_rows)
    best_row = weight_results_df.orderBy(F.desc('ndcg_at_k'), F.desc('mrr_at_k'), F.desc('recall_at_k'), F.asc('grid_index')).first()
    if best_row is None:
        raise ValueError('No best weight combination was selected.')
    best_weights = {'category': float(best_row['category_weight']), 'attributes': float(best_row['attribute_weight']), 'rating': float(best_row['rating_weight']), 'review_confidence': float(best_row['review_weight'])}
    validate_weights(best_weights)
    return (weight_results_df, best_weights)

def evaluate_test_split(evaluation_candidates_df, best_weights, config: EvaluationConfig):
    test_candidate_df = evaluation_candidates_df.filter(F.col('dataset_split') == 'test').cache()
    test_top_k_df, query_metrics_df, overall_metrics_df = score_and_evaluate(test_candidate_df, best_weights, config.top_k)
    candidate_business_count = test_candidate_df.select('business_id').distinct().count()
    recommended_business_count = test_top_k_df.select('business_id').distinct().count()
    coverage = recommended_business_count / candidate_business_count if candidate_business_count > 0 else 0.0
    return {'test_candidate_df': test_candidate_df, 'test_top_k_df': test_top_k_df, 'query_metrics_df': query_metrics_df, 'overall_metrics_df': overall_metrics_df, 'candidate_business_count': candidate_business_count, 'recommended_business_count': recommended_business_count, 'coverage': coverage}

# ==============================================================================
# Dependency check job
# ==============================================================================

def inspect_executor_partition(_rows):
    import socket as worker_socket
    import sys as worker_sys
    try:
        import numpy as worker_numpy
    except ModuleNotFoundError as error:
        raise RuntimeError(f'NumPy is missing on executor node {worker_socket.gethostname()}.') from error
    yield {'hostname': worker_socket.gethostname(), 'python_version': worker_sys.version.split()[0], 'numpy_version': worker_numpy.__version__}

def run_dependency_check():
    spark = SparkSession.builder.appName('YelpContentRecommenderDependencyCheck').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    try:
        partitions = max(2, spark.sparkContext.defaultParallelism)
        executor_results = spark.sparkContext.parallelize(range(partitions), partitions).mapPartitions(inspect_executor_partition).collect()
        unique_workers = {(row['hostname'], row['python_version'], row['numpy_version']) for row in executor_results}
        summary = {'status': 'PASSED', 'driver': {'hostname': socket.gethostname(), 'python_version': sys.version.split()[0], 'numpy_version': np.__version__, 'spark_version': spark.version, 'application_id': spark.sparkContext.applicationId}, 'executors_observed': [{'hostname': hostname, 'python_version': python_version, 'numpy_version': numpy_version} for hostname, python_version, numpy_version in sorted(unique_workers)], 'pyspark_ml_imports': ['CountVectorizer', 'CountVectorizerModel', 'Normalizer', 'VectorUDT']}
        print(json.dumps(summary, indent=2))
        print('Cluster dependency check passed.')
    finally:
        spark.stop()

# ==============================================================================
# Training job
# ==============================================================================

def configure_spark(spark):
    default_parallelism = max(1, spark.sparkContext.defaultParallelism)
    shuffle_partitions = max(64, min(400, default_parallelism * 3))
    output_partitions = max(8, min(64, default_parallelism))
    spark.conf.set('spark.sql.adaptive.enabled', 'true')
    spark.conf.set('spark.sql.adaptive.coalescePartitions.enabled', 'true')
    spark.conf.set('spark.sql.adaptive.skewJoin.enabled', 'true')
    spark.conf.set('spark.sql.shuffle.partitions', str(shuffle_partitions))
    spark.sparkContext.setLogLevel('WARN')
    print('Default parallelism:', default_parallelism)
    print('Shuffle partitions:', shuffle_partitions)
    print('Output partitions:', output_partitions)
    return output_partitions

def build_final_config(spark, args, paths, input_path, best_weights, final_metrics_df, coverage_result):
    metric_row = final_metrics_df.first()
    if metric_row is None:
        raise ValueError('Final test metrics are empty.')
    return {'model_name': 'Yelp_Content_Based_Recommender', 'model_version': args.model_version, 'model_type': 'Query-Based Content Recommender', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'spark_version': spark.version, 'top_k': int(DEFAULT_USER_QUERY['top_k']), 'weights': best_weights, 'test_metrics': {'precision_at_k': float(metric_row['precision_at_k']), 'recall_at_k': float(metric_row['recall_at_k']), 'hit_rate_at_k': float(metric_row['hit_rate_at_k']), 'mrr_at_k': float(metric_row['mrr_at_k']), 'ndcg_at_k': float(metric_row['ndcg_at_k']), 'test_queries': int(metric_row['query_count'])}, 'coverage': {'candidate_businesses': int(coverage_result['candidate_business_count']), 'unique_recommended_businesses': int(coverage_result['recommended_business_count']), 'catalogue_coverage': round(float(coverage_result['coverage']), 4), 'catalogue_coverage_percentage': round(float(coverage_result['coverage']) * 100, 2)}, 'feature_columns': ['categories', 'city', 'state', 'stars', 'review_count', 'price_range_numeric', 'wifi_clean', 'outdoor_seating_clean'], 'vector_column': 'category_vector_normalized', 'input_path': input_path, 'business_vector_path': paths.business_vectors, 'category_vectorizer_path': paths.category_vectorizer, 'normalizer_path': paths.normalizer}

def run_training(args):
    spark = SparkSession.builder.appName('YelpContentBasedRecommenderTraining').getOrCreate()
    output_partitions = configure_spark(spark)
    paths = build_model_paths(bucket=args.bucket, model_version=args.model_version, output_prefix=args.output_prefix)
    build_started_at = datetime.now(timezone.utc).isoformat()
    input_path = None
    try:
        prepare_output_root(spark, paths.output_root, args.overwrite)
        if args.input_path is not None:
            if not path_exists(spark, args.input_path):
                raise FileNotFoundError(f'Configured input path does not exist: {args.input_path}')
            input_path = args.input_path
        else:
            input_path = resolve_first_existing_path(spark, default_business_input_candidates(args.bucket))
        building_manifest = {'model_name': 'Yelp_Content_Based_Recommender', 'model_version': args.model_version, 'status': 'BUILDING', 'build_started_at_utc': build_started_at, 'input_path': input_path, 'output_root': paths.output_root}
        write_json_document(spark, building_manifest, paths.model_manifest)
        print('Loading business data:', input_path)
        raw_business_df = spark.read.parquet(input_path)
        prepared = prepare_business_features(raw_business_df)
        serving_df = prepared['business_serving_df']
        vectorizer_model = prepared['category_vectorizer_model']
        normalizer = prepared['normalizer']
        evaluation_config = EvaluationConfig()
        evaluation_queries_df, query_count = create_evaluation_queries(serving_df, evaluation_config)
        split_queries_df, split_counts = split_evaluation_queries(evaluation_queries_df, query_count, evaluation_config)
        evaluation_candidates_df, evaluation_candidate_count = build_evaluation_candidates(serving_df, split_queries_df, vectorizer_model, normalizer, evaluation_config)
        weight_results_df, best_weights = tune_weights(evaluation_candidates_df, WEIGHT_GRID, evaluation_config)
        coverage_result = evaluate_test_split(evaluation_candidates_df, best_weights, evaluation_config)
        final_test_metrics_df = coverage_result['overall_metrics_df']
        final_recommendations_df = recommend_businesses(user_query=DEFAULT_USER_QUERY, business_features_df=serving_df, vectorizer_model=vectorizer_model, normalizer_model=normalizer, weights=best_weights).cache()
        final_recommendation_count = final_recommendations_df.count()
        if final_recommendation_count == 0:
            raise ValueError('The final sample query returned zero recommendations.')
        final_model_config = build_final_config(spark, args, paths, input_path, best_weights, final_test_metrics_df, coverage_result)
        save_ml_artifact(vectorizer_model.write, paths.category_vectorizer, overwrite=args.overwrite)
        save_ml_artifact(normalizer.write, paths.normalizer, overwrite=args.overwrite)
        save_parquet(serving_df, paths.business_vectors, overwrite=args.overwrite, partitions=output_partitions)
        save_parquet(weight_results_df, paths.validation_results, overwrite=args.overwrite)
        save_parquet(final_test_metrics_df, paths.test_metrics, overwrite=args.overwrite)
        recommendation_columns = ['recommendation_rank', 'business_id', 'name', 'categories', 'city', 'state', 'stars', 'review_count', 'price_range_numeric', 'wifi_clean', 'outdoor_seating_clean', 'category_match_score', 'attribute_match_score', 'rating_score', 'review_confidence_score', 'final_score', 'final_score_percentage']
        save_parquet(final_recommendations_df.select(*recommendation_columns), paths.sample_recommendations, overwrite=args.overwrite)
        write_json_document(spark, final_model_config, paths.model_configuration)
        artifact_paths = {'category_vectorizer': paths.category_vectorizer, 'normalizer': paths.normalizer, 'business_vectors': paths.business_vectors, 'validation_results': paths.validation_results, 'test_metrics': paths.test_metrics, 'sample_recommendations': paths.sample_recommendations, 'model_configuration': paths.model_configuration}
        verify_artifact_paths(spark, artifact_paths)
        loaded_vectorizer = CountVectorizerModel.load(paths.category_vectorizer)
        loaded_normalizer = Normalizer.load(paths.normalizer)
        loaded_serving_df = spark.read.parquet(paths.business_vectors)
        loaded_metrics_df = spark.read.parquet(paths.test_metrics)
        loaded_recommendations_df = spark.read.parquet(paths.sample_recommendations)
        loaded_config = read_json_document(spark, paths.model_configuration)
        loaded_business_count = loaded_serving_df.count()
        if loaded_business_count != prepared['serving_row_count']:
            raise ValueError('Reloaded business-vector row count does not match.')
        if loaded_vectorizer.vocabulary != vectorizer_model.vocabulary:
            raise ValueError('Reloaded vocabulary does not match.')
        if loaded_metrics_df.first() is None:
            raise ValueError('Reloaded test metrics are empty.')
        if loaded_recommendations_df.count() != final_recommendation_count:
            raise ValueError('Reloaded recommendation count does not match.')
        if loaded_config.get('model_version') != args.model_version:
            raise ValueError('Reloaded configuration has the wrong version.')
        smoke_test_df = recommend_businesses(user_query={'categories': ['pizza', 'italian'], 'location': {'city': None, 'state': None}, 'attributes': {'price_range': 2.0}, 'minimum_rating': 4.0, 'top_k': 5, 'exclude_business_ids': []}, business_features_df=loaded_serving_df, vectorizer_model=loaded_vectorizer, normalizer_model=loaded_normalizer, weights=loaded_config['weights']).cache()
        smoke_test_count = smoke_test_df.count()
        if smoke_test_count == 0:
            raise ValueError('Saved-model smoke test returned zero rows.')
        ready_manifest = {'model_name': 'Yelp_Content_Based_Recommender', 'model_version': args.model_version, 'status': 'READY', 'build_started_at_utc': build_started_at, 'build_completed_at_utc': datetime.now(timezone.utc).isoformat(), 'spark_version': spark.version, 'spark_application_id': spark.sparkContext.applicationId, 'input_path': input_path, 'output_root': paths.output_root, 'input_business_rows': int(prepared['business_row_count']), 'saved_business_rows': int(loaded_business_count), 'vocabulary_size': int(len(loaded_vectorizer.vocabulary)), 'evaluation_query_count': int(query_count), 'evaluation_candidate_rows': int(evaluation_candidate_count), 'query_split_counts': split_counts, 'smoke_test_recommendations': int(smoke_test_count), 'artifacts': artifact_paths}
        write_json_document(spark, ready_manifest, paths.model_manifest)
        verified_manifest = read_json_document(spark, paths.model_manifest)
        if verified_manifest.get('status') != 'READY':
            raise ValueError('The final model manifest is not READY.')
        print(json.dumps(verified_manifest, indent=2))
        print('Training workflow completed successfully.')
    except Exception as error:
        print('Training workflow failed:', str(error), file=sys.stderr)
        traceback.print_exc()
        if input_path is not None:
            try:
                failed_manifest = {'model_name': 'Yelp_Content_Based_Recommender', 'model_version': args.model_version, 'status': 'FAILED', 'build_started_at_utc': build_started_at, 'build_failed_at_utc': datetime.now(timezone.utc).isoformat(), 'input_path': input_path, 'output_root': paths.output_root, 'error_type': type(error).__name__, 'error_message': str(error)[:2000]}
                write_json_document(spark, failed_manifest, paths.model_manifest)
            except Exception as manifest_error:
                print('Could not write FAILED manifest:', manifest_error, file=sys.stderr)
        raise
    finally:
        spark.stop()

# ==============================================================================
# Recommendation job
# ==============================================================================

def _comma_separated(value):
    if value is None:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]

def run_recommendation(args):
    spark = SparkSession.builder.appName('YelpContentBasedRecommendationInference').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    try:
        paths = build_model_paths(bucket=args.bucket, model_version=args.model_version, output_prefix=args.output_prefix)
        if not path_exists(spark, paths.model_manifest):
            raise FileNotFoundError(f'Model manifest does not exist: {paths.model_manifest}')
        manifest = read_json_document(spark, paths.model_manifest)
        if manifest.get('status') != 'READY':
            raise RuntimeError(f"The requested model version is not READY. Current status: {manifest.get('status')}")
        artifact_paths = manifest.get('artifacts', {})
        required_artifacts = {'business_vectors', 'category_vectorizer', 'normalizer', 'model_configuration'}
        missing_artifacts = required_artifacts - set(artifact_paths)
        if missing_artifacts:
            raise ValueError('Manifest is missing artifacts: ' + ', '.join(sorted(missing_artifacts)))
        for artifact_name in required_artifacts:
            artifact_path = artifact_paths[artifact_name]
            if not path_exists(spark, artifact_path):
                raise FileNotFoundError(f'Artifact does not exist: {artifact_name} -> {artifact_path}')
        vectorizer_model = CountVectorizerModel.load(artifact_paths['category_vectorizer'])
        normalizer_model = Normalizer.load(artifact_paths['normalizer'])
        business_vectors_df = spark.read.parquet(artifact_paths['business_vectors'])
        model_config = read_json_document(spark, artifact_paths['model_configuration'])
        attributes = {}
        if args.price_range is not None:
            attributes['price_range'] = args.price_range
        if args.wifi is not None:
            attributes['wifi'] = args.wifi
        if args.outdoor_seating is not None:
            attributes['outdoor_seating'] = args.outdoor_seating
        user_query = {'categories': _comma_separated(args.categories), 'location': {'city': args.city, 'state': args.state}, 'attributes': attributes, 'minimum_rating': args.minimum_rating, 'top_k': args.top_k, 'exclude_business_ids': _comma_separated(args.exclude_business_ids)}
        recommendations_df = recommend_businesses(user_query=user_query, business_features_df=business_vectors_df, vectorizer_model=vectorizer_model, normalizer_model=normalizer_model, weights=model_config['weights']).cache()
        result_count = recommendations_df.count()
        if result_count == 0:
            print('No recommendations matched the supplied filters. Relax city/state, rating, or attributes.', file=sys.stderr)
            return
        result_columns = ['recommendation_rank', 'business_id', 'name', 'categories', 'city', 'state', 'stars', 'review_count', 'price_range_numeric', 'wifi_clean', 'outdoor_seating_clean', 'category_match_percentage', 'matched_attribute_count', 'requested_attribute_count', 'attribute_match_type', F.round(F.col('rating_score') * 100, 2).alias('rating_score_percentage'), F.round(F.col('review_confidence_score') * 100, 2).alias('review_confidence_percentage'), 'final_score_percentage']
        print('User query:')
        print(json.dumps(user_query, indent=2))
        recommendations_df.select(*result_columns).show(args.top_k, truncate=False)
        if args.result_output_path:
            mode = 'overwrite' if args.overwrite_result else 'errorifexists'
            recommendations_df.select(*result_columns).coalesce(1).write.mode(mode).option('compression', 'snappy').parquet(args.result_output_path)
            print('Results saved to:', args.result_output_path)
    finally:
        spark.stop()

# ==============================================================================
# Artifact validation job
# ==============================================================================

def run_validation(args):
    spark = SparkSession.builder.appName('YelpContentModelArtifactValidation').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    try:
        paths = build_model_paths(bucket=args.bucket, model_version=args.model_version, output_prefix=args.output_prefix)
        if not path_exists(spark, paths.model_manifest):
            raise FileNotFoundError(f'Manifest does not exist: {paths.model_manifest}')
        manifest = read_json_document(spark, paths.model_manifest)
        if manifest.get('status') != 'READY':
            raise RuntimeError(f"Model status is {manifest.get('status')}, not READY.")
        artifacts = manifest.get('artifacts', {})
        required = {'business_vectors', 'category_vectorizer', 'normalizer', 'model_configuration', 'validation_results', 'test_metrics', 'sample_recommendations'}
        missing = required - set(artifacts)
        if missing:
            raise ValueError('Manifest is missing artifacts: ' + ', '.join(sorted(missing)))
        for artifact_name in sorted(required):
            artifact_path = artifacts[artifact_name]
            if not path_exists(spark, artifact_path):
                raise FileNotFoundError(f'Missing artifact: {artifact_name} -> {artifact_path}')
        vectorizer_model = CountVectorizerModel.load(artifacts['category_vectorizer'])
        normalizer_model = Normalizer.load(artifacts['normalizer'])
        business_vectors_df = spark.read.parquet(artifacts['business_vectors'])
        metrics_df = spark.read.parquet(artifacts['test_metrics'])
        recommendations_df = spark.read.parquet(artifacts['sample_recommendations'])
        model_config = read_json_document(spark, artifacts['model_configuration'])
        business_count = business_vectors_df.count()
        if business_count != int(manifest['saved_business_rows']):
            raise ValueError(f"Business row mismatch: {business_count} versus {manifest['saved_business_rows']}.")
        vocabulary_size = len(vectorizer_model.vocabulary)
        if vocabulary_size != int(manifest['vocabulary_size']):
            raise ValueError(f"Vocabulary mismatch: {vocabulary_size} versus {manifest['vocabulary_size']}.")
        if metrics_df.first() is None:
            raise ValueError('Test metrics are empty.')
        if recommendations_df.first() is None:
            raise ValueError('Sample recommendations are empty.')
        if model_config.get('model_version') != args.model_version:
            raise ValueError('Configuration version does not match.')
        smoke_test_df = recommend_businesses(user_query={'categories': ['pizza', 'italian'], 'location': {'city': None, 'state': None}, 'attributes': {'price_range': 2.0}, 'minimum_rating': 4.0, 'top_k': 5, 'exclude_business_ids': []}, business_features_df=business_vectors_df, vectorizer_model=vectorizer_model, normalizer_model=normalizer_model, weights=model_config['weights'])
        smoke_count = smoke_test_df.count()
        if smoke_count == 0:
            raise ValueError('Smoke test returned zero recommendations.')
        summary = {'status': 'VALID', 'model_version': args.model_version, 'business_rows': business_count, 'vocabulary_size': vocabulary_size, 'smoke_test_rows': smoke_count, 'manifest_status': manifest['status']}
        print(json.dumps(summary, indent=2))
        print('Saved model validation passed.')
    finally:
        spark.stop()

# ==============================================================================
# Unified command-line interface
# ==============================================================================

def _add_common_model_arguments(parser):
    parser.add_argument("--bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)


def build_cli_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Single-file EMR job for checking dependencies, training, "
            "validating, and running the Yelp content recommender."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "check",
        help="Check NumPy and PySpark ML on the driver and executor nodes.",
    )

    train_parser = commands.add_parser(
        "train",
        help="Train, tune, evaluate, save, reload, and mark a model READY.",
    )
    _add_common_model_arguments(train_parser)
    train_parser.add_argument("--input-path", default=None)
    train_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild an existing model version.",
    )

    validate_parser = commands.add_parser(
        "validate",
        help="Validate all artifacts for a saved READY model version.",
    )
    _add_common_model_arguments(validate_parser)

    recommend_parser = commands.add_parser(
        "recommend",
        help="Load a READY model and generate recommendations without retraining.",
    )
    _add_common_model_arguments(recommend_parser)
    recommend_parser.add_argument("--categories", required=True)
    recommend_parser.add_argument("--city", default=None)
    recommend_parser.add_argument("--state", default=None)
    recommend_parser.add_argument("--price-range", type=float, default=None)
    recommend_parser.add_argument(
        "--wifi", choices=["free", "paid", "no"], default=None
    )
    recommend_parser.add_argument(
        "--outdoor-seating",
        choices=["true", "false", "yes", "no"],
        default=None,
    )
    recommend_parser.add_argument("--minimum-rating", type=float, default=None)
    recommend_parser.add_argument("--top-k", type=int, default=10)
    recommend_parser.add_argument("--exclude-business-ids", default=None)
    recommend_parser.add_argument(
        "--result-output-path",
        default=None,
        help="Optional S3 path for saving recommendation results as Parquet.",
    )
    recommend_parser.add_argument("--overwrite-result", action="store_true")

    return parser


def main():
    args = build_cli_parser().parse_args()

    if args.command == "check":
        run_dependency_check()
    elif args.command == "train":
        run_training(args)
    elif args.command == "validate":
        run_validation(args)
    elif args.command == "recommend":
        run_recommendation(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
