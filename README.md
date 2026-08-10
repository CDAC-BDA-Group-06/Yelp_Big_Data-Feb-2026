# Yelp Review Data Analytics Using Big Data Technologies

## Basic Details

| Field             | Details                                                |
| ----------------- | ------------------------------------------------------ |
| **Project Title** | Yelp Review Data Analytics Using Big Data Technologies |
| **Team Size**     | 7 Members                                              |

---

# Team Members

* Nagesh -> Problem Statement, Data, Ingestion, CI/CD
* Maheshwari Phalke -> Bronze to Silver and Silver to Gold Glue Jobs & transformations
* Vishal Dewanjee -> Machine Learning Flow and Model
* Yogesh Thakre -> Dashboards Overview and Why Two Dashboards
* Shraddha Patil -> In-detail Yelp dataset overview
* Saurav Makde -> In-depth specific business performance overview
* Mihir Zope -> Interactive UI and Query Engine



# Problem Statement

Online businesses receive millions of customer reviews every day, making it difficult to manually analyze customer feedback and identify the factors affecting customer satisfaction and business performance.

This project leverages the **Yelp Complete Open Dataset 2024** to analyze customer reviews, business information, user behavior, and ratings using **Big Data Analytics, Machine Learning, and Business Intelligence** techniques.

The project aims to uncover valuable business insights, predict business ratings, understand customer sentiment, and help businesses make data-driven decisions that improve customer experience and overall performance.

---

# Project Objectives

1. Analyze customer reviews, ratings, and business information.
2. Perform Exploratory Data Analysis (EDA) to identify trends and patterns.
3. Conduct sentiment analysis to understand customer opinions.
4. Predict business ratings or review sentiment.
5. Analyze user behavior and business performance.
6. Create interactive Power BI dashboards for visualization.
7. Generate actionable insights to improve customer satisfaction and business decisions.

---

# 📊 Power BI Dashboard

The project includes an interactive Power BI dashboard for analyzing **business performance, customer engagement, ratings, reviews, and other Yelp business metrics**.

### 🔗 View the Interactive Dashboard

👉 **[Open Yelp Business Analytics Dashboard](https://app.powerbi.com/links/NZwkE1qJZK?ctid=56c1d497-700b-49cf-8f8d-3dd6b20d522f&pbi_source=linkShare)**

---

# Dataset Description

The project uses the **Yelp Complete Open Dataset 2024**, which contains multiple datasets representing businesses, users, reviews, check-ins, and tips.

## Dataset Summary

| Dataset         |   Records | Columns | Approx. Size |
| --------------- | --------: | ------: | -----------: |
| `business.json` |   150,346 |      14 |      ~118 MB |
| `review.json`   | 6,990,280 |       9 |      ~5.3 GB |
| `user.json`     | 1,987,897 |      22 |      ~3.2 GB |
| `checkin.json`  |   131,930 |       2 |      ~290 MB |
| `tip.json`      |   908,915 |       5 |      ~250 MB |

---

# Dataset Details

## Business Information

Contains information about businesses, including:

* Business ID
* Business Name
* Address
* City
* State
* Categories
* Operating Hours
* Average Star Rating
* Total Review Count

---

## Review Data

Contains customer reviews with:

* Review ID
* User ID
* Business ID
* Star Rating
* Review Text
* Review Date
* Useful Votes
* Funny Votes
* Cool Votes

---

## User Data

Contains information about Yelp users, including:

* User ID
* Review Count
* Average Rating Given
* Friends
* Fans
* Compliments
* Account Activity

---

## Check-in Data

Stores customer check-in timestamps to analyze:

* Business popularity
* Visit frequency
* Customer activity patterns

---

## Tip Data

Contains short customer tips including:

* User ID
* Business ID
* Tip Text
* Date
* Number of Likes

---

# Data Format

The Yelp dataset is provided in **JSON** format and contains both:

### Structured Data

* Business Information
* User Information
* Ratings
* Check-ins

### Unstructured Data

* Customer Reviews
* Tips

This makes the dataset suitable for:

* Big Data Analytics
* Machine Learning
* Natural Language Processing (NLP)
* Business Intelligence
* Data Visualization

---

# Tools & Technologies Used

| Technology           | Purpose                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Amazon S3**        | Stores raw and processed Yelp datasets in a scalable cloud storage environment.                               |
| **GitHub Actions**   | Automates Yelp data ingestion and deployment workflows using a GitHub-hosted runner.                          |
| **Amazon EMR**       | Distributed processing of large datasets using Apache Spark.                                                  |
| **PySpark**          | Large-scale data transformation, cleaning, feature engineering, and machine learning.                         |
| **Jupyter Notebook** | Interactive development, EDA, visualization, and experimentation.                                             |
| **AWS Glue**         | Serverless ETL pipelines and Data Catalog management.                                                         |
| **Amazon Athena**    | Query data stored in Amazon S3 using standard SQL.                                                            |
| **ODBC/JDBC**        | Connects Athena with Power BI and Python applications.                                                        |
| **Power BI/Tableau** | Interactive dashboards and business intelligence reporting.                                                   |
| **Git & GitHub**     | Version control, collaboration, and project management.                                                       |
| **Terraform**        | Infrastructure as Code (IaC) for provisioning AWS resources such as S3, Glue, IAM, and networking components. |

---

# Project Workflow

```text
                         ┌─────────────────────────┐
                         │    GitHub Repository    │
                         │  Source Code / Config   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   GitHub Actions Runner │
                         │     CI/CD + Ingestion   │
                         └────────────┬────────────┘
                                      │
                       ┌──────────────┴──────────────┐
                       │                             │
                       ▼                             ▼
              ┌────────────────┐           ┌────────────────┐
              │    Terraform   │           │ Data Ingestion │
              │      IaC       │           │ Automation     │
              └───────┬────────┘           └───────┬────────┘
                      │                            │
                      └──────────────┬─────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │        AWS CLOUD        │
                         │ S3 / Glue │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      S3 DATA LAKE       │
                         │                         │
                         │   BRONZE                │
                         │ Raw Yelp JSON           │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      AWS GLUE     │
                         │ Cleaning + Transformation│
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │        SILVER           │
                         │ Cleaned / Flattened      │
                         │ Structured Parquet       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      AWS GLUE           │
                         │ Joins + Aggregations    │
                         │ Feature Engineering     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │          GOLD           │
                         │ Analytics + ML/RAG Data │
                         └────────────┬────────────┘
                                      │
                  ┌───────────────────┼────────────────────┐
                  │                   │                    │
                  ▼                   ▼                    ▼
       ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
       │  BI / ANALYTICS  │  │ MACHINE LEARNING │  │   RAG PIPELINE   │
       └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                │                     │                    │
                ▼                     ▼                    ▼
       ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
       │     Athena       │  │ Feature Dataset  │  │ Review / Tip Text│
       │      SQL         │  │                  │  │    Corpus        │
       └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                │                     │                    │
                ▼                     ▼                    ▼
       ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
       │    Power BI      │  │ ML Model Training│  │ Text Chunking    │
       │   Dashboards     │  │                  │  └────────┬─────────┘
       └──────────────────┘  └────────┬─────────┘           │
                                      │                     ▼
                                      ▼            ┌──────────────────┐
                             ┌──────────────────┐  │ Embedding Model  │
                             │ Model Evaluation │  └────────┬─────────┘
                             └────────┬─────────┘           │
                                      │                     ▼
                                      ▼            ┌──────────────────┐
                             ┌──────────────────┐  │   Vector Store   │
                             │ Predictions /    │  │  Embeddings      │
                             │ Sentiment /      │  └────────┬─────────┘
                             │ Rating Prediction│           │
                             └────────┬─────────┘           ▼
                                      │            ┌──────────────────┐
                                      │            │ Query / Retrieval│
                                      │            └────────┬─────────┘
                                      │                     │
                                      │                     ▼
                                      │            ┌──────────────────┐
                                      │            │      LLM         │
                                      │            │ Answer Generation│
                                      │            └────────┬─────────┘
                                      │                     │
                                      └──────────┬──────────┘
                                                 ▼
                                      ┌──────────────────────┐
                                      │   Query / UI Layer   │
                                      │ Interactive Questions│
                                      │ Business Insights    │
                                      └──────────────────────┘
```

---

# Transformations

Bronze → Silver:  Flatten nested JSON → Remove nulls & duplicates 
                  Explode comma-separated check-in timestamps 
                  Parse & cast timestamps → Parse `yelping_since` 
                  Data type standardization

Silver → Gold:  Join datasets → Group-by aggregations → 
                Extract day-of-week & hour-of-day → Review-text feature engineering
                ML & RAG feature extraction → Store analytics-ready datasets in S3 Gold


# Expected Outcomes

* Centralized cloud-based data lake architecture
* Automated ETL pipeline using AWS Glue
* Distributed data processing using PySpark on Amazon EMR
* Interactive SQL analytics through Amazon Athena
* Business intelligence dashboards in Power BI
* Sentiment analysis of customer reviews
* Rating prediction using Machine Learning
* Customer behavior analysis
* Business performance analytics
* Actionable recommendations for business owners

---

# Key Features

* Large-scale distributed data processing
* Cloud-native architecture on AWS
* Automated data ingestion
* GitHub-hosted runner-based ingestion
* AWS S3 data lake
* Automated ETL processing using AWS Glue
* Data warehouse-ready Gold layer
* Machine Learning integration
* Interactive BI dashboards
* Infrastructure as Code using Terraform
* CI/CD automation using GitHub Actions

---

# Architecture Diagram
<img width="1600" height="1249" alt="WhatsApp Image 2026-08-10 at 01 42 37" src="https://github.com/user-attachments/assets/778bee00-fd6b-498c-bcb8-967bf84a7540" />


---

# Future Enhancements

* Recommendation System
* Fake Review Detection
* Topic Modeling using BERTopic/LDA
* Review Summarization using LLMs
* Predictive Business Performance Modeling
