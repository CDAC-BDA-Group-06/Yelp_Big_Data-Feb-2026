# Yelp Review Data Analytics Using Big Data Technologies

## Basic Details

| Field | Details |
|-------|---------|
| **Project Title** | Yelp Review Data Analytics Using Big Data Technologies |
| **Duration** |  |
| **Team Size** | 7 Members |

---

# Team Members

- Nagesh Khichade
- Saurav Makde
- Shraddha Patil
- Mihir Zope
- Maheshwari Phalke
- Vishal Dewanjee
- Yogesh Thakre

---

# Problem Statement

Online businesses receive millions of customer reviews every day, making it difficult to manually analyze customer feedback and identify the factors affecting customer satisfaction and business performance.

This project leverages the **Yelp Complete Open Dataset 2024** to analyze customer reviews, business information, user behavior, and ratings using **Big Data Analytics, Machine Learning, and Business Intelligence** techniques.

The project aims to uncover valuable business insights, predict business ratings, understand customer sentiment, and help businesses make data-driven decisions that improve customer experience and overall performance.

---

# Project Objectives

1. Analyze customer reviews, ratings, and business information.
2. Perform Exploratory Data Analysis (EDA) to identify trends and patterns.
3. Conduct sentiment analysis to understand customer opinions.
4. Build Machine Learning models to predict business ratings or review sentiment.
5. Analyze user behavior and business performance.
6. Create interactive Power BI dashboards for visualization.
7. Generate actionable insights to improve customer satisfaction and business decisions.
---

# Dataset Description

The project uses the **Yelp Complete Open Dataset 2024**, which contains multiple datasets representing businesses, users, reviews, check-ins, and tips.

## Dataset Summary

| Dataset | Records | Columns | Approx. Size |
|----------|---------|---------|--------------|
| `business.json` | 150,346 | 14 | ~118 MB |
| `review.json` | 6,990,280 | 9 | ~5.3 GB |
| `user.json` | 1,987,897 | 22 | ~3.2 GB |
| `checkin.json` | 131,930 | 2 | ~290 MB |
| `tip.json` | 908,915 | 5 | ~250 MB |

---

# Dataset Details

## Business Information

Contains information about businesses, including:

- Business ID
- Business Name
- Address
- City
- State
- Categories
- Operating Hours
- Average Star Rating
- Total Review Count

---

## Review Data

Contains customer reviews with:

- Review ID
- User ID
- Business ID
- Star Rating
- Review Text
- Review Date
- Useful Votes
- Funny Votes
- Cool Votes

---

## User Data

Contains information about Yelp users, including:

- User ID
- Review Count
- Average Rating Given
- Friends
- Fans
- Compliments
- Account Activity

---

## Check-in Data

Stores customer check-in timestamps to analyze:

- Business popularity
- Visit frequency
- Customer activity patterns

---

## Tip Data

Contains short customer tips including:

- User ID
- Business ID
- Tip Text
- Date
- Number of Likes

---

## Data Format

The Yelp dataset is provided in **JSON** format and contains both:

### Structured Data

- Business Information
- User Information
- Ratings
- Check-ins

### Unstructured Data

- Customer Reviews
- Tips

This makes the dataset suitable for:

- Big Data Analytics
- Machine Learning
- Natural Language Processing (NLP)
- Business Intelligence
- Data Visualization

---

# Tools & Technologies Used

| Technology | Purpose |
|------------|---------|
| **Amazon S3** | Stores raw and processed Yelp datasets in a scalable cloud storage environment. |
| **Amazon EMR** | Distributed processing of large datasets using Apache Spark. |
| **PySpark** | Large-scale data transformation, cleaning, feature engineering, and machine learning. |
| **Jupyter Notebook** | Interactive development, EDA, visualization, and experimentation. |
| **AWS Glue** | Serverless ETL pipelines and Data Catalog management. |
| **Amazon Athena** | Query data stored in Amazon S3 using standard SQL. |
| **ODBC/JDBC** | Connect Athena with Power BI and Python applications. |
| **Power BI/Tableu** | Interactive dashboards and business intelligence reporting. |
| **Git & GitHub** | Version control, collaboration, and project management. |
| **GitHub Actions** | CI/CD automation for uploading scripts to S3, triggering Glue jobs, crawlers, and workflow orchestration. |
| **Terraform / AWS CloudFormation** | Infrastructure as Code (IaC) for provisioning AWS resources such as S3, Glue, IAM, and networking components. |

---

# Project Workflow

```text
Yelp Dataset
      │
      ▼
 Amazon S3 (Raw Layer)
      │
      ▼
 AWS Glue Crawler
      │
      ▼
 AWS Glue ETL Jobs
      │
      ▼
 Amazon S3 (Silver Layer)
      │
      ▼
 AWS Glue Transformations
      │
      ▼
 Amazon S3 (Gold Layer)
      │
      ▼
 AWS Glue Catalog
      │
      ▼
 Amazon Athena
      │
      ▼
 Power BI Dashboards
```

---

# Expected Outcomes

- Centralized cloud-based data lake architecture
- Automated ETL pipeline using AWS Glue
- Distributed data processing using PySpark on Amazon EMR
- Interactive SQL analytics through Amazon Athena
- Business intelligence dashboards in Power BI/ Tableu
- Sentiment analysis of customer reviews
- Rating prediction using Machine Learning
- Customer behavior analysis
- Business performance analytics
- Actionable recommendations for business owners

---

# Key Features

- Large-scale distributed data processing
- Cloud-native architecture on AWS
- Automated ETL pipelines
- Data warehouse-ready Gold layer
- Machine Learning integration
- NLP-based sentiment analysis
- Interactive BI dashboards
- Infrastructure as Code (Terraform/CloudFormation)
- CI/CD automation using GitHub Actions

---

# Architecture Diagram

> **Insert the architecture diagram here.**

<img width="886" height="512" alt="image" src="https://github.com/user-attachments/assets/f5de5b0a-2628-44e2-85d5-39d22c36c155" />


---

# Repository Structure

```text
Yelp_Big_Data/
├── doc/                        # Documentation & Project Specifications
│   ├── Yelp_Synopsis_Complete.pdf
│   └── maheshwari/             # Architecture Screenshots & Silver Layer Steps
├── data/                       # Configs, Schemas & Sample Datasets
├── notebooks/                  # Team Member EDA Notebooks
│   ├── mihir/
│   ├── nagesh/
│   ├── saurav/
│   ├── shraddha/
│   ├── vishal/
│   ├── maheshwari/
│   └── yogesh/
├── src/                        # Production Source Code & Scripts
│   ├── ETL/                    # Member ETL Scripts
│   │   ├── mihir/
│   │   ├── nagesh/
│   │   ├── saurav/
│   │   ├── shraddha/
│   │   ├── vishal/
│   │   ├── maheshwari/
│   │   └── yogesh/
│   └── BI/                     # BI Dashboards & Scripts
│       └── shraddha/
├── presentation/               # Slide Decks & Presentation Assets
└── reports/                    # Final Project Reports
```

---

# Future Enhancements

- Recommendation System
- Fake Review Detection
- Aspect-Based Sentiment Analysis (ABSA)
- Topic Modeling using BERTopic/LDA
- Review Summarization using LLMs
- Real-time Streaming Analytics with Apache Kafka
- Predictive Business Performance Modeling

