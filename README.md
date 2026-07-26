# Yelp Review Data Analytics Using Big Data Technologies

## 📌 Overview
This project analyzes the **Yelp Open Dataset 2024** using Big Data technologies to uncover insights from customer reviews, ratings, and business information.  
The goal is to perform **sentiment analysis, rating prediction, and business performance evaluation** to help businesses make data-driven decisions.

## 👥 Team
- Nagesh Khichede  
- Saurav Makde  
- Shraddha Patil  
- Mihir Zope  
- Maheshwari Phalke  
- Vishal Dewanjee  
- Yogesh Thakre  

## 🎯 Objectives
- Analyze customer reviews, ratings, and business data.  
- Perform **Exploratory Data Analysis (EDA)** to identify trends.  
- Conduct **sentiment analysis** on customer opinions.  
- Build **ML models** to predict ratings/review sentiment.  
- Study user behavior and business performance.  
- Create **interactive Power BI dashboards**.  
- Generate actionable insights for customer satisfaction.  

## 📂 Dataset
The project uses the **Yelp Complete Open Dataset 2024**, including:
- `business.json` (~118 MB, 150K records)  
- `review.json` (~5.3 GB, 7M records)  
- `user.json` (~3.2 GB, 2M records)  
- `checkin.json` (~290 MB, 132K records)  
- `tip.json` (~250 MB, 909K records)  

Data is provided in **JSON format** with both structured and textual fields.

## 🛠️ Tools & Technologies
- **AWS S3** – Data storage  
- **AWS EMR + PySpark** – Distributed data processing  
- **AWS Glue** – ETL & data cataloging  
- **Amazon Athena** – SQL queries on S3  
- **Power BI** – Interactive dashboards  
- **Git & GitHub** – Version control & collaboration  
- **GitHub Actions** – CI/CD automation  
- **Terraform / CloudFormation** – Infrastructure as Code  

## 📊 Architecture
1. **Data Storage** → Raw Yelp data stored in S3  
2. **Processing** → EMR + PySpark for cleaning & ML pipelines  
3. **ETL** → AWS Glue jobs for integration & cataloging  
4. **Analytics** → Athena queries + BI dashboards  
5. **Visualization** → Power BI for insights  

## 🚀 Deliverables
- Cleaned & processed Yelp dataset  
- ML models for sentiment/rating prediction  
- Power BI dashboards with interactive insights  
- Documentation & reports  
