**Yelp Review Data Analytics Using Big Data Technologies 
**

**Basic Details 
**

Title: Yelp Review Data Analytics Using Big Data Technologies 
Duration: 2 Months 
Team Size: 7 
Team Members 
•	Nagesh Khichede
•	Saurav Makde
•	Shraddha Patil 
•	Mihir Zope
•	Maheshwari Phalke
•	Vishal Dewanjee
•	Yogesh Thakre
Problem Statement 
Online businesses receive a large volume of customer reviews every day, making it difficult to manually analyze customer feedback and identify the factors affecting customer satisfaction and business performance. This project uses the Yelp Complete Open Dataset 2024 to analyze customer reviews, business information, user behavior, and ratings using Big Data Analytics, Machine Learning, and Business Intelligence techniques. The goal is to uncover valuable insights, predict business ratings, understand customer sentiment, and help businesses make data-driven decisions to improve customer experience and overall performance.

Project Objectives 
1.	Analyze customer reviews, ratings, and business data. 
2.	Perform Exploratory Data Analysis (EDA) to identify trends and patterns.
3.	Conduct sentiment analysis to understand customer opinions.
4.	Build Machine Learning models to predict ratings or review sentiment. 
5.	Analyze user behavior and business performance.
6.	Create interactive Power BI dashboards for data visualization.
7.	Generate actionable insights to improve customer satisfaction and business decisions

Dataset Description 
The Yelp Complete Open Dataset 2024 consists of the following datasets:
Table	Records	Columns	Approx. File Size
business.json	150,346	14	~118 MB
review.json	6,990,280	9	~5.3 GB
user.json	1,987,897	22	~3.2 GB
checkin.json	131,930	2	~290 MB
tip.json	908,915	5	~250 MB

•	Business Information: 
•	Contains details about businesses, including business ID, name, address, location, categories, operating hours, star rating, and total review count. 
•	Review Data: 
•	Contains customer reviews with review ID, user ID, business ID, star rating, review text, review date, and feedback votes (useful, funny, and cool). 
•	User Data: 
•	Stores user information such as user ID, review count, average rating given, friends, fans, compliments, and account activity. 
•	 Check-in Data: 
•	Records customer check-in dates and times at businesses, helping analyze customer visit frequency and business popularity. 
•	Tip Data: 
•	Contains short customer tips and suggestions along with the user ID, business ID, date, and the number of likes received. 
•	Data Format: 
•	The dataset is provided in JSON format and contains both structured (business, user, ratings) and textual (reviews and tips) data, making it suitable for EDA, Machine Learning, NLP, and Business Intelligence applications.
Tools & Technologies Used 
•	Amazon S3 (Simple Storage Service): Used for storing raw and processed Yelp datasets in a scalable, durable cloud environment. 
•	Amazon EMR (Elastic MapReduce): Managed cluster platform used to process largescale data using distributed computing frameworks like PySpark. 
•	PySpark (on EMR): Used for distributed data processing, transformation, and building machine learning pipelines on large-scale data. 
•	Jupyter Notebook: For running PySpark interactively, visualizing results, and building exploratory analytics and ML models. 
•	AWS Glue: For serverless data integration, ETL (Extract, Transform, Load), and data cataloging to prepare data for analytics. 
•	Amazon Athena: An interactive query service used to analyze data directly in S3 using standard SQL via ODBC/JDBC connections. 
•	ODBC/JDBC Connections: To enable integration between Athena and external tools like BI dashboards or Python notebooks. 
•	Power BI: Business intelligence for creating interactive dashboards. Connected to Athena via ODBC/JDBC. 
•	Git & GitHub: Version control and collaboration; codebase, notebooks, scripts, and documentation maintained in a GitHub repository. 
•	GitHub Actions (YAML + Python): CI/CD automation to upload scripts to S3, trigger Glue jobs & crawlers, and orchestrate the pipeline. 
•	Terraform / CloudFormation: Infrastructure as Code to provision S3, Glue, IAM, and related resources. 
 

Architecture Diagram
 

