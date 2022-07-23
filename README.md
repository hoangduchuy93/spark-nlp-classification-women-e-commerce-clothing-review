![image](https://user-images.githubusercontent.com/91864024/180586422-faca150b-9c25-478e-9d1f-3eb77f248911.png)
# Spark NLP Classification for Women E-commerce Clothing Review
## I. Outline
- Women will leave reviews for the products they have purchased. 
- Model must classify the level of customer satisfaction.
- Implemented on Spark, using NLP algorithms and classification algorithms to process buyer comments.
## II. Business Objective/ Problem
- Let's say you work in the Data Science department of fashion company X. They specialize in selling women's clothing items on an e-commerce platform.
- Company X wants to expand its business by finding out how customers feel after purchasing from the company through customer ratings, comments. Thereby finding ways to better serve customers.
- This classification project is built based on that request.
## III. Project implementation
### 1. Business Understanding
Based on the above description => identify the problem:
- Find solutions to improve advertising effectiveness, thereby increasing sales, improving customer satisfaction.
- Objectives/problems: build a customer classification ratings to understand more about customers. By gathering information about customer sentiment, they hope to improve business and care strategy.
- Applied methods:
  - Working environment: Spark
  - Using NLP processes to handle text.
  - Using classification models to make prediction: LogisticRegression, NaiveBayes, RandomForestClassifier, DecisionTreeClassifier, ...
### 2. Data Understanding/ Acquire
- This is a Women’s Clothing E-Commerce dataset revolving around the reviews written by customers. 
- Because this is real commercial data, it has been anonymized, and references to the company in the review text and body have been replaced with “retailer”.
- You can download dataset at: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews?resource=download
- There are 2 working sheets: "Reviews" to train your model and "new_reviews" to make predictions.
- This dataset includes 23486 rows and 10 feature variables. Each row corresponds to a customer review, and includes the variables:
  - Clothing ID: Integer Categorical variable that refers to the specific piece being reviewed.
  - Age: Positive Integer variable of the reviewers age.
  - Title: String variable for the title of the review.
  - Review Text: String variable for the review body.
  - Rating: Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
  - Recommended IND: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
  - Positive Feedback Count: Positive Integer documenting the number of other customers who found this review positive.
  - Division Name: Categorical name of the product high level division.
  - Department Name: Categorical name of the product department name.
  - Class Name: Categorical name of the product class name.

![image](https://user-images.githubusercontent.com/91864024/180587620-484da30d-9395-421a-8b07-48b3b51ef713.png)

### 3. Build model
**3.1. Understand the dataset:**
![image](https://user-images.githubusercontent.com/91864024/180587769-e29904cf-b9ee-4ef6-bc58-c54760493823.png)

**a. Age feature:**

![image](https://user-images.githubusercontent.com/91864024/180587899-470b5ed3-f646-4ebe-a977-70376ee8f33b.png)

- From the dataset we can realize that:
  - Women age's range from 18 - 99. Average age around 43.
  - We can see that their age mainly from 30 - 50. This is the group of customer that we should pay more attention.

**b. Rating feature:**

![image](https://user-images.githubusercontent.com/91864024/180588030-5133d438-ea28-4954-9c0c-0959025daf83.png)

- From the dataset we can realize that:
  - Women rating range from 1 - 5.
  - Most ratings are in range 4 - 5. Are most of the customers who give reviews satisfied with the service or are they easygoing people?
**3.2. Pre-processing data:**



