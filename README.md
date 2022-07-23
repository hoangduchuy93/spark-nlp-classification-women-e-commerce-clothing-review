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

**- Check NaN, Null, duplicate from dataset:**

![image](https://user-images.githubusercontent.com/91864024/180588322-d747649e-72cd-4df1-99c9-a56fc870ca85.png)

==> there is no NaN, Null. There are some duplicated rows, we remove them.

**- Remove unnecessary words in Review Text: emoji, number, link, not characters, ...**

![image](https://user-images.githubusercontent.com/91864024/180588536-c0081384-4257-49c2-b11c-02b278940941.png)

**3.3 Build model:**

**3.3.1. Create pipeline for your model:**
- Tokenizer
- StopWordRemover
- CountVectorizer
- TF-IDF
- VectorAssembler

![image](https://user-images.githubusercontent.com/91864024/180588579-7df1ce80-dcf8-465c-8fff-ba037bfb135a.png)

**3.3.2. Make pipeline, transform your dataset:**

![image](https://user-images.githubusercontent.com/91864024/180588781-71ce9420-95ac-4189-afd2-09634379e04b.png)

**3.3.3. Split train/test:**

![image](https://user-images.githubusercontent.com/91864024/180588809-843c8ce8-b6a1-4c53-9ae0-7c51814cc721.png)

**3.3.4. Applied classification model:**

- Apply classification model: LogisticRegression, NaiveBayes, RandomForestClassifier, DecisionTreeClassifier
- Fit training and test on test set.

![image](https://user-images.githubusercontent.com/91864024/180588844-84d453cc-809e-447c-9797-c16fe46e000e.png)

**3.3.5. Evaluate accuracy score of these models:**

![image](https://user-images.githubusercontent.com/91864024/180588941-81cf246d-9eff-450e-8740-b328f8fa1473.png)

Comment: Logistic Regression algorithm gives higher accuracy score than other algorithms, but accuracy score is still low => consider grouping the Rating column into fewer groups (maybe 3 groups: positive, neutral, negative. Now there are 5 groups of rating from 1 to 5). 

**3.3.6. Groups for Rating column:**
- Create column Rating_idx where:
  - Rating >= 4: Positive
  - Rating <= 2: Negative
  - Otherwise: Neutral

![image](https://user-images.githubusercontent.com/91864024/180589011-b25fb271-80a2-48f1-a53d-113f907a91e3.png)
![image](https://user-images.githubusercontent.com/91864024/180589019-44245245-2b0c-4572-8174-9d4177f09ffe.png)

Comments: users tend to give Positive rating more than Neutral and Negative

**3.3.7. Applied model:**
- Repeat step 3.3.1 to 3.3.6 for new Rating_idx column.
- Accuracy score of new models are shown below:

![image](https://user-images.githubusercontent.com/91864024/180589135-b0154828-f69d-49d6-9cb7-8e152b044834.png)

Comment: these models give better results. Logistic Regression accuracy coefficient gives higher results than other algorithms => Choose Logistic Regression algorithm

**3.3.8. Check confusion matrix of Logistic Regression:**

![image](https://user-images.githubusercontent.com/91864024/180589204-7b955f5a-7e45-4398-bda0-f8fbd311d85a.png)
![image](https://user-images.githubusercontent.com/91864024/180589241-d722e795-f2bc-4f7d-bde6-ea31f568526d.png)

Comment: The accuracy, precision and recall give better results => Logistic Regression can be used for this dataset.

### 4. Predict on new comments (from "new_reviews" sheet)
- We have 5 sentences to predict customer ratings.
- They are:
  - Sentence 1: Dress runs small esp where the zipper area runs  ordered the sp which typically fits me and it was very tight  the material on the top looks and feels very cheap that even just pulling on it will cause it to rip the fabric  pretty disappointed as it was going to be my christmas dress this year  needless to say it will be going back                                                                                                        
  - Sentence 2: Nice top  armholes are bit oversized but as an older woman  am picky about that  the print is pretty and unusual  it just did not look great on me  there is slight peplum in the back that hangs nicely  it is lightweight tee fabric that is opaque  tried it on with black bra which was barely visible  great for warmer climates but there are so many gorgeous tops out now  that decided to return since summer is winding down  do recommend 
  - Sentence 3: Was really excited for this dress but should have paid more attention to the material it was made with  for the price of the dress  it felt very cheap  did not even end up trying it on after opened the packaging  the colors were not as vibrant as the picture  as another reviewer mentioned  it was more similar to something you would purchase off boardwalk  very disappointed and will be returning                                        
  - Sentence 4: If you are going for ridiculously high priced ugly sweater contest  this is the one for you  normally like clothing with some character and juxtaposition  but this one did not do it for me  cannot imagine the collar fitting right or flattering anyone  and the mixed layers end up making it look cheap rather than trendy     
  - Sentence 5: saw this online and immediately purchased the top in gray  it is so easy and casual but the shoulder detailing give it something different and unique for regular gray shirt  the fit is loose and comfortable but not overly big  just right   can not wait to pair it with my new white jeans for summer

Just be looking, I can tell that:
- Sentence 1: unsatisfied
- Sentence 2: satisfied
- Sentence 3: unsatisfied
- Sentence 4: unsatisfied
- Sentence 5: satisfied

We have defined emotion values as:
- Rating >= 4: Positive (0)
- Otherwise: Neutral (1)
- Rating <= 2: Negative (2)
                                                                                                                 
This is what our model predicted:

![image](https://user-images.githubusercontent.com/91864024/180589593-37e70746-c608-4a4f-b4ba-65e17098b85a.png)


| Sentence No. | Actual | Prediction| Different |
|--------------|-------------|-------------|-------|
| Sentence 1 | unsatisfied | unsatisfied |  No |
| Sentence 2 | satisfied | satisfied |  No |
| Sentence 3 | unsatisfied | unsatisfied |  No |
| Sentence 4 | **unsatisfied** | **neutral** |  **Yes** |
| Sentence 5 | satisfied | satisfied |  No |

Comment: we notice that model can predict quite well, overall 4/5 cases are correct. For the "neutral" case (actual is unsatisfied), it is hard for model to predict. Since "neutral" to somebody can be "It's Okay", but for others may be "Nope, it's not Okay"

Thank you for your experience with my project. Hope you enjoy it!














