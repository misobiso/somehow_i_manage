# Somehow I Manage 💼
Exploratory Data Analysis on The Office

## Goals
1) See how quickly I can perform EDA without a data science background
2) Exposure to data visualization library
    - Matplotlib
3) Understand different model types
    - Latent Dirichlet Allocation
    - Sentiment Analysis
    - Cluster Analysis
    - Principal Component Analysis
    - K-means Clustering

## Data
Scraped data from IMDb

**Sample Data**
| EpisodeTitle | About | Ratings | Votes | Viewership (in Millions) | Duration | Date | GuestStars | Director | Writers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pilot | The premiere episode introduces the boss and staff of the Dunder-Mifflin Paper Company in Scranton, Pennsylvania in a documentary about the workplace. | 7.5 | 4936 | 11.2 | 23 | 24 March 2005 | Ken Kwapis | Ricky Gervais \| Stephen Merchant and Greg Daniels |


## Questions
### How do IMDb ratings change over the seasons?

![image](https://user-images.githubusercontent.com/128242031/226137863-de3a9696-bc69-4d0d-a46a-4d5372ddff56.png)

> Sudden drop from season 7 to season 8 is likely caused by Michael leaving The Office

### Which episodes had the highest and lowest IMDb ratings?

![image](https://user-images.githubusercontent.com/128242031/226138589-3bc9117e-0481-4bd3-9db2-4b003897ed41.png)

```The slope of the line of best fit is -0.004078```


_*Season 7 Goodbye Michael - As the office gets ready for Michael's final day at Dunder Mifflin, Michael doesn't tell anyone that he's leaving a day early. Meanwhile, Deangelo accompanies a nervous Andy on a visit to one of Michael's old clients._

_*Season 8 Get the Girl - Andy goes to Tallahassee to tell Erin that he loves her and needs her to return to Scranton. Meanwhile, Nellie shows up in Scranton and tries to claim Andy's manager position by offering everyone raises._



> In this scenario, the slope of the line of best fit represents the trend of IMDb ratings across the episodes in the dataset. Specifically, the slope indicates the average change in IMDb ratings per episode.

> If the slope is positive, it suggests that, on average, IMDb ratings tend to increase as the episodes progress. Conversely, if the slope is negative, it indicates that IMDb ratings generally decrease as the episodes progress. If the slope is close to zero, it means that there's no strong trend in either direction, and the ratings remain relatively stable throughout the episodes.



### Which episodes had the highest and lowest viewership?

![image](https://user-images.githubusercontent.com/128242031/226137931-7466f525-3bf4-4bd8-ac43-e09101604716.png)

```The slope of the line of best fit is -0.022977```


_*Season 5 Stress Relief - Dwight's too-realistic fire alarm gives Stanley a heart attack. When he returns, Michael learns that he is the cause of Stanley's stress. To remedy the situation, he forces the office to throw a roast for him._

_*Season 9 Paper Airplane - The employees hold a paper airplane competition, Andy gets an acting role in a workplace safety video, and Jim and Pam's marriage tensions continue to build._

> Stress Relief had a large increase in viewership likely due to iconic scenes such as the "Fire Drill" and "First Aid Fail/Staying Alive CPR"

### How Does viewership correlate with IMDb ratings?

![image](https://user-images.githubusercontent.com/128242031/226139194-02285a60-9966-4125-8ea1-f950041ed6f3.png)

> A correlation coefficient of 0.49 indicates a moderate positive relationship between the two variables. In this case, it means that as the viewership of an episode increases, the IMDb rating of that episode is also likely to increase, but the relationship is not extremely strong.

> To put it simply, higher viewership episodes tend to have higher IMDb ratings, but there are other factors that may influence the ratings as well, and the relationship is not so strong that it would be accurate to predict IMDb ratings solely based on viewership.


### Are there any specific patterns or trends in episode themes (based on the "About" column) that are associated with higher ratings or viewership?

Use Latent Dirichlet Allocation

>Latent Dirichlet Allocation (LDA) is an unsupervised machine learning technique used to discover hidden topics within a collection of documents or text data. It assumes that documents are made up of a mixture of topics, and each topic is a combination of words with certain probabilities.

Here's a high-level explanation of what LDA does with the data:

1) The algorithm starts by randomly assigning each word in the documents to one of the pre-defined number of topics. At this stage, the topic assignment may not make much sense.

2) Next, it iteratively refines these topic assignments by looking at two factors:
a) How often a word appears in a topic across all documents.
b) How often a topic appears in a document.

3) The algorithm repeats this process many times, updating the topic assignments for words in the documents based on the factors above. Over time, the topics begin to take shape, and words that frequently co-occur start to cluster together in topics.

4) Once the algorithm converges, you are left with a set of topics, each represented by a group of words with associated probabilities. You can then use these topics to understand the underlying themes or patterns present in the documents.

In summary, LDA helps uncover hidden patterns (topics) in text data by looking at the co-occurrence of words within documents and identifying groups of words that frequently appear together.

```
Topic: 0 
Words: 0.026*"andy" + 0.020*"michael" + 0.019*"dwight" + 0.019*"office" + 0.014*"tries" + 0.012*"jim" + 0.011*"pam" + 0.010*"meanwhile" + 0.008*"party" + 0.008*"robert"

Topic: 1 
Words: 0.036*"michael" + 0.023*"dwight" + 0.018*"andy" + 0.014*"pam" + 0.014*"jim" + 0.013*"party" + 0.013*"new" + 0.013*"dunder" + 0.013*"mifflin" + 0.011*"angela"

Topic: 2 
Words: 0.035*"michael" + 0.029*"office" + 0.020*"dwight" + 0.014*"jim" + 0.014*"tries" + 0.014*"meanwhile" + 0.011*"šã" + 0.008*"gets" + 0.008*"pam" + 0.008*"andy"

Topic: 3 
Words: 0.034*"michael" + 0.026*"jim" + 0.016*"dwight" + 0.016*"pam" + 0.016*"office" + 0.013*"mifflin" + 0.013*"dunder" + 0.012*"andy" + 0.011*"new" + 0.009*"day"

Topic: 4 
Words: 0.029*"michael" + 0.026*"jim" + 0.023*"pam" + 0.023*"dwight" + 0.016*"office" + 0.011*"baby" + 0.011*"andy" + 0.010*"mifflin" + 0.010*"dunder" + 0.009*"tries"
```

These outputs represent the top 10 keywords for each topic generated by the LDA model. Each topic is a mixture of keywords with associated probabilities (weights). The higher the probability of a keyword in a topic, the more important the keyword is in defining that topic.

Here's a simplified interpretation of the topics based on the keywords:

1) Topic 0: This topic seems to revolve around Andy, Michael, Dwight, and the office environment. It also includes themes like trying something, parties, and interactions with Robert.
2) Topic 1: This topic is more focused on Michael, Dwight, Andy, Pam, and Jim, as well as parties, and Dunder Mifflin as a company. It also includes Angela as a character.
3) Topic 2: In this topic, Michael, the office, Dwight, and Jim are prominent. It seems to include situations where someone is trying something and interactions with other characters.
4) Topic 3: This topic is centered around Michael, Jim, Dwight, Pam, and the office, along with Dunder Mifflin as a company. It also involves themes related to new things or experiences and daily activities.
5) Topic 4: The main characters in this topic are Michael, Jim, Pam, and Dwight. The office environment and Dunder Mifflin are also present. It introduces themes like babies and trying something.

Keep in mind that LDA topics are not explicitly defined, and their interpretation is based on the keywords that have the highest probabilities within each topic. The simplified interpretation provided above is based on the assumption that the main characters and their interactions are the primary drivers of the topics.

![image](https://user-images.githubusercontent.com/128242031/226140392-10882df2-2130-4e7b-8b32-cc8f3799d3bd.png)

### Use the sentiment analysis tool to determine if episodes with positive or negative sentiment in the summaries have an impact on ratings or viewership

1) We used a tool called "SentimentIntensityAnalyzer" from nltk to measure the sentiment of the episode summaries. Sentiment is a way to determine if the text has a positive, negative, or neutral tone.

2) We created two scatter plots to visualize the relationship between the sentiment scores and IMDb ratings, and between sentiment scores and viewership. This helps us see if there is any connection between the tone of the episode summaries and their ratings or viewership.

![image](https://user-images.githubusercontent.com/128242031/226140659-ec468a4f-92da-4185-ba40-107fcb032da9.png)

3) We calculated the correlation coefficients to measure the strength of the relationship between sentiment scores and IMDb ratings, as well as sentiment scores and viewership. A correlation coefficient is a number between -1 and 1 that indicates how closely related two variables are.

```
Correlation between sentiment scores and IMDb ratings: -0.11642027169707882
Correlation between sentiment scores and viewership: -0.1013892861047804
```

These correlation coefficients tell us about the relationship between sentiment scores and IMDb ratings, and sentiment scores and viewership.

1) Correlation between sentiment scores and IMDb ratings: -0.116
This negative correlation coefficient indicates that there is a very weak negative relationship between sentiment scores and IMDb ratings. This means that when the sentiment score is higher (more positive), the IMDb ratings tend to be slightly lower, and vice versa. However, the relationship is very weak, so it's not a strong or reliable predictor.

2) Correlation between sentiment scores and viewership: -0.101
This negative correlation coefficient also indicates a very weak negative relationship between sentiment scores and viewership. This means that when the sentiment score is higher (more positive), the viewership tends to be slightly lower, and vice versa. But again, the relationship is very weak, so it's not a strong or reliable predictor.

In both cases, the correlation coefficients are close to 0, which means there is **little to no relationship between the variables**. In simpler terms, the sentiment scores of the episode summaries don't seem to have a significant impact on either the IMDb ratings or the viewership of the episodes.

### Can we use clustering algorithms (e.g., K-means) to group episodes based on their features (e.g., ratings, viewership, topics, sentiment) and identify common patterns among the clusters?

Steps
1) Prepared the features: We combined IMDb ratings, viewership, topics, and sentiment scores into a single DataFrame, where each row represents an episode, and each column represents one of the features.

2) Scaled the features using StandardScaler: Scaling is the process of transforming the data so that all features have the same importance. Since features like ratings and viewership have different ranges, they need to be scaled to give equal weight to all features. We used the StandardScaler, which standardizes the features by subtracting the mean and dividing by the standard deviation. This ensures that all features have a mean of 0 and a standard deviation of 1.

3) Reduced dimensions with PCA: Principal Component Analysis (PCA) is a technique used to reduce the dimensions of the data while preserving most of the information. It creates new features called "principal components" by combining the original features in a way that captures most of the variation in the data. By reducing the dimensions, we can visualize and analyze the data more easily. In our case, we transformed the multi-dimensional data into a two-dimensional space.

    Imagine you have a dataset with multiple features (columns), and these features are spread across a multi-dimensional space. Principal Component        Analysis (PCA) is a technique that helps you reduce the number of dimensions while keeping as much information as possible from the original dataset.

    The first principal component (PC1) is a new dimension that captures the most variance (or the most significant pattern) in the dataset. The second     principal component (PC2) is another new dimension, orthogonal (or perpendicular) to the first one, that captures the second most variance in the       dataset, and so on.

    By using PCA and reducing the dimensions, you can simplify the dataset and visualize it more easily, for example, on a two-dimensional scatter plot.    Keep in mind that some information is lost during the dimensionality reduction process, but PCA tries to minimize that loss by preserving the most      important patterns.

4) Applied K-means clustering: K-means is a clustering algorithm that groups data points (in our case, episodes) based on their similarity across the given features. It starts by randomly initializing a set number of cluster centers (we chose 3) and assigns each data point to the nearest center. Then, it iteratively updates the centers by calculating the mean of all data points in each cluster and reassigning data points to the nearest updated center. The algorithm stops when the cluster assignments no longer change or a maximum number of iterations is reached.

5) Evaluated the clusters: We analyzed the clusters by calculating the average IMDb ratings, viewership, and sentiment scores for each cluster. This allowed us to identify the common patterns among the clusters.

```
This output shows that the episodes have been divided into three clusters based on their features (ratings, viewership, topics, and sentiment). Here's a summary of each cluster:

Cluster 0:

Number of episodes: 48
Average IMDb rating: 8.35
Average viewership: 7.07 million
Average sentiment score: -0.0048

Cluster 1:

Number of episodes: 104
Average IMDb rating: 8.17
Average viewership: 7.26 million
Average sentiment score: 0.0359

Cluster 2:

Number of episodes: 36
Average IMDb rating: 8.28
Average viewership: 7.45 million
Average sentiment score: -0.0179

Cluster 0 has the highest average IMDb rating and the lowest average viewership, with a slightly negative average sentiment score. Cluster 1 has the largest number of episodes, with the lowest average IMDb rating but slightly higher average viewership than Cluster 0, and a slightly positive average sentiment score. Cluster 2 has the smallest number of episodes, with a relatively high average IMDb rating and the highest average viewership, along with a negative average sentiment score.

These clusters can help you identify any patterns or relationships between the episodes' features. For example, Cluster 0 has the highest IMDb ratings but lower viewership, while Cluster 2 has a high IMDb rating and the highest viewership. Cluster 1, with the most episodes, has the lowest IMDb rating but a slightly positive sentiment score. This information can help you understand how different aspects of the episodes are related and inform further analysis or decision-making based on your interests.
```

6) Visualized the clusters: We created a scatter plot to visualize the clusters in the two-dimensional space created by PCA. This plot helps us see the groups of episodes and their separation based on the given features.

![image](https://user-images.githubusercontent.com/128242031/226153064-e7ba19e9-f2e0-4368-9fbb-2854d83b5d4c.png)

> By following these steps, we were able to group episodes based on their features and identify common patterns among the clusters. The key steps, scaling the features, PCA, and K-means clustering, ensure that our analysis is accurate, easy to interpret, and captures meaningful patterns in the data.

Random 3 episodes in the clusters

```
Cluster 0 Episodes:
-------------------
Season 4 - Goodbye, Toby
About: Michael throws an extravagant going-away party for Toby, and falls in love with the woman who is replacing him. Jim plans to propose to Pam at the party, but gets out-staged. Back at corporate, Ryan is arrested for fraud.

Season 5 - Casual Friday
About: Michael returns to Dunder Mifflin and brings Ryan and Pam in as salespeople, but the existing salespeople demand that Michael return the clients they stole back to them. Meanwhile, some employees take the idea of Casual Friday too far.

Season 8 - Angry Andy
About: Andy and Erin return to Dunder Mifflin, only to discover that Nellie has taken his job and office and refuses to give them up. Meanwhile, Ryan becomes jealous after Pam tries to play matchmaker for Kelly.

Cluster 1 Episodes:
-------------------
Season 1 - Hot Girl
About: Michael is just one of the many male staff who start vying for the attention of an attractive saleswoman in the office.

Season 2 - The Injury
About: Michael's "injury" from a George Foreman Grill distracts the staff from Dwight, the one with the real injury.

Season 3 - Branch Closing
About: When Jan tells Michael that the Scranton Branch will be shutting down, Michael strives to keep his staff's spirits up. Meanwhile, everyone privately begins to envision how their lives will change in the aftermath.

Cluster 2 Episodes:
-------------------
Season 2 - Valentine's Day
About: When Michael visits Dunder Mifflin corporate headquarters in New York on Valentine's Day, he and Jan are both in for a surprise. Meanwhile, back in Scranton, the office staff celebrates Valentine's Day grade school style.

Season 7 - Threat Level Midnight
About: Michael finally finishes his movie "Threat Level Midnight" and screens it for his employees at Dunder Mifflin, but Holly is less than enthusiastic for his finished product.

Season 9 - Suit Warehouse
About: Dwight and Clark pose as father and son to get an account, Darryl goes for an interview with Jim's new business, and everybody in the office goes crazy over a new espresso machine.
```
