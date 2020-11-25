## Brunch-Article-Recommendation-System 
### Problem Description
Brunch is a platform for connecting people who loves reading and making contents. To make users find their taste, they designed article recommendation systems. For a better user experience, which can be made through personalized recommendation, they've been consistently developing their system. 

Now that many quality data is in Brunch, by using those, we need to make customized unique recommendation system.

### Objective of project
- In this project, we developed content recommendation system. First of all, we downloaded data in kakao arena that hosts brunch recommendation contests. Then we 

#### Data Sources : [Kakao Arena](https://arena.kakao.com/c/6)
- data overview
    - articles(author, issue-date, title, contents, etc)
    - reader/ author information(read-date, read-article, following author list, etc) 

### Project detail
#### 1. Period limitation
- It is found that within 2 weeks of the publication of the article, most consumption took place. In other words, users tended to read more recently published articles. So, We decided to recommend articles to target users within two weeks of the issued date.

#### 2. Target segmentation
- The number of articles read by users during 1 months is divided into three groups using descriptive statistics.
    - group1 (0~7) : passive user 
    - group2 (8~64): active user
    - group3 (65~ ): domain worker, expert, crawler, etc.


#### 3. Recommendation algorithms
- We tried to apply the recommendation system to each group in a different way   
  - Users in group1 will recommended by their following author list, magazine list they've read and recent popular articles   
  - Users in group2 will recommended by similarity of their taste with other users or authors using cosine similarity   
  - Users in group3 read more 65 articles within 2weeks. we considered them to be realted. we will recommend them to recent & popular articles except for they read 
