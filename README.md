# Brunch-Article-Recommendation-System 
### Problem Description
Brunch is a platform for connecting people who loves reading and making contents. To make users find their taste, they designed article recommendation systems. For a better user experience, which can be made through personalized recommendation, they've been consistently developing their system. 

Now that many quality data is in Brunch, by using those, we need to make customized unique recommendation system.

### objective of project
- Implementing customized content recommendation system.

### Data Sources : [Kakao Arena](https://arena.kakao.com/c/6)
    - contents
    - reader/ author information
    - user behavior
---
># project detail
### *project execution procedure*
- Repeat EDA and modelling and upgrade model

### *1st modeling* 
### contents based and Collaborative filtering for all users
- Content-based recommendation and collaborative filtering methods were attemped
- Massive data makes analysis difficult

### *2nd modeling*
### period limitation
- We find that within 14 days of the publication of the article, most consumption takes place.

- Users tend to read more up-to-date articles
- we would like to recommend registered articles to target users within two weeks of the registration date.

### *3rd modeling*
### target segmentation
- The number of articles read by target users during validation periods is divided into three groups using descriptive statistics.

- We tried to apply the recommendation system to each group in a different way   
  - Users in group1 will recommended by their following author list, magazine list they've read and recent popular articles   
  - Users in group2 will recommended by similarity of their taste with other users or authors using cosine similarity   
  - Users in group3 read more 65 articles within 2weeks. we considered them to be realted. we will recommend them to recent & popular articles except for they read 
