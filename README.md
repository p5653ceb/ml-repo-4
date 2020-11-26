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
- we found that within 2 weeks of the publication of the article, most consumption took place. Users tended to read more recently published articles. Therefore, we decided to recommend articles to target users within two weeks of the issued date.

#### 2. Target segmentation
- The number of articles read by users during 1 months is divided into three groups using descriptive statistics.
    - group1 (0~7) : passive user (who read below average number of articles)
    - group2 (8~64): active user ( who read above average number of articles)
    - group3 (65~ ): domain worker, expert, crawler, etc. ( who read above upper-fence number of articles)

#### 3. Recommendation algorithms
- Articles of following author
    - 98 % of all users have author list who follow.
    - Each users follow an average of 8.6 authors. 
    - Recommend users recent article published by author they subscribe to.
    
- Articles of magazine
    - Article in magazines tends to be more read by users than not in.
    - New users are likely to read articles in Brunch magazines.
    - Recommend users another popular&recent articles in magazine where they read articles at least once.
    
- Articles based on similar tastes(collaborative filtering)
    - collaborative filtering

    i) reader-author :  
    ii) reader-reader : 

- Articles of Popular & Recent
    - 

