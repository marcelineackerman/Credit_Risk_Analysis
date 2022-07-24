# Credit Risk Analysis

## Overview
The overall purpose of this analysis was to utilize data from LendingClub, a peer-to-peer lending services company, to predict credit risk using the following machine learning analytical model algorithms, from Python's imbalanced-learn library:

 - `RandomOverSampler`
 - `SMOTE`
 - `ClusterCentroids`
 - `SMOTEENN`
 - `BalancedRandomForestClassifier`
 - `EasyEnsembleClassifier`
 
Each model was run using the same data and then evaluated for performance and accuracy at predicting credit risk.

## Results

To determine the results, we will look at the Balanced Accuracy Score (BAS) of each model, as well as the Imbalanced Classification Report (ICR) from each model, primarily focusing on the F-score(f1 column) of the total/average, and the high-risk row, as we are primarily focused on detecting high credit risk individuals.

Results will be displayed in ascending order, starting at the worst-performing model according to BAS and ending at the best.

### Cluster Centroid Undersampling

 - **BAS:** 51.1%
 ![image](https://user-images.githubusercontent.com/100869713/180656426-3025ee00-3479-45e4-a098-63df0935e4c0.png)

 - **ICR:**
 ![image](https://user-images.githubusercontent.com/100869713/180656476-418252bb-8273-47e7-b0d4-65e0da3f72c2.png)

As demonstrated here, the Cluster Centroid Undersampling model performed the worst, with an accuracy score of 51%, meaning the model was effectively equivalent to a coin-flip in accuracy of predicting high credit risk individuals. Also of note is the extremely low F-score (0.01) for high-risk predictions, with a total of only 0.56 overall.

### Combination (SMOTEENN) Sampling

- **BAS:** 62.0%
![image](https://user-images.githubusercontent.com/100869713/180656640-9c6ebeae-c8ff-4d6b-a516-23774fba52da.png)

- **ICR:**
![image](https://user-images.githubusercontent.com/100869713/180656651-06c43f73-95d0-46c5-8c54-7a92bf09ff28.png)

Next-worst is the SMOTEENN model, with an accuracy score of only 62%, a little less than 2/3rds accurate at predicting high credit risk. The F-score is barely better than the Cluster Centroid model, with high risk only at 0.02 and totaling only 0.70

### SMOTE Oversampling

- **BAS:** 62.8%
![image](https://user-images.githubusercontent.com/100869713/180656814-e6cc53d4-b1ae-417f-bb15-00a3f8c4923f.png)

- **ICR:**
![image](https://user-images.githubusercontent.com/100869713/180656822-52bad8b1-acea-4166-b1f6-9c5e5ed61017.png)

Barely better than the SMOTEENN model, the SMOTE model comes up next with a similarly bad BAS of 62.8%, still a little less than 2/3rds accurate. The F-score total is only slightly better at 0.77 while the high risk F-score sits at 0.02 as well.

### Naive Random Oversampling

- **BAS:** 62.9%

- **ICR:**
![image](https://user-images.githubusercontent.com/100869713/180656900-15b4b078-ec2c-4df3-8d27-b38c6068f281.png)

An even *smaller* increase in accuracy comes from the Random Oversample prediction model, with a BAS of 62.9%. The high risk F-score matches the SMOTE model at 0.02 while we see a relatively good F-score of 0.81 total. Still not great, but better than the others by a smidge.

### Balanced Random Forest Classifier

- **BAS:** 78.9%

- **ICR:**
![image](https://user-images.githubusercontent.com/100869713/180657087-4b7c2ec6-2902-42b4-9432-fa1df393b699.png)

Our second-best model is the Balanced Random Forest Classifier (BRFC) algorithm. With a BAS of 78.9%, we see a moderate increase in accuracy above the rest of the models thus far, but it is still at a risky level, barely better than 3/4ths accurate at predicting high risk individuals. The F-score sees moderate increases, with high risk at 0.06 and total at 0.93. While the total accuracy is relatively good, if we are only predicting 6% of high-risk individuals accurately, then the model is still failing.

### Easy Ensemble AdaBoost Classifier (EEAC)

 - **BAS:** 93.2%
 ![image](https://user-images.githubusercontent.com/100869713/180657191-5cba1fa3-7243-4f80-bfa4-b98d214df58b.png)

 - **ICR:**
 ![image](https://user-images.githubusercontent.com/100869713/180657199-f3b57877-a87e-4a03-8f7b-f1e20accbf44.png)
 
 Our best model by a mile utilizes the Easy Ensemble algorithm. At 93% accuracy, it blows all of the other models out of the water. However, the F-score for high-risk individuals is still extremely low at 0.16, even if the total is at an impressive 0.97.
 
## Summary

In conclusion, even with a large dataset and several models, credit risk still proves to be extremely difficult to predict. Even our best model, the Easy Ensemble algorithm, could only achieve a high-risk F-score of 0.16, even with accuracy and total F-score above 90%. If creditors must use an algorithm, they should pick the EEAC model, but none of these models are particularly prescient when it comes to who is safe to lend to.


