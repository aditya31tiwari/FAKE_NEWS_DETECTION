**#DetectoNews: A Smart System for Automatic Fake News Detection**

**19/09/25** -- Trained our first ever model, a basic one but we had to start somewhere. below is the accuracy chart:

![WhatsApp Image 2025-09-19 at 21 23 19_0fa6d340](https://github.com/user-attachments/assets/40716a04-9370-43e6-b259-8690c8f624d1)

**20/09/25** -- OK, so everything was going fine, but while testing I noticed the model often flagged *well-written fake news as real*, and *odd-sounding but true news as fake*.  
After some research, I realized this might be due to the dataset we were using (*ISOT Dataset*). Since ISOT true news mostly comes from Reuters (formal, factual style) and fake news comes from clickbait sources (sensational, emotional style), the model was learning **writing style instead of truthfulness**.  
To tackle this, we decided to *expand the horizon* of our dataset. Today we picked **two more datasets** along with ISOT, cleaned and preprocessed them, and created one large **final_merged_dataset**.  
The merged dataset now has **~50k true articles and ~50k fake articles** (~110k total), giving us a much broader and balanced foundation for training.  

**22/09/2025** -- Trained model based on Naive Bayes and SVM classifier, using the new integrated merged dataset. There integration with the UI is still left.

**NaiveBayes_V2**'s accuracy, f1 score, presicion, etc:--


![WhatsApp Image 2025-09-21 at 18 09 09_0324e1b9](https://github.com/user-attachments/assets/1cae82cc-5be6-4299-8e7a-a0616c49ed2c)


**SVM**'s accuracy, f1 score, presicion, etc:--


![WhatsApp Image 2025-09-21 at 18 12 43_f0487682](https://github.com/user-attachments/assets/4741d8ab-6820-4be0-ab52-05c32ec34c53)


**DATE: 03-11-2025**
We worked on the final_mergerd_dataset.csv file which was used in the Phase1 of the project.
We were trying to make it BERT ready which is the big part of our next phase, i.e. Phase2, now, this phase will focus on showing that transformer based model will outperform baseline models on the same dataset. We are using BERT embeddings with XGBoost to do so.
