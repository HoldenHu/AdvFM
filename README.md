# AdvFM (Adversarial Factorization Machine)

This is the PyTorch implementation for AdvFM proposed in the paper [Automatic Feature Fairness in Recommendation via Adversaries](), In Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, 2023. 
The AdvFM model can be found in the `model` folder.

Contributors: [Yiming Cao](https://github.com/caoymg), [Hengchang Hu](https://holdenhu.github.io/)



### Preprocess

- Download the raw data from the orginal source, e.g., ml_100k (https://grouplens.org/datasets/movielens/100k/)
- User feature enriched recommendation datasets include movie dataset MovieLens-100K (user gender, occupation, and zip code), and image dataset Pinterest (user preference categories). Item feature enriched recommendation datasets include movie dataset MovieLens-100K (movie category, and release timestamp), and business dataset Yelp (business city, star). 
- We filtered out the user with more than 20 interactions in Yelp, and randomly selected 6,000 users to construct our Pinterest dataset. 
- We convert all continuous feature values into categorical values, and consider the user and item IDs as additional features. Frequency and combination variety is calculated for each feature.
  - Frequency indicates the occurrence rate of the value concerning its feature domain.
  - Combination variety indicates the number of diverse samples where the value co-occurs with other features in combination. [One naive calculation method: frequency of  feature value v / number of other feature values that  co-occurs with v. A more accurate method should consider the frequency of other feature values with which v  co-occurs.]
- For the train–test data split, we employ the standard leave-one-out.



### How to Run

Use `process.py` to run the model.

```python
 python3 process.py --model [model_name] --dataset [dataset_name]
```
