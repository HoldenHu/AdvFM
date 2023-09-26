# AdvFM (Adversarial Factorization Machine)

This is the PyTorch implementation for AdvFM proposed in the paper [Automatic Feature Fairness in Recommendation via Adversaries](), In Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, 2023. 
The AdvFM model can be found in the `model` folder.

Contributors: [Yiming Cao](https://github.com/caoymg), [Hengchang Hu](https://holdenhu.github.io/)


### Preprocess

- **First Step**. Download the raw data from the orginal source, e.g., ml_100k (https://grouplens.org/datasets/movielens/100k/)

- **Second Step**. Preprocess the raw data, making it in the following format.

*ratings.txt*

| user_id | item_id | label | timestamp  |
| ------- | ------- | ----- | ---------- |
| 0       | 1       | 1     | 1678980922 |

*user_history.npy*

user_id: [ item_id1,item_id2, ... ]

*user_side.csv*

| user_id | sparse_feature | dense_feature |
| ------- | -------------- | ------------- |
| 0       | [2]            | 15            |

*item_side.csv*

| item_id | sparse_feature | dense_feature |
| ------- | -------------- | ------------- |
| 0       | [2]            | 15            |


### How to Run

- **Third Step**. Use `process.py` to preprocess the raw data.
```python
 python3 process.py --model [model_name] --dataset [dataset_name]
```
