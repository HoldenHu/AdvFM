
# AdvFM
Adversarial Deep Factorization Machine

#### Data format
##### *ratings.txt*

| user_id | item_id | label | timestamp  |
| ------- | ------- | ----- | ---------- |
| 0       | 1       | 1     | 1678980922 |

##### *user_history.npy*

user_id: [ item_id1,item_id2, ... ]

##### *user_side.csv*

| user_id | sparse_feature | dense_feature |
| ------- | -------------- | ------------- |
| 0       | [2]            | 15            |

##### *item_side.csv*

| item_id | sparse_feature | dense_feature |
| ------- | -------------- | ------------- |
| 0       | [2]            | 15            |

#### Run

```python
 python3 process.py --model [model_name] --dataset [dataset_name]
```

