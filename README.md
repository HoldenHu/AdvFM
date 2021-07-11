
# AdvFM
Adversarial Deep Factorization Machine

#### Data format
##### **- ratings**
uid: 0

iid: 1

label: 1

ts: 1678980922

##### **- user_history**

uid: [iid]

##### - user_side

uid: 0

sparse_fea: [2]

dense_fea: 20

##### - item_side

uid: 0

sparse_fea: [2]

dense_fea: 20

#### Run

```python
 python3 process.py --model [model_name] --dataset [dataset_name]
```

