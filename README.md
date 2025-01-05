# MR-RCA
Artifacts accompanying the paper MR-RCA

# Data
Our source data is available at https://www.aiops.cn/%e5%a4%9a%e6%a8%a1%e6%80%81%e6%95%b0%e6%8d%ae/

we also release our pre-processed data in AIops2022 folders, respectively.

# Dependencies
```
pip install -r requirements.txt
```



# Run

```
  python main.py --entity pod    --beta 0.7 --alpha 0.6 --k 3 --lamb 60
  python main.py --entity node    --beta 0.7 --alpha 0.6 --k 3 --lamb 60
  python main.py --entity service    --beta 0.7 --alpha 0.6 --k 3 --lamb 60
```

# Architecture
![image](https://github.com/user-attachments/assets/9b031759-f6ff-4c9c-bf48-a2e9332d0c8b)




# Contact us
Any questions can leave messages in "Issues"!
