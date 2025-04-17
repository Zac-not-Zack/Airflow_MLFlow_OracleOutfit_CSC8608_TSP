# 🤖 Project DL Avancé  : Building a pipeline for UUCF part of Outfit Oracle

Contributor : WONG Hoe Ziet, ZHU Xingyu & Samir CHAFI RAHAMATTOULLA (TSP MAIA 2025 🐝)

# 🔡 Dataset
https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=transactions_train.csv 

# 🗺️ Directory of files

mini-projet/

├── airflow_dags/

│   └── pipeline_ml_dag.py

├── data/

├── src/

│   ├── data_ingestion.py

│   ├── preprocessing.py

│   ├── train_knn.py

│   └── validation.py

└── models/

# 🎯 Objective 
Create a pipeline for the UUCF part of our Oracle Outfit Application using Airflow & MLFlow

# 💻 How to launch the pipeline 

1) Install Airflow & MLFlow
```ruby
pip install mlflow apache-airflow
```
2) Launch MLFlow local web interface
```ruby
mlflow ui
```
3) Initiate the database of Airflow
```ruby
airflow db init
```
4) Create an user on Airflow
```ruby
airflow users create \
  --username admin \
  --firstname FIRST_NAME \
  --lastname LAST_NAME \
  --role Admin \
  --email admin@example.com
```
5) Change the following parameter according to the path of your files in ~/airflow/airflow.cfg
```ruby
dags_folder = _path_to_your_files_
load_examples = False
```
6) Launch Airflow
```ruby
airflow standalone
```
You should see your pipeline appears and you will be to click on play to launch it.

# :lady_beetle: Reporting Bugs

Teamwork is the key, please report at (https://github.com/Zac-not-Zack/MobileNetV1_DL_CSC8607/issues) if you come across any bug.


# :warning: Licence

[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/DAVFoundation/captain-n3m0/blob/master/LICENSE)

MIT License

Copyright (c) 2025 Wong Hoe Ziet 
