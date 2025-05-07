import numpy as np
import pandas as pd


fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")
fires['burned_area'] = np.log(fires['burned_area'] + 1)

# 1-2: 데이터 개요 및 카테고리형 특성 분포 확인

# (1) 상위 5개 행 출력
print("=== fires.head() ===")
print(fires.head())

# (2) 데이터 정보 출력
print("\n=== fires.info() ===")
fires.info()

# (3) 수치형 변수 통계 요약 출력
print("\n=== fires.describe() ===")
print(fires.describe())

# (4) month, day 카테고리 분포 출력
print("\n=== month value_counts ===")
print(fires['month'].value_counts())

print("\n=== day value_counts ===")
print(fires['day'].value_counts())

# --- 1-3: 데이터 시각화 ---

import matplotlib.pyplot as plt


numeric_cols = ['avg_temp', 'max_temp', 'avg_wind', 'max_wind_speed', 'burned_area']
n_cols = 3
n_rows = int(np.ceil(len(numeric_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4), constrained_layout=True)
for ax, col in zip(axes.flatten(), numeric_cols):
    ax.hist(fires[col], bins=30, edgecolor='k')
    ax.set_title(f"{col} distribution")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

for ax in axes.flatten()[len(numeric_cols):]:
    ax.set_visible(False)
plt.show()

# --- 1-4: burned_area 왜곡 현상 개선을 위한 히스토그램 비교 ---

df = pd.read_csv('./sanbul2district-divby100.csv', sep=',')
raw_burned = df['burned_area']

log_burned = np.log(raw_burned + 1)

fig, axes = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)

axes[0].hist(raw_burned, bins=30, edgecolor='k')
axes[0].set_title('burned_area')
axes[0].set_xlabel('burned_area')
axes[0].set_ylabel('Frequency')

axes[1].hist(log_burned, bins=30, edgecolor='k')
axes[1].set_title('ln(burned_area + 1)')
axes[1].set_xlabel('ln(burned_area + 1)')
axes[1].set_ylabel('Frequency')

plt.show()

# --- 1-5: Train/Test 분리 및 비율 확인 ---

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)

test_set.head()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

print("\nMonth category proportion: \n",
    strat_test_set["month"].value_counts()/len(strat_test_set))

print("\nOverall month category proportion: \n",
    fires["month"].value_counts()/len(fires))

# --- 1-6: scatter_matrix()를 이용한 특성 매트릭스 시각화 ---

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

attributes = ['burned_area', 'max_temp', 'avg_temp', 'max_wind_speed']

scatter_matrix(
    fires[attributes],
    alpha=0.5,
    figsize=(10, 10),
    diagonal='hist'
)
plt.suptitle('Scatter Matrix of Selected Features')

plt.show()

# --- 1-7: 지역별로 ‘burned_area’에 대해 plot 하기: 원의 반경은 max_temp(옵션 s), 컬러는 burned_area(옵션 c)를 의미---

import matplotlib.pyplot as plt
plt.close('all')

fires.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=fires["max_temp"], label="max_temp",
            c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)

plt.title("Geographical Scatter Plot of Burned Area")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

# --- 1-8: 카테고리형 특성 month, day에 대해 OneHotEncoder()를 이용한 인코딩/출력

from sklearn.preprocessing import OneHotEncoder

fires = strat_train_set.drop(["burned_area"], axis=1)
fires_labels = strat_train_set["burned_area"].copy()

fires_cat = fires[["month", "day"]]

# sparse 매개변수를 제거하고, 결과를 toarray()로 밀집 배열로 변환
cat_encoder = OneHotEncoder()  
fires_cat_1hot = cat_encoder.fit_transform(fires_cat).toarray()

print("month categories:", cat_encoder.categories_[0])
print("day categories:  ", cat_encoder.categories_[1])

# 첫 5개 인코딩 결과 확인
print("\nOne-hot encoding (first 5 rows):\n", fires_cat_1hot[:5])

# --- 1-9: Scikit-Learn의 Pipeline, StandardScaler를 이용하여 카테고리형 특성을 인코딩한 training set 생성하기

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

fires = strat_train_set.drop("burned_area", axis=1)
fires_labels = strat_train_set["burned_area"].copy()

num_attribs = ["longitude", "latitude", "avg_temp", "max_temp", "avg_wind", "max_wind_speed"]
cat_attribs = ["month", "day"]

print("\n############################################")
print("Now let's build a pipeline for preprocessing the numerical attributes:")

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

fires_prepared = full_pipeline.fit_transform(fires)

print("\nPrepared training set shape:", fires_prepared.shape)