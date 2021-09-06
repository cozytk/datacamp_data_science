# Introduction & Intermediate Python

`np.array.shape`

- 몇행 몇열인지 출력
- ex) `(2, 5)`

`np.array.corrcoef()`

- 피어슨 상관계수 계산

`plt.plot(x,y)`

- 선으로 표현

`plt.scatter(x,y)`

- 점으로 표현

 `plt.show()`

- 그린거 보여줌

그 랜덤으로 표준편차랑 평균주면 만들어주는거해보자

 hystogram

`plt.hist()`

- (obj, bins)

`plt.clf()`

- clear

plt.scatter()의 s parameter.

```python
# Scatter plot
# x축, y축, size. color, alpha(투명도)
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)

# Previous customizations
 
plt.xscale('log') # x 값 폭이 너무 넓어서 log 처리
plt.xlabel('GDP per Capita [in USD]') # x축의 라벨
plt.ylabel('Life Expectancy [in years]') # y축의 라벨
plt.title('World Development in 2007') # plot의 타이틀
plt.xticks([1000,10000,100000], ['1k','10k','100k']) # x축값을 변환 (1000 -> 1k)

# Additional customizations, text 기입
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call
plt.grid(True)

# Show the plot
plt.show()
```

`pd.DataFrame()`

- 데이터프레임 생성

`pd.DataFrame.index`

- 데이터프레임의 인덱스를 나타냄
- 위에 실제 행 만큼의  리스트를 할당하는 식으로 각각의 인덱스를 설정

`pd.read_csv()`

ex)

```python
# 매개변수는 문자열 형태
cars = pd.read_csv('cars.csv')
# index_col = cloumn 넘버
cars = pd.read_csv('cars.csv', index_col=0)
```

## Dictionary

`dict.keys()`

- dict의 키들 나열

## Pandas

**DataFrame.loc**

label을 기준

**DataFrame.iloc**

index를 기준 

**괄호**

- 괄호의 개수로 공식적으로 외우기 보다, 1x1이면 int64, 1xn or nx1 이면 Series, nxn이면 DataFrame이라는 것을 이해
- loc이 행과 열로 한개의 값을 특정하면 `numpy.int64`  타입으로 나옴

```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# 결과로 Series가 나옴
print(cars.loc['JPN'])

# 결과로 DataFrame이 나옴
print(cars.loc[['JPN']])

# 결과로 DataFrame이 나옴
# rows가 2개, columns가 n개인 2차원 배열이므로 Series로 출력 불가능
print(cars.loc[['AUS', 'EG']])

--------------------------------------------------------
In [1]:
cars
Out[1]:

     cars_per_cap        country  drives_right
US            809  United States          True
AUS           731      Australia         False
JPN           588          Japan         False
IN             18          India         False
RU            200         Russia          True
MOR            70        Morocco          True
EG             45          Egypt          True

In [2]:
cars.loc['IN','cars_per_cap']
Out[2]:
18
In [3]:
cars.loc[['IN','cars_per_cap']]
Out[3]:

              cars_per_cap country drives_right
IN                    18.0   India        False
cars_per_cap           NaN     NaN          NaN
In [4]:
cars.loc[['IN'],['cars_per_cap']]
Out[4]:

    cars_per_cap
IN            18
In [5]:
type(cars.loc[['IN'],['cars_per_cap']])
Out[5]:
pandas.core.frame.DataFrame
In [6]:
type(cars.loc['IN','cars_per_cap'])
Out[6]:
numpy.int64
```

**슬라이싱**

슬라이싱은 하는 것만으로 리스트를 반환 (0:1 처럼 값이 하나여도 무관)

이러면 loc을 쓰는것과 그냥 괄호로 참조하는 게 어떤 차이지?

## Comparision Operation

**Numpy with Comparision**

단일 요소와 비교를 하면 모든 요소 각각의 bool 값을 담은 List를 반환

ex) `['True', 'False', 'False', 'True', ...']`

**NumPy with logical opeartors**

- `logical_and()`
- `logical_or()`
- `logical_not()`

### 일반 괄호, loc, iloc

**세팅**

```python
In [1]: import pandas as pd
In [2]: cars = pd.read_csv('cars.csv', index_col = 0)
In [3]: cars
Out[3]:
     cars_per_cap        country  drives_right
US            809  United States          True
AUS           731      Australia         False
JPN           588          Japan         False
IN             18          India         False
RU            200         Russia          True
MOR            70        Morocco          True
EG             45          Egypt          True
```

**일반괄호, 행으로 접근**

```python
In [4]:
cars['US']
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    cars['US']
  File "<stdin>", line 2685, in __getitem__
    return self._getitem_column(key)
  File "<stdin>", line 2692, in _getitem_column
    return self._get_item_cache(key)
  File "<stdin>", line 2486, in _get_item_cache
    values = self._data.get(item)
  File "<stdin>", line 4115, in get
    loc = self.items.get_loc(item)
  File "<stdin>", line 3065, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "<stdin>", line 140, in pandas._libs.index.IndexEngine.get_loc
  File "<stdin>", line 162, in pandas._libs.index.IndexEngine.get_loc
  File "<stdin>", line 1492, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "<stdin>", line 1500, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'US'
```

- 불가능
- Dataframe 을 list 에 할당하면 열만 저장됨

    ```python
    In [5]:
    list_cars = list(cars)
    In [6]:
    list_cars
    Out[6]:
    ['cars_per_cap', 'country', 'drives_right']
    ```

**한 열을 series로 출력**

```python
cars['cars_per_cap']
cars.loc[:,'cars_per_cap']
cars.iloc[:, 0]

US     809
AUS    731
JPN    588
IN      18
RU     200
MOR     70
EG      45
Name: cars_per_cap, dtype: int64
```

**series to list**

```python
In [13]: cpc_list = list(cars.iloc[:,0])
In [14]: cpc_list
Out[14]: [809, 731, 588, 18, 200, 70, 45]
```

- 시리즈에서 라벨 다 떼고 저런 식으로 나옴

**한 열을 DataFrame으로 출력**

```python
cars[['cars_per_cap']]
cars.loc[:,['cars_per_cap']]
cars.iloc[:,[0]]

cars_per_cap
US            809
AUS           731
JPN           588
IN             18
RU            200
MOR            70
EG             45
```

**두 열을 DataFrame으로 출력**

- loc과 iloc 없이 불가능

**결론**

한 열이나 요소는 괄호로, 한개 이상의 행 또는 두개 이상 열을 다룰때는 loc과 iloc

### Apply

```python
_df.apply(attribute)
_df.apply(type.method())
_df.apply(type.method(), axis=bool)
```

### Random

`np.random.rand()`

- 0부터 1의 랜덤 설정

`np.random.seed(num)`

- 시드 넘버 설정
- pseudo 랜덤
- 같은 시드값에 대해서는 같은 랜덤값

 `np.random.randint(strat, end)`

- start, end : 범위 값 (end 포함 안함)
- int 단위

**random step & random walk**

※ **np_array 행열 바꾸기**

- `np_aw.transpose()`

# For

**enumermate**

```python
for index, x in enumerate(src): # list str etc
	print(index, x)
```

**Dictionary**

```python
for key, val in dict.items():
	...
```

**Numpy array**

```python
for val in np.nditer(np_array) :
	...
```

### DataFrame

**Basic**

```python
for val in brics :
	print(val)
```

- 그냥 기본 for문으로 요소를 접근하면, **column(열)**이 접근

**Iterrows**

```python
for label, row in _df.iterrows():
	print(lab)
	print(row)
```

- 행 기반 요소 접근
- 앞에 변수 라벨, 뒤에 변수 Series(행)

**apply**

```python
_df['new_feature'] = _df['feature'].apply(len)
```

- DataFrame 에서 for 문 보다 간단

 

### zip, 리스트 두개를 딕셔너리로

**zip**

- `new_dict = dict(zip(list1,list2))`

```python
years = list(range(2011,2021))
durations = [103, 101, 99, 100, 100, 95, 95, 96, 93, 90]
movie_dict = dict(zip(years,durations))

movie_dict
-------------------------------------------------------
{2011: 103,
 2012: 101,
 2013: 99,
 2014: 100,
 2015: 100,
 2016: 95,
 2017: 95,
 2018: 96,
 2019: 93,
 2020: 90}
```# Data Manipulation with pandas

# Transforming DataFrames

## Pandas 개요

- pandas is built on NumPy and Matplotlib

### `_df.head()`

- 상위 몇개 행을 보여줌
- 행이 많을 때 유용

### `_df.info()`

- 열 이름, 열에 포함 된 데이터 유형 및 누락된 값이 있는지 여부 확인

### `_df.shape`

- 행 수와 열수를 포함하는 튜플
- no method, attribute.

### `_df.describe()`

- 수치형 데이터 열에대한 통계적 요약 제공.
- count, mean, std, min, 25% 50%, 75%, max

### `_df.values`

- value들을 2차 numpy 배열로 반환

### `_df.coulmns`

- column 들의 이름을 가지고 있음

### `_df.index`

- row의 숫자들 혹은 이름들을 가지고 있음

## Sorting

### `_df.sort_values()`

- change the order of rows
- 특정 열에 대하여 오름차순으로 정렬됨
- 매개변수로 리스트도 올 수 있음 (리스트 앞에 거 부터 정렬, 앞에가 같으면 뒤에 걸로 또 정렬
- `...(..., ascending=str or list)`
    - 오름차순 내림차순 여부 정렬 가능

## Subsetting

### `_df[column].isin(['s1','s2'])`

- 특정 열에 있는 특정 요소들

## Aggregating DataFrames

## Summary statistics

### Summarizing numeric data

- mean, median, mode, min, max, var, std, sum, quantile

### `_df['coulmn'].agg(function)`

- 컬럼에 대하여 매개변수로 받는 함수를 실행시켜줌
- 여러 컬럼, 여러 함수 가능

### `_df['column'].cumsum()`

- cumulative sum
- 1, 1+2, 1+2+3 ... 이런식으로 각 행을 누적하여 더하려 남김
- cummax, cummin, cumprod

## Counting

### `_df.drop_duplicates(subset='something')`

- Dropping duplicate name
- something이 중복이면 하나빼고 삭제
- subset 에 list를 받아 두개로 판단

### `_df['colmn'].value_counts(sort=bool, normalize=bool)`

- 해당 column에 value들이 몇개 있는지 세어줌
- sort 사용 가능(True일시 desc)
- normalize 하면 비율화됨

## Grouped summary statistics

### `_df.groupby('col1')['col2'].method()`

- Group summaries
- agg활용 가능
- col1을 그룹별로 지정해서, 'col2'값을 활용
- co1, col2 복수 가능

## Pivot table

### `_df.pivot_table(values='col1', index='col2', columns='col3' aggfunc=method, fill_value=num(0), margins=bool)`

- groupy의 다른 적용
- default는 mean()함수
- 다 복수가능합니다 물론
- fill_value : NaN 채워줌
- margins : 총계 넣어줌 (총합, 총평균 등)
    - fill_values로 넣어준 0 은 평균에 안들어감

---

# Slicing and Indexing DataFrames

## Explicit indexes

### Setting a column as the index

### `_new_df = _df.set_index("index_name")`

- 특정 칼럼을 인덱스로 설정
- 매개변수로 **리스트 [받을](https://www.notion.so/11af4df33d0143919a86d6fba68fd5fb) 수 있음**
    - `_df_idx.loc[[("a1", "b1"), ("a2", "b2")]]`
    - 복수개의 매개변수는 위와 같이 접근
    - `sort_values()` 처럼 `sort_index()`가능
        - `(level= , ascending=)`

### `_df.reset_index(drop=bool[_d=false])`

- 인덱스 삭제
- drop이 False(디폴트)면 기존 인덱스는 하나의 열이 되고 새로 0~n의 numeric index 생성
- drop이 True 면 set_index로 지정해주었던 것 삭제

### Indexes make subsetting simpler

### `_df[_df['col'].isin(['val1','val2'])]`

```jsx
          *co1* co2 co3 co4 co5
index_a    va1   .   .   .   .
index_b    va2   .   .   .   . 
```

### `_df_idx.loc[["val1","val2"]]`

- index가 사전에 지정이 되어있다면 다음과 같이 변수를 접근할 때 편리함

### Now you have two problems

- Index values are just data
- Indexes violate 'tidy data' principles
- You need to learn two syntaxes
- 인덱스 쓰면 문제가 있음, 데이터 안에서의 통일성을 잃게 됨 안쓰는거 추천하지만 다른 사람 코드를 이해하기 위해 공부해놓읍시다

## Slicing and subsetting with .loc and .iloc

### slicing the outer index level

### `_df.loc["rowA":"rowB"]`

- numeric index, position 만이 아닌 loc을 사용해 행의 이름으로도 슬라이싱 가능
- 맨 마지막이 포함된다는 차이점 ↔ iloc
- 판다스는 오류를 출력하지 않는 것을 항상 조심해야함

## Working with pivot tables

### The axis argument

### `df.mean(axis='index')`

- 축 인수
- index가 디폴트
- columns로 하면 축을 열로 설정

### `dataframe["column"].dt.component`

- component는 month, year 등이 올 수 있음
- 날짜형 포맷을 component에 맞게 반환해줌

# Creating and Visualizing DataFrames

## Visualizing your data

### `_plt.plot(kind='bar^line^scatter', rot=degree)`

- kind : bar, line, scatter
- rot: x축 레이블을 회전시킴
- 플롯은 서로 겹쳐 놓을 수 있음

### `_plt.legend(["a", "b"])`

- 앞에 플롯과 뒤에 플롯이 무엇인지 각각 이름을 붙혀줌

## Missing values

### Detecting missing values

### `_df.isna()`

- 모든 단일값에 대하여 NaN인지 아닌지 True^False
- .any()라는 메소드가 붙으면, 각 열에 대해 bool이 있는지 없는지 얻음
- .sum() 메소드를 사용하면, 각 열의 NaN의 개수 구할 수 있음

### `_df.dropna()`

- 누락된 값이 포함된 DF 행을 제거

### `_df.fillna(num)`

- 누적된 값을 num으로 바꿈

### `_df[list].hist()`

- list에 개수만큼 histogram을 만들어줌

## Creating DataFrames

### 1. From a list of dictionaries

- Constructed row by row
- Each case of Lists

### 2.From a dictionary of lists

- Constructed column by column
- Key = column name
- Value = list of column values

## Reading and writing CSVs

물론 항목 별 입력이 일반적으로 데이터를 DataFrame으로 가져 오는 가장 효율적인 방법 아님

### `_df.to_csv('filename.csv')`

- DataFrame to CSV

## Wrap-up

### More to learn

- Joining Data with pandas
- Streamlined Data Ingestion with pandas