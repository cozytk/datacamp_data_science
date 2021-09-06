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
```