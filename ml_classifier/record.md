# 基本特征

训练集：

| 编号 | 1    | 2   | 3        | 4     | 5    | 6   | 7   | 8   | 9     |
| ---- | ---- | --- | -------- | ----- | ---- | --- | --- | --- | ----- |
| 名称 | 日期 | AQI | 质量等级 | PM2.5 | PM10 | SO2 | CO  | NO2 | O3_8h |
| max  | 12   | 438 | -        | 407   | 520  | 144 | 8.6 | 180 | 216   |
| min  | 1    | 36  | -        | 10    | 23   | 5   | 0.2 | 13  | 5     |

测试集：

| 编号 | 1    | 2   | 3        | 4     | 5    | 6   | 7    | 8   | 9     |
| ---- | ---- | --- | -------- | ----- | ---- | --- | ---- | --- | ----- |
| 名称 | 日期 | AQI | 质量等级 | PM2.5 | PM10 | SO2 | CO   | NO2 | O3_8h |
| max  | 12   | 500 | -        | 621   | 866  | 153 | 10.4 | 183 | 278   |
| min  | 1    | 37  | -        | 0     | 0    | 6   | 0.3  | 13  | 7     |

## MLP

- 模型结构测试

  ```python
  'mlp':{
          'alpha':[0.01],
          'hidden_layer_sizes':[(10,10)],
          'solver':['lbfgs'],
          'activation':['identity'],
          'learning_rate':['constant']
      }
  ```
- 编码 `日期`和 `质量等级`：0.5849

  ```python

  train_df['日期'] = pd.to_datetime(train_df['日期'], format='%Y/%m/%d')
  train_df["month"]= train_df["日期"].apply(lambda x : x.month)

  test_df['日期'] = pd.to_datetime(test_df['日期'], format='%Y/%m/%d')
  test_df["month"]= test_df["日期"].apply(lambda x : x.month)
  if encoder_list is not None:
      label_coder = LabelEncoder()
      for col in encoder_list:
          train_df[col] = label_coder.fit_transform(train_df[col])
          test_df[col] = label_coder.transform(test_df[col])
  ```
- 移除 `日期`和 `质量等级`：0.0591
- 移除 `日期`和 `质量等级`，并对特征做归一化：0.0581
- 编码 `日期` (年月日)和 `质量等级` （one hot），并对特征和目标值做归一化: 0.06176

  ```python
  def date_encoder(df,key='日期'):
      df["year"] = pd.to_datetime(df[key]).dt.year
      df["month"] = pd.to_datetime(df[key]).dt.month
      df["day"] = pd.to_datetime(df[key]).dt.day
      del df[key]
      return df

  def onehot_encoder(df,key='质量等级',label_list=['重度污染', '良', '中度污染', '轻度污染', '严重污染']):
      ff = pd.get_dummies(df[key].values)
      for label in label_list:
          df[label] = ff[label]
      del df[key]
      return df


  def scaler_normalize(train_df,test_df,scale_list=None,label=None):

      target = train_df[label]
      del train_df[label]
      data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
      data = data.fillna(0)
      data = date_encoder(data)
      data = onehot_encoder(data)

      scaler = MinMaxScaler(feature_range=(0, 1))

      for col in scale_list:
          data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

      train_df = data[:train_df.shape[0]]
      train_df[label] = target
      test_df = data[test_df.shape[0]:]
      # print(train_df)
      # print(test_df)
      return train_df,test_df
  ```

# Linear Regressor

- 移除 `日期`和 `质量等级`，并对特征做归一化：0.05912
- 编码 `日期` (月日)和 `质量等级` （one hot)：**0.04362**
- 移除 `日期` (月日)和 编码 `质量等级` （one hot)，并对特征做归一化：**0.04362**
- 编码 `日期` (月日)和 `质量等级` （one hot)，并对特征做归一化：**0.04362**
