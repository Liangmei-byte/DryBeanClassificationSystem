# Dry Bean 多分类算法对比实验系统

本项目基于 Dry Bean 脏数据集完成多分类实验，工程化集成了数据加载、数据清洗、特征工程、模型训练、测试评估、推理速度评估、鲁棒性分析和结果可视化。算法运行阶段全部通过命令行完成，不使用 UI 界面。

## 1. 数据集说明

数据集中每条样本表示一粒干豆，输入特征为图像分割后得到的形态学特征，例如面积、周长、长轴长度、短轴长度、离心率、圆度、紧致度和形状因子等。分类目标为 `Class`，共 7 个类别：

`BARBUNYA`、`BOMBAY`、`CALI`、`DERMASON`、`HOROZ`、`SEKER`、`SIRA`

## 2. 数据处理方法

本项目的数据处理包括：

- 缺失值处理：处理 `Perimeter`、`Solidity` 等字段缺失；
- 异常缺失符号处理：将 `?` 转换为缺失值；
- 类型污染处理：将 `0.8252 cm` 等带单位字符串提取为数值；
- 标签污染处理：统一大小写、去除空格，将 `0→O`、`3→E`；
- 物理异常处理：将 `Area <= 0` 等不合理值置为缺失；
- 派生特征修复：重算 `AspectRation`、`roundness`、`Solidity`、`ShapeFactor1`、`ShapeFactor3` 等；
- 特征工程：新增对数特征、轴比例特征、凸包缺陷特征和尺度归一化特征；
- 标准化处理：为 Softmax 和 KNN 提供标准化输入。

## 3. 已实现算法

| 算法 | 实现方式 | 输入数据 | 是否绘制 Loss |
|---|---|---|---|
| LightGBM | `lightgbm.LGBMClassifier` | 未标准化特征 | 是 |
| XGBoost | `xgboost.XGBClassifier` | 未标准化特征 | 是 |
| Softmax | numpy 手写 Softmax 回归 | 标准化特征 | 是 |
| KNN | `sklearn.neighbors.KNeighborsClassifier` | 标准化特征 | 否 |

## 4. 实验结果汇总

| 模型 | 测试集精度 | 单样本推理时间 | 过拟合差距 | 简要结论 |
|---|---:|---:|---:|---|
| LightGBM | 0.921 | 16.3 us | 0.079 | 精度较高，鲁棒性强，但推理速度相对慢 |
| XGBoost | 0.922 | 4.9 us | 0.055 | 综合表现最均衡 |
| Softmax | 0.920 | 0.1 us | 0.004 | 推理最快，泛化稳定，但模型表达能力较弱 |
| KNN | 0.922 | 11.1 us | 0.078 | 精度较高，但对异常值和距离扰动较敏感 |
![此为测试集精度对比图](D:\code\Machine_Learning\DryBeanClassificationSystem\results\figures\test_accuracy_comparison.png"测试集精度对比图")
## 5. 项目结构

```text
DryBeanClassificationSystem/
├── main.py                         # 统一命令行入口
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                         # 放置三个原始脏数据 CSV
│   └── processed/                   # 保存预处理输出
├── scripts/
│   ├── preprocess_drybean.py         # 数据清洗与特征工程脚本
│   └── drybean_main_experiment.py    # 四算法训练、测试和分析脚本
├── src/drybean/
│   ├── data_loader.py
│   ├── models.py
│   ├── evaluate.py
│   ├── noise.py
│   └── plot_utils.py
├── results/
│   ├── summary_metrics.csv
│   └── figures/
└── docs/
    └── index.html                   # GitHub Pages 静态展示页面
```

## 6. 安装依赖

```bash
pip install -r requirements.txt
```

## 7. 运行方式

### 7.1 数据预处理

```bash
python main.py preprocess --input_dir data/raw --output_dir data/processed
```

### 7.2 训练与测试

```bash
python main.py experiment --data_dir data/processed --output_dir results
```

### 7.3 一键运行

```bash
python main.py all --input_dir data/raw --processed_dir data/processed --output_dir results
```

如果想先快速测试主流程，可以跳过鲁棒性实验：

```bash
python main.py all --input_dir data/raw --processed_dir data/processed --output_dir results --skip_robustness
```



