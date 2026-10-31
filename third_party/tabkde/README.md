# TabKDE（对比方法）

- 来源：官方仓库 https://github.com/tabkde/tabkde-main 。该仓库未声明许可证，因此本仓库**不包含**其代码，
  只提供对官方代码的修改补丁 `tabkde_changes.patch` 和组装脚本 `setup_tabkde.sh`。
  TabKDE 在 TabSyn（Apache-2.0）代码库上扩展，其中的 `src/` 与官方 TabSyn 一致。
- 补丁内容：路径改为相对包位置、`latent_utils` 改用相对导入；`sample.py` 直接返回 DataFrame 并支持
  `num_samples`；`process_dataset.py` 改为 `process_data(name, train_df, test_df)`，直接读取已划分好的数据
  （标签列须在 `cat_col_idx` 中）。其余仅为 print/注释。
- 使用：从官方仓库下载 zip（或自行获取目录），在仓库根目录运行
      bash third_party/tabkde/setup_tabkde.sh /path/to/tabkde-main.zip
  生成根目录下的 `tabkde/`（已在 .gitignore 中，不提交）。
- 五折复现还需生成列声明：`python -m evaluation.make_tabkde_info`（输出 data/Info/<数据集>_fold<k>.json）。


## 使用

### 1. 数据与列声明

每个数据集一个 `data/<名称>.csv`，并在 `data/datasets_info.json` 中登记：

```json
{
  "chronic_kidney_disease": {
    "numerical":   ["age", "bp", "..."],
    "categorical": ["al", "su", "..."],
    "target": "class",
    "minority_class": "ckd"
  }
}
```

`target` 会自动并入类别列并与其他列联合建模。新增自定义数据集只需增加这一条目（用于第 3 节）。
若要对新数据集跑第 2 节的五折流程，还需在 `data/datasets_idx.json` 与 `utils/common.py` 的 `DATASETS` 中登记。

### 2. 五折复现论文实验

```bash
# 训练（Ours：自回归类别头，cat_ar 默认 True）。固定预算 200 epochs，无验证集，取最终权重
python -m experiments.run_cv --dataset chronic_kidney_disease --fold 0 1 2 3 4 \
    --tag main --model "cat_ar=True" --train "target_epochs=200" --seed 42
# 消融：w/o AR（类别列在给定数值表示下条件独立采样）
python -m experiments.run_cv --dataset chronic_kidney_disease --tag noar --model "cat_ar=False"
# 其余消融：--model "share_decoder=False"；legacy / sampled / nointeract 见下方 tag

# 每折重复采样（共 5 次：s0 来自训练时，s1..s4 由此生成）
python -m experiments.resample_ours --dataset chronic_kidney_disease --device cuda --js 1 2 3 4

# 对比方法（--dataset / --fold 只接受单个值，多折请循环；每折采样 5 次）
for k in 0 1 2 3 4; do
  python -m evaluation.gen_baselines --dataset chronic_kidney_disease --fold $k \
      --methods smote ctgan tvae tabddpm tabkde --n-samples 5
done
# diabetic_data 的 tabddpm 需加 --ddpm-lr 1e-4

# 评估（--hp = h256_t200_3_<tag>）
python -m evaluation.run --dataset chronic_kidney_disease --fold 0 1 2 3 4 --hp h256_t200_3_main \
    --baselines smote ctgan tvae tabddpm tabkde
# 只重算部分指标：--only shape cpt ；全部重算：--redo ；TabSyn 结果：--tabsyn-root <目录>

# 汇总（无命令行参数；在 scripts/aggregate_main.py 顶部设置 OURS_MAIN 等，输出到 agg_out/）
python scripts/aggregate_main.py
```

结果与 tag 沿用实验时的命名；汇总出表时将 `ours_h256_t200_3_<主方法 tag>` 记为 Ours，
`ours_h256_t200_3_<noar tag>` 记为 w/o AR。
论文中的 legacy / sampled / nointeract 消融使用旧代码，`git checkout backup-cat-switches` 复现。
若重采样旧版（w/o AR）结果，请使用与其一致的代码版本（tag `experiments-final`）。

### 3. 不使用五折：quickstart.py

自己划分训练/测试集即可。测试集可选，仅用于评估。

```bash
# 方式 A：登记到 data/datasets_info.json 后按名称使用（data/<名称>.csv 作为训练集）
python quickstart.py run --dataset hepatitis --test my_test.csv --repeats 5
# 方式 B：自定义文件，spec.json 格式同 datasets_info.json 的一个条目
python quickstart.py run --train train.csv --test test.csv --spec spec.json --out runs/demo

# 也可分步：fit → sample → eval
python quickstart.py fit    --dataset hepatitis --test my_test.csv
python quickstart.py sample --dataset hepatitis --repeats 5
python quickstart.py eval   --dataset hepatitis [--fake runs/hepatitis/synthetic_s1.csv]

# 对比方法（smote ctgan tvae tabddpm tabkde，输出到 runs/<名称>/baselines/<方法>_s<j>.csv）
python quickstart.py baselines --dataset hepatitis --methods smote ctgan --n-samples 5
python quickstart.py eval --dataset hepatitis --fake runs/hepatitis/baselines/smote_s0.csv
```

常用覆盖项：`--epochs`、`--seed`、`--model "dim_hidden=128"`、`--training "batch_size=32"`。
训练集样本数需不小于 batch_size（默认 64）。
若出现 `libgomp: Invalid value for environment variable OMP_NUM_THREADS`，quickstart 会自动修正；
其他脚本请先 `export OMP_NUM_THREADS=8`。

### 4. TabKDE 与 TabSyn

- **TabKDE**：见 `third_party/tabkde/README.md`。先 `bash third_party/tabkde/setup_tabkde.sh <官方 zip>`，
  五折流程再 `python -m evaluation.make_tabkde_info`；quickstart 的 baselines 会自动生成所需的列声明。
- **TabSyn**：不随仓库提供，请使用官方仓库 https://github.com/amazon-science/tabsyn 并采用其默认参数。
  `python -m experiments.prepare_tabsyn_folds --dataset <名称> [--all] --seed 2026 --n_folds 5`
  将本仓库的五折划分转成 TabSyn 的数据格式；TabSyn 生成结果通过 `evaluation.run --tabsyn-root` 评估。