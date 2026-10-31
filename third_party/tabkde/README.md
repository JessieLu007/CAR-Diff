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
- `make_tabkde_info` 生成的 `data_path` 是相对仓库根的路径，TabKDE 的生成需在仓库根目录运行
  （`python -m evaluation.gen_baselines ... --methods tabkde`）。
- 五折复现还需生成列声明：`python -m evaluation.make_tabkde_info`（输出 data/Info/<数据集>_fold<k>.json）。