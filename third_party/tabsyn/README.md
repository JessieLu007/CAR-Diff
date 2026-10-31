# TabSyn（对比方法）

- 来源：官方仓库（Apache-2.0）。本目录不含其代码，只提供补丁 `tabsyn_changes.patch` 和组装脚本。
- 补丁内容：新版 torch 兼容
  （去掉 `ReduceLROnPlateau(verbose=...)`、`map_location`、无 GPU 时回退 CPU）和 `sample.py` 支持 `--num_samples`。
- 使用：
      bash third_party/tabsyn/setup_tabsyn.sh /path/to/tabsyn-main.zip
  生成 `third_party/tabsyn/tabsyn-main/`（已在 .gitignore 中）。
- 流程（在仓库根目录；需先用 run_cv 生成五折划分 splits/，再按下面顺序）：
      python -m experiments.prepare_tabsyn_folds --dataset <名称> --seed 2026 --n_folds 5
      cd third_party/tabsyn/tabsyn-main
      python run_tabsyn_all.py --dataset <名称> --n_folds 5
      python resample_tabsyn.py --dataset <名称> --fold 0 1 2 3 4 --n 4
  prepare_tabsyn_folds 把我们的划分拷入 tabsyn-main/data/<名称>_fold<k>/，并在 info.json 中给出 test_path，
  所以官方的 process_dataset.py 直接使用，不再自行切分。
- 评估：回到仓库根目录，`python -m evaluation.run ... --tabsyn-root third_party/tabsyn/tabsyn-main`