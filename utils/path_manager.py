"""
统一管理对比模型（baselines）和自己模型（results）的路径与数据读取
支持：
- baselines: {model}_gen/seed_{i}/full_data 或 minority (内含 comparison/augment 文件)
- self model: results/{dataset}/{run_dir}/seed_{i}/synthetic_data.csv
- run_dir 命名: {mode}_{goal}_h{hidden_dim}_t{num_timesteps}_{num_heads}_{timestamp}
- 提供访问模型文件、配置文件的接口
"""

import os
import re
import glob
import pandas as pd
from typing import List, Dict, Optional, Union, Any

import utils.common as common

# ================== 基础常量 ==================
# Baselines 模型列表
BASELINES_MODELS = common.BASELINE_MODELS
GEN_SUFFIX = '_gen'
SUBDIR_FULL = 'full_data'
SUBDIR_MINORITY = 'minority'

# minority 目录下的文件类型关键词
MINORITY_COMPARISON_KEY = 'comparison'
MINORITY_AUGMENT_KEY = 'augment'

# 种子列表（可根据需要修改）
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# ================== 自己模型相关常量 ==================
RESULTS_ROOT = './results'          # 自己模型根目录

# 运行目录名正则解析
# 例如: full_data_fidelity_h128_t200_3_0606_204121
#      minority_cond_fidelity_h128_t200_3_0522_225531
RUN_DIR_PATTERN = re.compile(
    r'^(?P<mode>full_data|minority_cond|minority_uncond)_'
    r'(?P<goal>fidelity|cfg_boost)_'
    r'h(?P<hidden_dim>\d+)_t(?P<num_timesteps>\d+)_'
    r'(?P<num_heads>\d+)_(?P<timestamp>\d{4}_\d{6})$'
)

# 生成数据文件名（每个 seed 目录下固定名称）
SYNTHETIC_DATA_FILENAME = 'synthetic_data.csv'

# ================== 辅助函数 ==================
def ensure_path_exists(path: str) -> str:
    """确保路径存在，若不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# ================== Baselines 管理器 ==================
class BaselinePathManager:
    """
    对比模型生成数据路径管理器
    目录结构：baselines/{model}_gen/seed_{i}/{full_data|minority}/
    """

    def __init__(self, root_dir: str = './baselines'):
        self.root_dir = os.path.abspath(root_dir)

    def get_model_dir(self, model_name: str) -> str:
        """获取某个模型的根目录（例如 baselines/ctgan_gen）"""
        if model_name not in BASELINES_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {BASELINES_MODELS}")
        folder = f"{model_name}{GEN_SUFFIX}"
        return os.path.join(self.root_dir, folder)

    def get_seed_dir(self, model_name: str, seed: int) -> str:
        """获取某个模型下指定种子的根目录（例如 baselines/ctgan_gen/seed_0）"""
        return os.path.join(self.get_model_dir(model_name), f"seed_{seed}")

    def get_subdir_path(self, model_name: str, seed: int, subdir: str) -> str:
        """
        获取模型、种子下的子目录路径
        subdir: 'full_data' 或 'minority'
        """
        if subdir not in (SUBDIR_FULL, SUBDIR_MINORITY):
            raise ValueError(f"subdir must be '{SUBDIR_FULL}' or '{SUBDIR_MINORITY}'")
        return os.path.join(self.get_seed_dir(model_name, seed), subdir)

    def list_available_seeds(self, model_name: str) -> List[int]:
        """列出某个模型下实际存在的种子编号（基于文件夹存在性）"""
        model_dir = self.get_model_dir(model_name)
        if not os.path.exists(model_dir):
            return []
        seeds = []
        for item in os.listdir(model_dir):
            if item.startswith('seed_') and os.path.isdir(os.path.join(model_dir, item)):
                try:
                    seed = int(item.split('_')[1])
                    seeds.append(seed)
                except ValueError:
                    pass
        return sorted(seeds)

    def read_data(self, model_name: str, seed: int, subdir: str,
                  minority_type: Optional[str] = None, **read_kwargs) -> pd.DataFrame:
        """
        读取指定模型、种子、子目录下的数据
        - subdir='full_data': 读取目录下所有 CSV 文件，若只有一个则直接返回，否则合并
        - subdir='minority': 需要指定 minority_type = 'comparison' 或 'augment'
        """
        dir_path = self.get_subdir_path(model_name, seed, subdir)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if subdir == SUBDIR_FULL:
            files = glob.glob(os.path.join(dir_path, "*.csv"))
            if not files:
                raise FileNotFoundError(f"No CSV files in {dir_path}")
            if len(files) == 1:
                return pd.read_csv(files[0], **read_kwargs)
            else:
                dfs = [pd.read_csv(f, **read_kwargs) for f in files]
                return pd.concat(dfs, ignore_index=True)

        else:  # minority
            if minority_type not in (MINORITY_COMPARISON_KEY, MINORITY_AUGMENT_KEY):
                raise ValueError(f"minority_type must be '{MINORITY_COMPARISON_KEY}' or '{MINORITY_AUGMENT_KEY}'")
            pattern = f"*{minority_type}*.csv"
            files = glob.glob(os.path.join(dir_path, pattern))
            if not files:
                raise FileNotFoundError(f"No {minority_type} file in {dir_path}")
            if len(files) == 1:
                return pd.read_csv(files[0], **read_kwargs)
            else:
                dfs = [pd.read_csv(f, **read_kwargs) for f in files]
                return pd.concat(dfs, ignore_index=True)

    # 便捷方法（特定种子）
    def read_full_data(self, model_name: str, seed: int, **kwargs) -> pd.DataFrame:
        return self.read_data(model_name, seed, SUBDIR_FULL, **kwargs)

    def read_minority_comparison(self, model_name: str, seed: int, **kwargs) -> pd.DataFrame:
        return self.read_data(model_name, seed, SUBDIR_MINORITY,
                              minority_type=MINORITY_COMPARISON_KEY, **kwargs)

    def read_minority_augment(self, model_name: str, seed: int, **kwargs) -> pd.DataFrame:
        return self.read_data(model_name, seed, SUBDIR_MINORITY,
                              minority_type=MINORITY_AUGMENT_KEY, **kwargs)

    # 批量读取所有种子（可选合并）
    def read_all_seeds(self, model_name: str, subdir: str,
                       minority_type: Optional[str] = None,
                       **kwargs) -> Union[Dict[int, pd.DataFrame], pd.DataFrame]:
        """
        读取某个模型所有种子的数据, 返回 {seed: df}
        """
        seeds = self.list_available_seeds(model_name)
        if not seeds:
            raise FileNotFoundError(f"No seeds found for model {model_name}")
        result = {}
        for seed in seeds:
            try:
                df = self.read_data(model_name, seed, subdir, minority_type, **kwargs)
                result[seed] = df
            except FileNotFoundError as e:
                print(f"Warning: seed {seed} missing: {e}")
        
        return result


    def read_all_seeds_full_data(self, model_name: str, **kwargs):
        return self.read_all_seeds(model_name, SUBDIR_FULL, **kwargs)

    def read_all_seeds_minority_comparison(self, model_name: str, **kwargs):
        return self.read_all_seeds(model_name, SUBDIR_MINORITY,
                                   minority_type=MINORITY_COMPARISON_KEY,
                                   **kwargs)

    def read_all_seeds_minority_augment(self, model_name: str, **kwargs):
        return self.read_all_seeds(model_name, SUBDIR_MINORITY,
                                   minority_type=MINORITY_AUGMENT_KEY,
                                   **kwargs)


# ================== 自己模型结果管理器 ==================
class ResultModelManager:
    """
    自己训练的生成模型结果管理器
    目录结构：results/{dataset_name}/{run_dir}/seed_{i}/
    每个 seed 目录下存放：synthetic_data.csv 及可能的模型文件、配置文件等
    run_dir 命名格式：{mode}_{goal}_h{hidden_dim}_t{num_timesteps}_{num_heads}_{timestamp}
    """

    def __init__(self, results_root: str = RESULTS_ROOT):
        self.results_root = os.path.abspath(results_root)

    # ---------- 路径获取（核心接口） ----------
    def get_dataset_dir(self, dataset_name: str) -> str:
        """返回数据集根目录路径"""
        return os.path.join(self.results_root, dataset_name)

    def get_run_dir(self, dataset_name: str, run_dirname: str) -> str:
        """返回某个运行目录的路径"""
        return os.path.join(self.get_dataset_dir(dataset_name), run_dirname)

    def get_seed_dir(self, dataset_name: str, run_dirname: str, seed: int) -> str:
        """返回种子目录路径（例如 .../run_dir/seed_0）"""
        return os.path.join(self.get_run_dir(dataset_name, run_dirname), f"seed_{seed}")

    # ---------- 文件访问（模型、配置等） ----------
    def get_model_file_path(self, dataset_name: str, run_dirname: str, seed: int,
                            filename: str) -> str:
        """
        获取种子目录下的任意文件路径（例如训练好的模型文件、配置json等）
        """
        return os.path.join(self.get_seed_dir(dataset_name, run_dirname, seed), filename)

    def get_config_file_path(self, dataset_name: str, run_dirname: str, seed: int) -> str:
        """获取配置文件（假设名为 config.json）的完整路径"""
        return self.get_model_file_path(dataset_name, run_dirname, seed, 'config.json')

    def get_model_weights_path(self, dataset_name: str, run_dirname: str, seed: int,
                               weights_filename: str = 'model.pt') -> str:
        """获取模型权重文件路径（可自定义文件名）"""
        return self.get_model_file_path(dataset_name, run_dirname, seed, weights_filename)

    # ---------- 数据读取 ----------
    def read_synthetic_data(self, dataset_name: str, run_dirname: str, seed: int,
                            **read_kwargs) -> pd.DataFrame:
        """读取指定种子目录下的生成数据（synthetic_data.csv）"""
        file_path = self.get_model_file_path(dataset_name, run_dirname, seed,
                                             SYNTHETIC_DATA_FILENAME)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Synthetic data not found: {file_path}")
        return pd.read_csv(file_path, **read_kwargs)

    # ---------- 运行目录解析与查询 ----------
    def parse_run_dirname(self, dirname: str) -> Dict[str, Any]:
        """解析运行目录名，返回参数字典"""
        m = RUN_DIR_PATTERN.match(dirname)
        if m:
            return {
                'mode': m.group('mode'),
                'goal': m.group('goal'),
                'hidden_dim': int(m.group('hidden_dim')),
                'num_timesteps': int(m.group('num_timesteps')),
                'num_heads': int(m.group('num_heads')),
                'timestamp': m.group('timestamp')
            }
        return {'raw': dirname}

    def list_datasets(self) -> List[str]:
        """列出 results 下所有数据集名称"""
        if not os.path.exists(self.results_root):
            return []
        return [d for d in os.listdir(self.results_root)
                if os.path.isdir(os.path.join(self.results_root, d))]

    def list_runs(self, dataset_name: str) -> List[Dict[str, Any]]:
        """列出某个数据集下的所有运行，每个运行包含目录名、完整路径、解析后的参数"""
        dataset_dir = self.get_dataset_dir(dataset_name)
        if not os.path.exists(dataset_dir):
            return []
        runs = []
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if not os.path.isdir(item_path):
                continue
            params = self.parse_run_dirname(item)
            runs.append({
                'run_dir': item,
                'full_path': item_path,
                'params': params
            })
        return runs

    def list_seeds(self, dataset_name: str, run_dirname: str) -> List[int]:
        """列出某个运行下存在的种子编号（基于 seed_ 文件夹）"""
        run_dir = self.get_run_dir(dataset_name, run_dirname)
        if not os.path.exists(run_dir):
            return []
        seeds = []
        for item in os.listdir(run_dir):
            if item.startswith('seed_') and os.path.isdir(os.path.join(run_dir, item)):
                try:
                    seed = int(item.split('_')[1])
                    seeds.append(seed)
                except ValueError:
                    pass
        return sorted(seeds)

    def get_latest_run(self, dataset_name: str, mode: Optional[str] = None,
                       goal: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取最新（按 timestamp 降序）的运行，可按 mode/goal 过滤"""
        runs = self.list_runs(dataset_name)
        if mode:
            runs = [r for r in runs if r['params'].get('mode') == mode]
        if goal:
            runs = [r for r in runs if r['params'].get('goal') == goal]
        if not runs:
            return None
        # timestamp 格式 MMDD_HHMMSS，可直接字符串比较
        runs.sort(key=lambda x: x['params'].get('timestamp', ''), reverse=True)
        return runs[0]

    def get_run_by_params(self, dataset_name: str, mode: str, goal: str,
                          hidden_dim: int, num_timesteps: int, num_heads: int,
                          timestamp: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """根据精确参数查找运行"""
        runs = self.list_runs(dataset_name)
        for r in runs:
            p = r['params']
            if (p.get('mode') == mode and p.get('goal') == goal and
                p.get('hidden_dim') == hidden_dim and
                p.get('num_timesteps') == num_timesteps and
                p.get('num_heads') == num_heads):
                if timestamp is None or p.get('timestamp') == timestamp:
                    return r
        return None

    # ---------- 批量读取所有种子（生成数据） ----------
    def read_all_seeds_synthetic(self, dataset_name: str, run_dirname: str,
                                 **read_kwargs) -> Union[Dict[int, pd.DataFrame], pd.DataFrame]:
        """读取某个运行下所有种子的生成数据，可选合并（会添加 'seed' 列）"""
        seeds = self.list_seeds(dataset_name, run_dirname)
        if not seeds:
            raise FileNotFoundError(f"No seeds found in {self.get_run_dir(dataset_name, run_dirname)}")
        result = {}
        for seed in seeds:
            try:
                df = self.read_synthetic_data(dataset_name, run_dirname, seed, **read_kwargs)
                result[seed] = df
            except FileNotFoundError as e:
                print(f"Warning: seed {seed} missing: {e}")
        return result


# ================== 全局单例管理及便捷函数 ==================
_default_baseline_manager = None
_default_own_manager = None

def get_baseline_manager(root_dir: str = './baselines') -> BaselinePathManager:
    global _default_baseline_manager
    if _default_baseline_manager is None or _default_baseline_manager.root_dir != os.path.abspath(root_dir):
        _default_baseline_manager = BaselinePathManager(root_dir)
    return _default_baseline_manager

def get_own_manager(results_root: str = RESULTS_ROOT) -> ResultModelManager:
    global _default_own_manager
    if _default_own_manager is None or _default_own_manager.results_root != os.path.abspath(results_root):
        _default_own_manager = ResultModelManager(results_root)
    return _default_own_manager

# 便捷读取 baselines 数据
def read_baseline_full_data(model: str, seed: int, **kwargs) -> pd.DataFrame:
    return get_baseline_manager().read_full_data(model, seed, **kwargs)

def read_baseline_minority_comparison(model: str, seed: int, **kwargs) -> pd.DataFrame:
    return get_baseline_manager().read_minority_comparison(model, seed, **kwargs)

def read_baseline_minority_augment(model: str, seed: int, **kwargs) -> pd.DataFrame:
    return get_baseline_manager().read_minority_augment(model, seed, **kwargs)

# 便捷读取自己模型数据（最新运行）
def read_own_latest_synthetic(dataset_name: str, mode: str = 'minority_cond',
                              goal: str = 'fidelity', seed: int = 0, **kwargs) -> pd.DataFrame:
    mgr = get_own_manager()
    latest = mgr.get_latest_run(dataset_name, mode=mode, goal=goal)
    if latest is None:
        raise FileNotFoundError(f"No run found for {dataset_name} with mode={mode}, goal={goal}")
    return mgr.read_synthetic_data(dataset_name, latest['run_dir'], seed, **kwargs)

# ================== 对比整合工具（跨种子） ==================
def compare_baselines_and_own_across_seeds(
    baseline_manager: BaselinePathManager,
    own_manager: ResultModelManager,
    dataset_name: str,
    own_run: Optional[Dict] = None,
    minority_type: str = 'comparison'
) -> pd.DataFrame:
    """
    整合所有 baselines 模型的所有种子数据，以及自己模型的指定运行的所有种子数据
    返回一个包含 'model', 'seed' 列的 DataFrame
    """
    all_dfs = []
    # Baselines
    for model in BASELINES_MODELS:
        try:
            seeds_data = baseline_manager.read_all_seeds(
                model, SUBDIR_MINORITY, minority_type=minority_type
            )
            # 将字典转换为带 model 和seed 列的DataFrame
            for seed, df in seeds_data.items():
                df_c = df.copy()
                df_c['model'] = f'baseline_{model}'
                df_c['seed'] = seed
                all_dfs.append(df_c)
        except Exception as e:
            print(f"Warning: Could not load baseline {model}: {e}")
    # Own model
    if own_run is None:
        own_run = own_manager.get_latest_run(dataset_name, mode='minority_cond', goal='fidelity')
    if own_run:
        run_dir = own_run['run_dir']
        try:
            seeds_data = own_manager.read_all_seeds_synthetic(dataset_name, run_dir)
            params = own_run['params']
            model_label = f"own_{params.get('mode')}_{params.get('goal')}_h{params.get('hidden_dim')}_t{params.get('num_timesteps')}"
            for seed, df in seeds_data.items():
                df_c = df.copy()
                df_c['model'] = model_label
                df_c['seed'] = seed
                all_dfs.append(df_c)
        except Exception as e:
            print(f"Warning: Could not load own model: {e}")
    if not all_dfs:
        raise ValueError("No data found for comparison")