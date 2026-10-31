import warnings
import os
import sys

from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import DataConversionWarning

def setup_jupyter_env():
    # 1. 忽略特定类型的警告
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    # 2. 关键：统一路径逻辑
    # 获取此 env_config.py 所在的绝对路径（即项目根目录）
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 切换当前工作目录到根目录
    # 这一步非常重要，它解决了您“results目录不存在”的问题
    os.chdir(root_dir)
    
    # 将根目录添加到 sys.path 
    if root_dir not in sys.path:
        sys.path.append(root_dir)

    # 3. 字体设置
    try:
        from utils.set_fonts import apply_times_with_cn
        apply_times_with_cn()
    except ImportError:
        print("⚠️ Warning: Could not import font setup.")

    print(f"🚀 环境初始化完成。")
    #print(f"📂 当前工作目录: {os.getcwd()}")
    #print(f"📦 已添加搜索路径: {root_dir}")

# 执行初始化
if __name__ != "__main__":
    setup_jupyter_env()