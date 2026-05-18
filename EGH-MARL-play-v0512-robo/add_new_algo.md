根据我对代码库的分析，如果要添加一种新算法，需要修改以下地方：

## 添加新算法需要修改的地方

### 1. **算法实现文件** (`harl/algorithms/actors/`)
创建新的算法类文件，例如 `new_algo.py`：
- 继承自 `OnPolicyBase`（如果是on-policy算法）或 `OffPolicyBase`（如果是off-policy算法）
- 实现 `__init__`, `update`, `train` 等核心方法
- 如果需要特殊的网络结构，可以重写actor初始化

### 2. **算法注册** (`harl/algorithms/actors/__init__.py`)
在文件中添加：
```python
from harl.algorithms.actors.new_algo import NewAlgo
```
并在 `ALGO_REGISTRY` 字典中添加映射：
```python
ALGO_REGISTRY = {
    # ... 现有算法
    "new_algo": NewAlgo,
}
```

### 3. **Critic注册** (`harl/algorithms/critics/__init__.py`)
如果需要新的critic：
- 在 `harl/algorithms/critics/` 中创建新的critic类
- 在 `__init__.py` 中导入并注册到 `CRITIC_REGISTRY`

### 4. **配置文件** (`harl/configs/algos_cfgs/`)
创建新的配置文件，例如 `new_algo.yaml`：
- 定义算法的超参数（学习率、clip参数、网络结构等）
- 参考现有的 [mappo.yaml](file:///home/yuxin/fuzzymarl/casia_fuzzy_attack/EGH-MARL-play-v0512-robo/harl/configs/algos_cfgs/mappo.yaml) 或 [egnn_mappo.yaml](file:///home/yuxin/fuzzymarl/casia_fuzzy_attack/EGH-MARL-play-v0512-robo/harl/configs/algos_cfgs/egnn_mappo.yaml)

### 5. **Runner注册** (`harl/runners/__init__.py`)
在 `RUNNER_REGISTRY` 中添加映射：
```python
RUNNER_REGISTRY = {
    # ... 现有算法
    "new_algo": OnPolicyMARunner,  # 或其他合适的runner
}
```

### 6. **模型文件** (可选)
如果需要特殊的网络结构：
- 在 `harl/models/policy_models/` 中创建新的policy模型
- 在 `harl/models/value_function_models/` 中创建新的value function模型
- 在算法类的 `__init__` 中使用这些新模型

### 7. **训练脚本**
在 `examples/` 目录下创建新的训练脚本，或直接使用命令行：
```bash
python train.py --algo new_algo --env your_env --exp_name test
```

## 关键文件位置总结

| 文件类型 | 路径 | 说明 |
|---------|------|------|
| 算法实现 | `harl/algorithms/actors/` | 继承Base类实现算法逻辑 |
| Critic实现 | `harl/algorithms/critics/` | 价值网络实现 |
| 算法注册 | `harl/algorithms/actors/__init__.py` | 注册到ALGO_REGISTRY |
| Critic注册 | `harl/algorithms/critics/__init__.py` | 注册到CRITIC_REGISTRY |
| 配置文件 | `harl/configs/algos_cfgs/` | YAML格式的超参数配置 |
| Runner注册 | `harl/runners/__init__.py` | 注册到RUNNER_REGISTRY |
| 模型定义 | `harl/models/policy_models/` | 策略网络定义 |
| 训练脚本 | `examples/` | Shell脚本示例 |

## 示例：添加一个基于MAPPO的新算法

如果你想添加一个类似 `egnn_mappo` 的新算法，可以参考以下文件：
- [egnn_mappo.py](file:///home/yuxin/fuzzymarl/casia_fuzzy_attack/EGH-MARL-play-v0512-robo/harl/algorithms/actors/egnn_mappo.py) - 算法实现
- [egnn_mappo.yaml](file:///home/yuxin/fuzzymarl/casia_fuzzy_attack/EGH-MARL-play-v0512-robo/harl/configs/algos_cfgs/egnn_mappo.yaml) - 配置文件
- [egnn_policy.py](file:///home/yuxin/fuzzymarl/casia_fuzzy_attack/EGH-MARL-play-v0512-robo/harl/models/policy_models/egnn_policy.py) - 策略网络

主要需要修改的核心文件是这7个地方。你想添加什么类型的算法？我可以提供更具体的指导。
        