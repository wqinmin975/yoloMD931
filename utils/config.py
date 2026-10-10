"""
扁平 YAML 配置加载器 (零依赖)
- 兼容 configs/*.yaml 的 "# 注释" 与 "key: value" 风格
- 支持 int / float / bool / str 自动推断
- apply_config_to_args(args, path): 把 yaml 里的键合并进 argparse 的 args 命名空间
  (yaml 键优先; 若 args 已有同名属性则覆盖, 否则新建)
"""
import os


def _parse_value(raw):
    raw = raw.strip()
    if raw == "":
        return None
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    # 去掉可能的引号
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def load_flat_config(path):
    """读取扁平 yaml -> dict。忽略空行与 '#' 注释行。"""
    cfg = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = _parse_value(val)
            cfg[key] = val
    return cfg


def apply_config_to_args(args, config_path):
    """把 yaml 配置合并进 args 命名空间。"""
    if not config_path:
        return args
    cfg = load_flat_config(config_path)
    for k, v in cfg.items():
        if v is None:
            continue
        setattr(args, k, v)
    return args
