#!/usr/bin/env python3
"""
测试门型障碍生成脚本

运行方式：
python test_gate_obstacles.py

预期结果：
1. 成功创建地形
2. 看到 "Creating gate assets..." 消息
3. 看到 "Created X gate asset configurations" 消息
4. 环境中显示蓝色立柱和白色横梁
5. 机器人可以穿过门框中间
"""

import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), "legged_gym"))

from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import task_registry
import torch


def test_gate_obstacles():
    print("=" * 80)
    print("测试门型障碍生成")
    print("=" * 80)

    # 注册任务
    task_registry.register(
        "galileo_parkour", GalileoParkour, GalileoParkourCfg(), GalileoParkourCfgPPO()
    )

    # 创建环境配置
    env_cfg, train_cfg = task_registry.get_cfgs(name="galileo_parkour")

    # 修改配置以便测试
    env_cfg.env.num_envs = 4  # 只创建4个环境用于测试
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.terrain_dict = {
        "smooth slope": 0.0,
        "rough slope up": 0.0,
        "rough slope down": 0.0,
        "rough stairs up": 0.0,
        "rough stairs down": 0.0,
        "discrete": 0.0,
        "stepping stones": 0.0,
        "gaps": 0.0,
        "smooth flat": 0.0,
        "pit": 0.0,
        "wall": 0.0,
        "platform": 0.0,
        "large stairs up": 0.0,
        "large stairs down": 0.0,
        "parkour": 0.0,
        "parkour_hurdle": 0.0,
        "parkour_flat": 0.0,
        "parkour_step": 0.0,
        "parkour_gap": 0.0,
        "demo": 0.0,
        "h_hurdle_urdf": 0.0,
        "h_hurdle_geometric": 1.0,  # 100% 门型障碍
    }
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())

    print("\n配置信息：")
    print(f"  环境数量: {env_cfg.env.num_envs}")
    print(f"  地形行数: {env_cfg.terrain.num_rows}")
    print(f"  地形列数: {env_cfg.terrain.num_cols}")
    print(f"  门型障碍比例: 100%")

    try:
        # 创建环境
        print("\n创建环境...")
        env, _ = task_registry.make_env(
            name="galileo_parkour", args=None, env_cfg=env_cfg
        )

        print("\n✅ 环境创建成功！")

        # 检查地形是否有门框信息
        if hasattr(env.terrain, "gate_obstacles_dict"):
            num_gates = sum(
                len(gates) for gates in env.terrain.gate_obstacles_dict.values()
            )
            print(f"\n✅ 门框信息已记录: {num_gates} 个门框")

            # 显示每个环境的门框数量
            for (row, col), gates in env.terrain.gate_obstacles_dict.items():
                if gates:
                    print(f"  - 环境 ({row}, {col}): {len(gates)} 个门框")
                    for i, gate in enumerate(gates):
                        print(
                            f"    门框 {i}: 高度={gate['height']:.2f}m, "
                            f"宽度={gate['gate_width']:.2f}m, "
                            f"位置=({gate['x']:.2f}, {gate['y']:.2f}, {gate['z']:.2f})"
                        )
        else:
            print("\n❌ 未找到门框信息")

        # 检查是否创建了gate assets
        if hasattr(env, "gate_assets") and env.gate_assets:
            print(f"\n✅ Gate assets 已创建: {len(env.gate_assets)} 种配置")
            for config, assets in env.gate_assets.items():
                height, width, depth, thickness = config
                print(
                    f"  - 配置: 高={height}m, 宽={width}m, 深={depth}m, 柱={thickness}m"
                )
        else:
            print("\n❌ Gate assets 未创建")

        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        print("\n请观察仿真窗口：")
        print("  - 蓝色方块 = 立柱")
        print("  - 白色方块 = 横梁")
        print("  - 中间应该是镂空的")
        print("\n按 Ctrl+C 退出...")

        # 运行几步仿真以便观察
        obs = env.get_observations()
        for i in range(1000):
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            obs, _, _, _ = env.step(actions)

            if i % 100 == 0:
                print(f"  仿真步数: {i}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_gate_obstacles()
    sys.exit(0 if success else 1)



