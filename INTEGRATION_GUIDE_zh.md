# H型栏杆程序化障碍物 - 集成指南

## 📝 概述

本指南说明如何将新创建的程序化H型栏杆集成到你的训练环境中。

## ✅ 已完成的工作

1. ✅ 创建了 `h_hurdle_procedural_terrain()` 函数在 `terrain.py` 中
2. ✅ 实现了 `_create_h_hurdle_assets()` 方法在 `legged_robot.py` 中
3. ✅ 实现了 `_add_h_hurdle_static_geometry()` 方法在 `legged_robot.py` 中
4. ✅ 更新了 `Terrain` 类以支持 `h_hurdles_dict`

## 🔧 集成步骤

### 步骤1: 更新 terrain.py 的 make_terrain() 函数

在 `terrain.py` 的 `make_terrain()` 方法中添加新的地形类型。找到类似这样的代码结构：

```python
def make_terrain(self, choice, difficulty):
    # ... 现有代码 ...
    
    elif choice < self.proportions[23]:  # 使用下一个可用的索引号
        idx = 24  # 对应的idx
        # 程序化H型栏杆地形
        h_hurdle_procedural_terrain(
            terrain,
            num_hurdles=4,  # 4个栏杆
            total_goals=self.num_goals,
            x_range=[2.0, 2.5],  # 栏杆间距（米）
            y_range=[0.0, 0.0],  # 居中放置
            height_range=[0.2, 0.5],
            progressive_heights=True,  # 递进高度：20,30,40,50cm
        )
        self.add_roughness(terrain)
    
    terrain.idx = idx
    return terrain
```

### 步骤2: 更新配置文件

编辑 `galileo_parkour_config.py` (或你的配置文件):

```python
class terrain(LeggedRobotCfg.terrain):
    terrain_dict = {
        "smooth slope": 0.0,
        # ... 其他地形类型 ...
        "crawl_through": 0.0,  # 可以关闭其他栏杆类型
        "jump_over": 0.0,
        "h_hurdle_procedural": 1.0,  # 启用程序化H型栏杆！
    }
    terrain_proportions = list(terrain_dict.values())
    num_goals = 8  # 确保足够的目标点
```

### 步骤3: 验证配置

运行以下命令检查配置是否正确：

```bash
cd /home/wh/RL/extreme-parkour
python -c "from legged_gym.envs.galileo.galileo_parkour_config import GalileoParkourCfg; print('Config OK')"
```

## 🎮 测试新障碍物

### 测试1: 可视化测试

```bash
# 如果你已经有训练好的模型
python legged_gym/legged_gym/scripts/play.py --task=galileo_parkour

# 或者创建一个简单的环境测试
python test_gate_obstacles.py  # 如果这个文件存在
```

### 测试2: 训练测试

```bash
python legged_gym/legged_gym/scripts/train.py --task=galileo_parkour --num_envs=64
```

观察：
- ✅ 是否能看到H型栏杆
- ✅ 栏杆是否有正确的颜色（白色横杆、蓝色立柱、灰色底座、红色连接杆）
- ✅ 机器人是否能与栏杆碰撞
- ✅ 是否有错误信息

## 🎨 自定义参数

### 调整栏杆高度

在 `make_terrain()` 中修改：

```python
h_hurdle_procedural_terrain(
    terrain,
    height_range=[0.15, 0.35],  # 降低难度：15-35cm
    progressive_heights=False,  # 关闭递进高度，随机选择
)
```

### 调整栏杆间距

```python
h_hurdle_procedural_terrain(
    terrain,
    x_range=[3.0, 4.0],  # 增加间距，给机器人更多空间
    y_range=[-0.2, 0.2],  # 允许Y轴偏移
)
```

### 调整栏杆数量

```python
h_hurdle_procedural_terrain(
    terrain,
    num_hurdles=2,  # 减少到2个栏杆（初期训练）
    total_goals=self.num_goals,
)
```

## 📊 与其他地形类型混合使用

如果你想同时使用多种地形：

```python
class terrain(LeggedRobotCfg.terrain):
    terrain_dict = {
        "smooth flat": 0.1,
        "parkour_gap": 0.2,
        "jump_over": 0.2,
        "h_hurdle_procedural": 0.5,  # 50%的环境使用H型栏杆
    }
```

## 🐛 故障排查

### 问题1: 看不到栏杆

**可能原因**:
- `make_terrain()` 中没有添加对应分支
- `proportions` 数组配置错误

**解决方案**:
```python
# 在terrain.py的curiculum()或randomized_terrain()中打印choice值
print(f"Choice: {choice}, Proportions: {self.proportions}")
```

### 问题2: 报错 "AttributeError: 'Terrain' object has no attribute 'h_hurdles_dict'"

**原因**: terrain.py没有正确更新

**解决方案**: 确认以下代码存在于 `Terrain.__init__()`:
```python
self.h_hurdles_dict = {}
```

### 问题3: 训练时崩溃

**可能原因**: actor数量不一致

**解决方案**: 确保每个环境创建相同数量的栏杆。检查 `_add_h_hurdle_static_geometry()` 方法。

### 问题4: 栏杆位置偏移

**原因**: 环境原点坐标计算错误

**解决方案**: 在 `_add_h_hurdle_static_geometry()` 中添加调试输出：
```python
print(f"Env {env_id}: hurdles at {[(h['x'], h['y'], h['z']) for h in hurdles]}")
```

## 📈 性能优化建议

### 1. 减少栏杆组件数量（如果性能不足）

修改 `_add_h_hurdle_static_geometry()` 方法，只创建关键组件：

```python
# 注释掉底座和连接杆，只保留立柱和横杆
# self._add_obstacle_geometry(...) # 底座
# self._add_obstacle_geometry(...) # 连接杆
```

### 2. 使用简化版本

如果只需要门框效果，可以使用更简单的 `h_hurdle_geometric_terrain()`:

```python
elif choice < self.proportions[22]:
    idx = 23
    h_hurdle_geometric_terrain(
        terrain,
        num_hurdles=4,
        gate_width=0.8,
        gate_depth=0.10,
    )
```

## 📚 代码参考位置

| 功能 | 文件 | 行号 |
|------|------|------|
| H型栏杆生成函数 | `terrain.py` | 1516 |
| 创建assets | `legged_robot.py` | 1590 |
| 添加到环境 | `legged_robot.py` | 1664 |
| Terrain类初始化 | `terrain.py` | 77 |
| 存储栏杆信息 | `terrain.py` | 545-555 |

## 🎯 下一步建议

1. **添加奖励函数**: 在配置文件的 `rewards` 类中添加针对H型栏杆的奖励
2. **课程学习**: 从低栏杆开始，逐步增加高度
3. **多样化**: 创建不同风格的栏杆（圆形、方形等）
4. **传感器**: 添加障碍物检测传感器帮助机器人判断栏杆高度

## ✨ 完整示例

以下是一个完整的集成示例，展示如何在 `make_terrain()` 中添加H型栏杆：

```python
# 在 terrain.py 的 make_terrain() 函数中

def make_terrain(self, choice, difficulty):
    terrain = terrain_utils.SubTerrain(
        "terrain",
        width=self.length_per_env_pixels,
        length=self.width_per_env_pixels,
        vertical_scale=self.cfg.vertical_scale,
        horizontal_scale=self.cfg.horizontal_scale,
    )
    
    # ... 前面的其他地形类型 ...
    
    elif choice < self.proportions[23]:  # 新增！
        idx = 24
        # 程序化H型栏杆 - 完整结构，包含底座和连接杆
        h_hurdle_procedural_terrain(
            terrain,
            platform_len=2.5,
            platform_height=0.0,
            num_hurdles=4,
            total_goals=self.num_goals,
            x_range=[2.0, 2.5],
            y_range=[0.0, 0.0],
            height_range=[0.2, 0.5],
            pad_width=0.1,
            pad_height=0.5,
            progressive_heights=True,  # 20cm -> 30cm -> 40cm -> 50cm
        )
        self.add_roughness(terrain)
    
    terrain.idx = idx
    return terrain
```

## 📞 支持

如果遇到问题，请检查：
1. 日志输出中是否有 "Creating procedural H-shaped hurdle assets..."
2. 是否有 "Created N H-hurdle asset configurations" 消息
3. Isaac Gym 版本是否支持 `create_capsule()` 和 `create_box()` 方法

## 🎉 总结

你现在已经拥有：
- ✅ 完全程序化的H型栏杆生成系统
- ✅ 无需URDF文件，更高效的内存使用
- ✅ 灵活的参数配置
- ✅ 与Isaac Gym完全兼容的实现

祝训练顺利！🚀

