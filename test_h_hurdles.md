# 程序化H型栏杆测试指南

## 概述
本项目已成功实现了在Isaac Gym中程序化创建H型栏杆障碍物，无需使用URDF文件。

## 实现的功能

### 1. terrain.py 中的新函数

#### `h_hurdle_procedural_terrain()`
- **位置**: `legged_gym/legged_gym/utils/terrain.py` 第1516行
- **功能**: 生成完整的H型栏杆地形信息
- **组件**:
  - 顶部横杆（水平圆柱，长0.7m，半径0.008m，白色）
  - 两根立柱（垂直圆柱，根据高度调整长度，半径0.008m，蓝色）
  - 两个底座（长方体，0.35×0.03×0.03m，灰色）
  - 底部连接杆（水平圆柱，长0.6m，半径0.005m，红色）

**特点**:
- 支持递进高度（20cm, 30cm, 40cm, 50cm）
- 可配置栏杆数量、间距、Y轴偏移
- 完全程序化，无需URDF文件

### 2. legged_robot.py 中的新方法

#### `_create_h_hurdle_assets()`
- **位置**: `legged_gym/legged_gym/envs/base/legged_robot.py` 第1590行
- **功能**: 创建所有需要的H型栏杆几何体assets
- **使用的Isaac Gym API**:
  - `gym.create_capsule()` - 创建圆柱体（横杆、立柱、连接杆）
  - `gym.create_box()` - 创建长方体（底座）

#### `_add_h_hurdle_static_geometry()`
- **位置**: `legged_gym/legged_gym/envs/base/legged_robot.py` 第1664行
- **功能**: 在每个环境中添加H型栏杆
- **特点**:
  - 根据URDF文件结构精确计算各组件位置
  - 使用独立的collision group，不影响机器人的actor索引
  - 确保所有环境的actor数量一致

#### `_add_obstacle_geometry()`
- **位置**: `legged_gym/legged_gym/envs/base/legged_robot.py` 第1774行
- **功能**: 添加单个障碍物几何体

### 3. Terrain类的更新

- **新属性**: `h_hurdles_dict`
  - 存储每个环境的H型栏杆信息
  - 键: (row, col) 地形坐标
  - 值: H型栏杆列表

## 如何使用

### 方法1: 在配置文件中启用

编辑 `galileo_parkour_config.py`:

```python
class terrain(LeggedRobotCfg.terrain):
    terrain_dict = {
        # ... 其他地形类型 ...
        "h_hurdle_procedural": 1.0,  # 启用程序化H型栏杆
    }
    terrain_proportions = list(terrain_dict.values())
    num_goals = 8  # 确保有足够的目标点
```

### 方法2: 在make_terrain函数中添加

编辑 `terrain.py` 的 `make_terrain()` 方法，添加新的地形类型:

```python
elif choice < self.proportions[N]:  # N是下一个可用的索引
    idx = N
    h_hurdle_procedural_terrain(
        terrain,
        num_hurdles=4,
        total_goals=self.num_goals,
        x_range=[2.0, 2.5],
        y_range=[0.0, 0.0],
        height_range=[0.2, 0.5],
        progressive_heights=True,
    )
    self.add_roughness(terrain)
```

## 测试步骤

1. **修改配置文件**
   ```bash
   cd /home/wh/RL/extreme-parkour
   # 编辑 galileo_parkour_config.py
   ```

2. **运行训练**
   ```bash
   python legged_gym/legged_gym/scripts/train.py --task=galileo_parkour
   ```

3. **运行可视化**
   ```bash
   python legged_gym/legged_gym/scripts/play.py --task=galileo_parkour
   ```

## 优势

1. **性能优化**: 
   - 不使用URDF文件，减少文件I/O
   - 复合对象作为单个asset创建，减少内存占用
   - 所有环境使用相同的asset配置，提高效率

2. **灵活性**:
   - 可以轻松调整栏杆参数（高度、宽度、颜色等）
   - 支持动态生成不同配置的栏杆
   - 易于扩展到其他类型的障碍物

3. **一致性**:
   - 确保所有环境的actor数量一致（Isaac Gym要求）
   - 使用collision group隔离障碍物和机器人

## 与URDF方法的比较

| 特性 | URDF方法 | 程序化方法 |
|------|---------|-----------|
| 文件依赖 | 需要URDF文件 | 无需外部文件 |
| 内存占用 | 每个栏杆独立加载 | 共享asset，更省内存 |
| 灵活性 | 需要修改URDF文件 | 代码中直接调整 |
| 性能 | 较慢（文件I/O） | 更快（纯内存操作） |
| 可视化 | URDF定义的颜色 | 可自定义颜色 |

## 参数说明

### h_hurdle_procedural_terrain() 参数

- `platform_len`: 起始平台长度 [米]
- `platform_height`: 起始平台高度 [米]
- `num_hurdles`: 栏杆数量（默认4个）
- `total_goals`: 目标点总数
- `x_range`: 栏杆间距范围 [米]
- `y_range`: Y轴偏移范围 [米]
- `height_range`: 栏杆高度范围 [米]
- `pad_width`: 边缘填充宽度 [米]
- `pad_height`: 边缘填充高度 [米]
- `progressive_heights`: 是否使用递进高度（0.2, 0.3, 0.4, 0.5）

## 组件尺寸（基于URDF文件）

### H_hurdel_200.urdf
- 立柱长度: 0.2m
- 顶部横杆: 0.7m × φ0.016m
- 底座: 0.35 × 0.03 × 0.03m
- 底部连接杆: 0.6m × φ0.010m

### H_hurdel_300.urdf
- 立柱长度: 0.3m
- 其他组件相同

### H_hurdel_400.urdf
- 立柱长度: 0.4m
- 其他组件相同

## 故障排查

### 问题1: 栏杆不显示
- 检查 `terrain_dict` 中是否启用了 `h_hurdle_procedural`
- 确认 `make_terrain()` 函数中是否添加了对应的分支

### 问题2: 训练时出现错误
- 检查所有环境是否创建了相同数量的栏杆
- 确认collision group设置正确（应使用`self.num_envs`）

### 问题3: 栏杆位置不正确
- 检查 `x_range` 和 `y_range` 参数
- 确认环境原点坐标正确

## 下一步工作

1. 在 `make_terrain()` 函数中集成新的地形类型
2. 调整栏杆参数以适应galileo机器人
3. 添加碰撞检测和奖励函数
4. 测试不同难度的栏杆配置

## 相关文件

- `legged_gym/legged_gym/utils/terrain.py`
- `legged_gym/legged_gym/envs/base/legged_robot.py`
- `legged_gym/legged_gym/envs/galileo/galileo_parkour_config.py`
- `legged_gym/resources/terrain/H_hurdel_*.urdf` (参考)

