# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GalileoParkourCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 512  # 训练环境数量

    class terrain(LeggedRobotCfg.terrain):
        num_rows = 8  # 难度等级数量（200mm→500mm递进）
        num_cols = 4  # 只有1种地形类型（h_hurdle）

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # x,y,z [m] - 降低初始高度，与base_height_normal一致
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_hip_joint": -0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]
            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 0.8,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 0.8,  # [rad]
            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"  # PD控制器类型
        # 增强关节刚度以提高稳定性
        stiffness = {"joint": 50.0}  # [N*m/rad] - 提高刚度，增强稳定性
        damping = {"joint": 1.5}  # [N*m*s/rad] - 提高阻尼以匹配高刚度，防止振荡
        action_scale = 0.25  # 动作缩放系数
        decimation = 4  # 控制频率降采样

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/galileo_e1_v1d6_e1r/e1_v1d6_e1r.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base_link"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1  # 1表示禁用自碰撞检测
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            # ============ 核心任务奖励 ============
            tracking_goal_vel = 1.5  # 跟踪目标速度 - 激励机器人快速通过障碍
            tracking_yaw = 0.5  # 跟踪偏航角 - 保持正确方向

            # ============ 高度和姿态稳定 ============
            base_height_regional = 0.5  # 降低权重，避免机器人躺平也能获得高奖励
            no_fly = -0.5  # 大幅减弱！只惩罚持续的正向Z速度，允许短暂跳跃
            orientation = -1.5  # 惩罚姿态翻滚（防止摔倒）
            ang_vel_xy = -0.05  # 轻微惩罚角速度（阻尼效果）

            # ============ 动作平滑性 ============
            action_rate = -0.15  # 惩罚动作变化率（增强平滑性）
            dof_acc = -2.5e-7  # 惩罚关节加速度
            dof_error = -0.05  # 惩罚关节偏离默认姿态

            # ============ 碰撞和接触（核心约束！）============
            collision = -10.0  # 惩罚非足部碰撞
            body_obstacle_contact = -20.0  # 这是核心约束，必须不惜一切代价避免碰撞
            feet_stumble = -1.0  # 惩罚脚碰到垂直面
            feet_edge = -1.0  # 惩罚脚碰到边缘
            feet_contact_forces = -0.01  # 轻微惩罚过大的脚部接触力

            # ============ 基本激励 ============
            stand_still = -0.5  # 惩罚静止不动
            alive_bonus = 0.5  # 存活奖励

        soft_dof_pos_limit = 0.9  # 关节位置软限制
        base_height_target = 0.25  # 钻过栏杆时的目标低姿态 [m]
        base_height_normal = 0.35  # 平地正常站立高度 [m] - 新增，提高稳定性

        # 钻过栏杆相关参数
        obstacle_contact_force_threshold = 5.0  # 接触力阈值 [N]
        low_hurdle_threshold = 0.30  # 低栏杆高度阈值 [m]（<=30cm保持更低姿态）
        high_hurdle_threshold = 0.40  # 高栏杆高度阈值 [m]（>=40cm需要最低姿态）
        obstacle_detection_range = 1.0  # 障碍物检测范围 [m]

        # 接触力惩罚相关参数
        post_contact_proximity_threshold = 0.5  # 柱子附近检测范围 [m]
        contact_force_penalty_scaling = 50.0  # 接触力惩罚缩放因子 [N]
        max_contact_force_penalty = 2.0  # 最大接触力惩罚倍数
        enable_contact_force_logging = True  # 是否启用接触力日志输出

        # 区域感知高度奖励参数
        obstacle_safe_distance = 1.0  # 障碍物"安全距离" [m]：超过此距离激活平地高度奖励

    class commands(LeggedRobotCfg.commands):
        curriculum = False  # 速度指令课程学习
        num_commands = 4  # 指令数量
        resampling_time = 6.0  # 指令重采样时间[s]
        heading_command = True

        class ranges:
            lin_vel_x = [0.3, 1.0]  # x方向线速度范围 [m/s]
            lin_vel_y = [0.0, 0.0]  # y方向线速度（保持直行）
            ang_vel_yaw = [0, 0]  # 偏航角速度
            heading = [0, 0]

    class depth(LeggedRobotCfg.depth):
        # 【重要】训练策略：
        # 阶段1（Teacher训练）: use_camera = False
        #   - Critic使用特权观测（包括障碍物绝对位置）
        #   - Actor学习基础运动模式
        # 阶段2（Student训练）: use_camera = True
        #   - Actor使用深度相机替代特权观测
        #   - Critic仍然使用特权观测指导训练
        use_camera = False  # 当前处于Teacher训练阶段
        camera_num_envs = 32  # 使用深度相机的环境数量

        # 优化相机位置和角度
        position = [0.28, 0, 0.08]  # 相机位置 [x, y, z]（略微提高和前移）
        angle = [-15, 5]  # 俯仰角范围 [deg]（向下看更多，看到更近的障碍）

        update_interval = 4  # 更新间隔（每4步更新一次）
        original = (106, 60)  # 原始分辨率
        resized = (87, 58)  # 调整后分辨率
        horizontal_fov = 90  # 水平视场角 [deg]（略微增加视野）
        buffer_len = 2  # 缓冲区长度

        near_clip = 0.0  # 近裁剪面 [m]
        far_clip = 3.5  # 远裁剪面 [m]（从2.5增加到3.5，看得更远）
        dis_noise = 0.01  # 距离噪声 [m]

        scale = 1.0
        invert = True

    class terrain(LeggedRobotCfg.terrain):
        terrain_dict = {
            "h_hurdle": 1.0,  # H型栏杆：两根立柱+悬空横梁（高度200mm-500mm递进）
        }
        terrain_proportions = list(terrain_dict.values())
        num_goals = 8  # 目标点数量（起点+4个栏杆+终点+额外2个）
        # 【修复】启用 Isaac Gym 默认的课程学习机制
        # terrain_levels (行) 会映射到 difficulty (0.0-1.0)
        # terrain.py 中的 make_terrain 将根据 difficulty 调整栏杆高度
        curriculum = True


class GalileoParkourCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        # 基础PPO参数
        entropy_coef = 0.01  # 熵系数（鼓励探索）

        num_mini_batches = 4  # Mini-batch数量
        num_learning_epochs = 5  # 每批数据的训练轮数

    class depth_encoder(LeggedRobotCfgPPO.depth_encoder):
        """深度编码器配置"""

        if_depth = GalileoParkourCfg.depth.use_camera
        depth_shape = GalileoParkourCfg.depth.resized
        buffer_len = GalileoParkourCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.0e-3
        num_steps_per_env = GalileoParkourCfg.depth.update_interval * 24

    class runner(LeggedRobotCfgPPO.runner):
        # 实验配置
        run_name = ""
        experiment_name = "galileo"  # 区分实验名称

        # 训练步数配置
        num_steps_per_env = 24  # 每个环境收集24步经验
        max_iterations = 100000  # 增加最大训练迭代次数
        save_interval = 100  # 每100次迭代保存一次模型
