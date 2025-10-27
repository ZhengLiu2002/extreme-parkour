# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GalileoParkourCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 1024  # 训练环境数量

    class terrain(LeggedRobotCfg.terrain):
        num_rows = 10  # 难度等级数量（200mm→500mm递进）
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
        damping = {"joint": 1.5}  # [N*m*s/rad] - 提高阻尼，减少震荡
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
            tracking_goal_vel = 1.5  # 跟踪目标速度
            tracking_yaw = 0.5  # 跟踪偏航角

            # ============ 高度和姿态稳定（关键！）============
            base_height = 3.0  # 【新增】强力约束高度跟踪（替代base_height_stability和low_posture_reward）
            no_fly = -4.0  # 【新增】强力惩罚z轴速度，防止抖动
            orientation = -1.5  # 惩罚姿态翻滚（保留，因为与ang_vel_xy互补）
            ang_vel_xy = -0.05  # 轻微惩罚角速度（保留，用于阻尼）

            # ============ 动作平滑性（简化）============
            action_rate = -0.15  # 惩罚动作变化率（提高权重，增强平滑性）
            dof_acc = -2.5e-7  # 惩罚关节加速度
            dof_error = -0.05  # 惩罚关节偏离默认姿态（略微提高）

            # ============ 碰撞和接触 ============
            collision = -10.0  # 惩罚非足部碰撞
            body_obstacle_contact = -10.0  # 惩罚机器人身体与立柱接触
            feet_stumble = -1.0  # 惩罚脚碰到垂直面
            feet_edge = -1.0  # 惩罚脚碰到边缘
            feet_contact_forces = -0.01  # 轻微惩罚过大的脚部接触力

            # ============ 钻过栏杆相关 ============
            virtual_crossbar_penalty = -8.0  # 惩罚机器人身体高度超过虚拟横杆
            strategy_efficiency = 1.5  # 奖励根据栏杆高度选择合适姿态
            obstacle_approach_speed = 0.5  # 奖励接近障碍物时的合理速度
            stable_crawl = 1.0  # 奖励稳定钻行

            # ============ 其他 ============
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
        """深度相机配置（学生网络使用）"""

        use_camera = False  # 训练教师网络时为False，训练学生网络时改为True
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
        curriculum = True  # 启用课程学习（逐渐增加栏杆高度）

    class curriculum:
        """课程学习配置：逐步增加难度"""

        # 障碍物高度课程（从低到高）
        obstacle_height_start = 0.15  # 起始高度 [m]
        obstacle_height_end = 0.50  # 最终高度 [m]
        obstacle_height_increment = 0.05  # 每次增加 [m]

        # 障碍物数量课程（从少到多）
        num_obstacles_start = 2  # 起始障碍物数量
        num_obstacles_end = 4  # 最终障碍物数量

        # 障碍物间距课程（从远到近）
        obstacle_spacing_start = [2.5, 3.0]  # 起始间距范围 [m]
        obstacle_spacing_end = [1.5, 2.5]  # 最终间距范围 [m]

        # 课程推进条件
        success_rate_threshold = 0.7  # 成功率达到70%时进入下一阶段
        min_episodes_per_stage = 500  # 每个阶段最少训练episode数


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
