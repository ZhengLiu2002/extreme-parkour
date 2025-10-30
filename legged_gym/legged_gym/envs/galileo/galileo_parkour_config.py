# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GalileoParkourCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 512

    class terrain(LeggedRobotCfg.terrain):
        num_rows = 8
        num_cols = 4

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 50.0}
        damping = {"joint": 1.5}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/galileo_e1_v1d6_e1r/e1_v1d6_e1r.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base_link"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            strategic_height = 0.0
            no_fly = -0.0
            orientation = -1.5
            ang_vel_xy = -0.05
            action_rate = -0.15
            dof_acc = -2.5e-7
            dof_error = -0.05
            collision = -5.0
            body_obstacle_contact = -10.0
            feet_stumble = -1.0
            feet_edge = -1.0
            feet_contact_forces = -0.01
            stand_still = -1.5
            alive_bonus = 0.5

        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        base_height_normal = 0.35
        obstacle_contact_force_threshold = 5.0
        low_hurdle_threshold = 0.30
        high_hurdle_threshold = 0.40
        obstacle_detection_range = 1.0
        post_contact_proximity_threshold = 0.5
        contact_force_penalty_scaling = 50.0
        max_contact_force_penalty = 2.0
        enable_contact_force_logging = True
        obstacle_safe_distance = 1.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 6.0
        heading_command = True

        class ranges:
            lin_vel_x = [0.3, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0, 0]
            heading = [0, 0]

    class depth(LeggedRobotCfg.depth):
        use_camera = False
        camera_num_envs = 32
        position = [0.28, 0, 0.08]
        angle = [-15, 5]
        update_interval = 4
        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 90
        buffer_len = 2
        near_clip = 0.0
        far_clip = 3.5
        dis_noise = 0.01
        scale = 1.0
        invert = True

    class terrain(LeggedRobotCfg.terrain):
        terrain_dict = {"h_hurdle": 1.0}
        terrain_proportions = list(terrain_dict.values())
        num_goals = 8
        curriculum = True


class GalileoParkourCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        num_mini_batches = 4
        num_learning_epochs = 5

    class depth_encoder(LeggedRobotCfgPPO.depth_encoder):
        if_depth = GalileoParkourCfg.depth.use_camera
        depth_shape = GalileoParkourCfg.depth.resized
        buffer_len = GalileoParkourCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.0e-3
        num_steps_per_env = GalileoParkourCfg.depth.update_interval * 24

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "galileo"
        num_steps_per_env = 24
        max_iterations = 100000
        save_interval = 100
