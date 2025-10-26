# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(
            cfg.terrain_proportions
        )
        self.proportions = [
            np.sum(cfg.terrain_proportions[: i + 1])
            for i in range(len(cfg.terrain_proportions))
        ]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        # self.env_slope_vec = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        self.num_goals = cfg.num_goals

        # Store URDF hurdle positions for each environment: dict mapping (row, col) -> list of hurdles
        # Each hurdle is a tuple: (x, y, z, urdf_file)
        self.urdf_hurdles = {}

        # Store geometric gate obstacles for each environment: dict mapping (row, col) -> list of gates
        # Each gate is a dict with keys: x, y, z, height, gate_width, gate_depth, post_thickness
        self.gate_obstacles_dict = {}

        # Store procedural H-shaped hurdles for each environment: dict mapping (row, col) -> list of h_hurdles
        # Each h_hurdle is a dict with complete component information
        self.h_hurdles_dict = {}

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            if hasattr(cfg, "max_difficulty"):
                self.curiculum(random=True, max_difficulty=cfg.max_difficulty)
            else:
                self.curiculum(random=True)
            # self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, self.x_edge_mask = (
                    convert_heightfield_to_trimesh(
                        self.height_field_raw,
                        self.cfg.horizontal_scale,
                        self.cfg.vertical_scale,
                        self.cfg.slope_treshold,
                    )
                )
                half_edge_width = int(
                    self.cfg.edge_width_thresh / self.cfg.horizontal_scale
                )
                structure = np.ones((half_edge_width * 2 + 1, 1))
                self.x_edge_mask = binary_dilation(
                    self.x_edge_mask, structure=structure
                )
                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(
                        target_count=int(0.05 * self.triangles.shape[0]),
                        aggressiveness=7,
                        preserve_border=True,
                        verbose=10,
                    )

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert (
                    cfg.hf2mesh_method == "fast"
                ), "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(
                    self.height_field_raw,
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    max_error=cfg.max_error,
                )
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(-0.2, 1.2)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self, random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = (
                    i / (self.cfg.num_rows - 1) if self.cfg.num_rows > 1 else 0.0
                )
                choice = j / self.cfg.num_cols + 0.001
                if random:
                    if max_difficulty:
                        terrain = self.make_terrain(choice, np.random.uniform(0.7, 1))
                    else:
                        terrain = self.make_terrain(choice, np.random.uniform(0, 1))
                else:
                    terrain = self.make_terrain(choice, difficulty)

                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def add_roughness(self, terrain, difficulty=1):
        max_height = (
            self.cfg.height[1] - self.cfg.height[0]
        ) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(
            terrain,
            min_height=-height,
            max_height=height,
            step=0.005,
            downsampled_scale=self.cfg.downsampled_scale,
        )

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        slope = difficulty * 0.4
        step_height = 0.02 + 0.14 * difficulty
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        if choice < self.proportions[0]:
            idx = 0
            if choice < self.proportions[0] / 2:
                idx = 1
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
            # self.add_roughness(terrain)
        elif choice < self.proportions[2]:
            idx = 2
            if choice < self.proportions[1]:
                idx = 3
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[4]:
            idx = 4
            if choice < self.proportions[3]:
                idx = 5
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.31, step_height=step_height, platform_size=3.0
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[5]:
            idx = 6
            num_rectangles = 20
            rectangle_min_size = 0.5
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[6]:
            idx = 7
            stones_size = 1.5 - 1.2 * difficulty
            # terrain_utils.stepping_stones_terrain(terrain, stone_size=stones_size, stone_distance=0.1, stone_distance_rand=0, max_height=0.04*difficulty, platform_size=2.)
            half_sloped_terrain(
                terrain, wall_width=4, start2center=0.5, max_height=0.00
            )
            stepping_stones_terrain(
                terrain,
                stone_size=1.5 - 0.2 * difficulty,
                stone_distance=0.0 + 0.4 * difficulty,
                max_height=0.2 * difficulty,
                platform_size=1.2,
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[7]:
            idx = 8
            # gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            gap_parkour_terrain(terrain, difficulty, platform_size=4)
            self.add_roughness(terrain)
        elif choice < self.proportions[8]:
            idx = 9
            self.add_roughness(terrain)
            # pass
        elif choice < self.proportions[9]:
            idx = 10
            pit_terrain(terrain, depth=pit_depth, platform_size=4.0)
        elif choice < self.proportions[10]:
            idx = 11
            if self.cfg.all_vertical:
                half_slope_difficulty = 1.0
            else:
                difficulty *= 1.3
                if not self.cfg.no_flat:
                    difficulty -= 0.1
                if difficulty > 1:
                    half_slope_difficulty = 1.0
                elif difficulty < 0:
                    self.add_roughness(terrain)
                    terrain.slope_vector = np.array([1, 0.0, 0]).astype(np.float32)
                    return terrain
                else:
                    half_slope_difficulty = difficulty
            wall_width = 4 - half_slope_difficulty * 4
            # terrain_utils.wall_terrain(terrain, height=1, start2center=0.7)
            # terrain_utils.tanh_terrain(terrain, height=1.0, start2center=0.7)
            if self.cfg.flat_wall:
                half_sloped_terrain(
                    terrain, wall_width=4, start2center=0.5, max_height=0.00
                )
            else:
                half_sloped_terrain(
                    terrain, wall_width=wall_width, start2center=0.5, max_height=1.5
                )
            max_height = terrain.height_field_raw.max()
            top_mask = terrain.height_field_raw > max_height - 0.05
            self.add_roughness(terrain, difficulty=1)
            terrain.height_field_raw[top_mask] = max_height
        elif choice < self.proportions[11]:
            idx = 12
            # half platform terrain
            half_platform_terrain(terrain, max_height=0.1 + 0.4 * difficulty)
            self.add_roughness(terrain, difficulty=1)
        elif choice < self.proportions[13]:
            idx = 13
            height = 0.1 + 0.3 * difficulty
            if choice < self.proportions[12]:
                idx = 14
                height *= -1
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=1.0, step_height=height, platform_size=3.0
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[14]:
            x_range = [-0.1, 0.1 + 0.3 * difficulty]  # offset to stone_len
            y_range = [0.2, 0.3 + 0.1 * difficulty]
            stone_len = [
                0.9 - 0.3 * difficulty,
                1 - 0.2 * difficulty,
            ]  # 2 * round((0.6) / 2.0, 1)
            incline_height = 0.25 * difficulty
            last_incline_height = incline_height + 0.1 - 0.1 * difficulty
            parkour_terrain(
                terrain,
                num_stones=self.num_goals - 2,
                x_range=x_range,
                y_range=y_range,
                incline_height=incline_height,
                stone_len=stone_len,
                stone_width=1.0,
                last_incline_height=last_incline_height,
                pad_height=0,
                pit_depth=[0.2, 1],
            )
            idx = 15
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif choice < self.proportions[15]:
            idx = 16
            parkour_hurdle_terrain(
                terrain,
                num_stones=self.num_goals - 2,
                # stone_len=0.1 + 0.3 * difficulty,
                stone_len=0.1,
                hurdle_height_range=[0.1 + 0.1 * difficulty, 0.35 + 0.25 * difficulty],
                pad_height=0,
                x_range=[1.5, 2.5],
                # y_range=self.cfg.y_range,
                y_range=[-0.05, 0.05],
                half_valid_width=[0.75, 0.8],
            )
            # terrain.height_field_raw[:] = 0
            self.add_roughness(terrain)
        elif choice < self.proportions[16]:
            idx = 17
            parkour_hurdle_terrain(
                terrain,
                num_stones=self.num_goals - 2,
                stone_len=0.1 + 0.3 * difficulty,
                hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty],
                pad_height=0,
                y_range=self.cfg.y_range,
                half_valid_width=[0.45, 1],
                flat=True,
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[17]:
            idx = 18
            parkour_step_terrain(
                terrain,
                num_stones=self.num_goals - 2,
                step_height=0.1 + 0.35 * difficulty,
                x_range=[0.3, 1.5],
                y_range=self.cfg.y_range,
                half_valid_width=[0.5, 1],
                pad_height=0,
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[18]:
            idx = 19
            parkour_gap_terrain(
                terrain,
                num_gaps=self.num_goals - 2,
                gap_size=0.1 + 0.7 * difficulty,
                gap_depth=[0.2, 1],
                pad_height=0,
                x_range=[0.8, 1.5],
                y_range=self.cfg.y_range,
                half_valid_width=[0.6, 1.2],
                # flat=True
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[19]:
            idx = 20
            demo_terrain(terrain)
            self.add_roughness(terrain)
        elif choice < self.proportions[20]:
            idx = 21
            # H-shaped URDF hurdle terrain with progressive heights (20, 30, 40, 50cm)
            h_hurdle_terrain(
                terrain,
                num_hurdles=4,  # Fixed 4 hurdles for progressive training
                total_goals=self.num_goals,  # Pass total goals to ensure consistent array size
                x_range=[2.0, 2.5],  # Consistent spacing for learning
                y_range=[2.0, 2.5],  # Center alignment for easier learning
                height_range=[0.2, 0.5],
                pad_height=0,
                progressive_heights=True,  # Enable progressive height sequence
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[21]:
            idx = 22
            # 几何体H型栏杆地形（使用几何体actors，真正镂空的门框）
            h_hurdle_geometric_terrain(
                terrain,
                num_hurdles=4,  # 4个栏杆递进训练
                total_goals=self.num_goals,  # 确保目标点数组大小一致
                x_range=[2.0, 2.5],  # 栏杆间距（米）
                y_range=[0.0, 0.0],  # 完全居中，确保阻挡路径
                height_range=[0.2, 0.5],  # 高度范围
                pad_height=0,
                progressive_heights=True,  # 启用递进高度序列（20,30,40,50cm）
                gate_width=0.8,  # 门框宽度80cm（给机器人足够空间）
                gate_depth=0.10,  # 门框厚度10cm（更明显）
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[22]:
            idx = 23
            # 钻过模式：两根立柱 + 虚拟横杆高度约束
            crawl_through_hurdle_terrain(
                terrain,
                num_hurdles=4,  # 4个栏杆递进训练
                total_goals=self.num_goals,  # 确保目标点数组大小一致
                x_range=[2.0, 2.5],  # 栏杆间距（米）
                y_range=[0.0, 0.0],  # 完全居中，确保阻挡路径
                height_range=[0.2, 0.5],  # 虚拟横杆高度范围
                pad_height=0,
                progressive_heights=True,  # 启用递进高度序列（20,30,40,50cm）
                post_width=0.12,  # 立柱宽度12cm
                post_depth=0.12,  # 立柱深度12cm
                passage_width=0.7,  # 通道宽度70cm
            )
            self.add_roughness(terrain)
        elif choice < self.proportions[23]:
            idx = 24
            # 跳跃模式：实心墙障碍
            jump_over_wall_terrain(
                terrain,
                num_walls=4,  # 4个墙体递进训练
                total_goals=self.num_goals,  # 确保目标点数组大小一致
                x_range=[2.0, 2.5],  # 墙体间距（米）
                y_range=[-0.3, 0.3],  # 允许一定Y轴偏移
                height_range=[0.2, 0.5],  # 墙体高度范围
                pad_height=0,
                progressive_heights=True,  # 启用递进高度序列（20,30,40,50cm）
                wall_depth=0.15,  # 墙体厚度15cm
                wall_width=1.0,  # 墙体宽度1米
            )
            self.add_roughness(terrain)
        # np.set_printoptions(precision=2)
        # print(np.array(self.proportions), choice)
        terrain.idx = idx
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int(
            (self.env_length / 2.0 - 0.5) / terrain.horizontal_scale
        )  # within 1 meter square range
        x2 = int((self.env_length / 2.0 + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = (
                np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
            )
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.terrain_type[i, j] = terrain.idx
        self.goals[i, j, :, :2] = terrain.goals + [
            i * self.env_length,
            j * self.env_width,
        ]
        # self.env_slope_vec[i, j] = terrain.slope_vector

        # Store URDF hurdle positions if they exist
        if hasattr(terrain, "hurdle_positions") and terrain.hurdle_positions:
            # Transform hurdle positions to world coordinates
            hurdles_world = []
            for (
                hurdle_x,
                hurdle_y,
                hurdle_z,
                hurdle_height,
                urdf_file,
            ) in terrain.hurdle_positions:
                world_x = hurdle_x + i * self.env_length
                world_y = hurdle_y + j * self.env_width
                world_z = hurdle_z + env_origin_z
                hurdles_world.append(
                    (world_x, world_y, world_z, hurdle_height, urdf_file)
                )
            self.urdf_hurdles[(i, j)] = hurdles_world

        # Store geometric gate obstacles if they exist
        if hasattr(terrain, "gate_obstacles") and terrain.gate_obstacles:
            # Transform gate positions to world coordinates
            gates_world = []
            for gate_info in terrain.gate_obstacles:
                gate_world = gate_info.copy()
                gate_world["x"] = gate_info["x"] + i * self.env_length
                gate_world["y"] = gate_info["y"] + j * self.env_width
                gate_world["z"] = gate_info["z"] + env_origin_z
                gates_world.append(gate_world)
            self.gate_obstacles_dict[(i, j)] = gates_world

        # Store procedural H-shaped hurdles if they exist
        if hasattr(terrain, "h_hurdles") and terrain.h_hurdles:
            # Transform hurdle positions to world coordinates
            hurdles_world = []
            for hurdle_info in terrain.h_hurdles:
                hurdle_world = hurdle_info.copy()
                hurdle_world["x"] = hurdle_info["x"] + i * self.env_length
                hurdle_world["y"] = hurdle_info["y"] + j * self.env_width
                hurdle_world["z"] = hurdle_info["z"] + env_origin_z
                hurdles_world.append(hurdle_world)
            self.h_hurdles_dict[(i, j)] = hurdles_world


def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -1000
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0


def gap_parkour_terrain(terrain, difficulty, platform_size=2.0):
    gap_size = 0.1 + 0.3 * difficulty
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -400
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0

    slope_angle = 0.1 + difficulty * 1
    offset = 1 + 9 * difficulty  # 10
    scale = 15
    wall_center_x = [center_x - x1, center_x, center_x + x1]
    wall_center_y = [center_y - y1, center_y, center_y + y1]

    # for i in range(center_y + y1, center_y + y2):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)

    # for i in range(center_y - y2, center_y - y1):
    #     for j in range(center_x-x1, center_x + x1):
    #         for w in wall_center_x:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[j, i] < height:
    #                 terrain.height_field_raw[j, i] = int(height)

    # for i in range(center_x + x1, center_x + x2):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)

    # for i in range(center_x - x2, center_x - x1):
    #     for j in range(center_y-y1, center_y + y1):
    #         for w in wall_center_y:
    #             height = scale * (-(slope_angle * np.abs(j - w)) + offset)
    #             if terrain.height_field_raw[i, j] < height:
    #                 terrain.height_field_raw[i, j] = int(height)


def parkour_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    x_range=[1.8, 1.9],
    y_range=[0.0, 0.1],
    z_range=[-0.2, 0.2],
    stone_len=1.0,
    stone_width=0.6,
    pad_width=0.1,
    pad_height=0.5,
    incline_height=0.1,
    last_incline_height=0.6,
    last_stone_len=1.6,
    pit_depth=[0.5, 1.0],
):
    # 1st dimension: x, 2nd dimension: y
    goals = np.zeros((num_stones + 2, 2))
    terrain.height_field_raw[:] = -round(
        np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale
    )

    mid_y = terrain.length // 2  # length is actually y width
    stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / terrain.horizontal_scale)
    dis_x_min = stone_len + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    dis_z_min = round(z_range[0] / terrain.vertical_scale)
    dis_z_max = round(z_range[1] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_width = round(stone_width / terrain.horizontal_scale)
    last_stone_len = round(last_stone_len / terrain.horizontal_scale)

    incline_height = round(incline_height / terrain.vertical_scale)
    last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
    goals[0] = [platform_len - stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    # dis_z = np.random.randint(dis_z_min, dis_z_max)
    dis_z = 0

    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2 * (left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = (
                np.tile(
                    np.linspace(-last_incline_height, last_incline_height, stone_width),
                    (last_stone_len, 1),
                )
                * pos_neg
            )
            terrain.height_field_raw[
                dis_x - last_stone_len // 2 : dis_x + last_stone_len // 2,
                dis_y - stone_width // 2 : dis_y + stone_width // 2,
            ] = (
                heights.astype(int) + dis_z
            )
        else:
            heights = (
                np.tile(
                    np.linspace(-incline_height, incline_height, stone_width),
                    (stone_len, 1),
                )
                * pos_neg
            )
            terrain.height_field_raw[
                dis_x - stone_len // 2 : dis_x + stone_len // 2,
                dis_y - stone_width // 2 : dis_y + stone_width // 2,
            ] = (
                heights.astype(int) + dis_z
            )

        goals[i + 1] = [dis_x, dis_y]

        left_right_flag = 1 - left_right_flag
    final_dis_x = dis_x + 2 * np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = (
        dis_x + last_stone_len // 2 + round(0.05 // terrain.horizontal_scale)
    )
    terrain.height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def parkour_gap_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_gaps=8,
    gap_size=0.3,
    x_range=[1.6, 2.4],
    y_range=[-1.2, 1.2],
    half_valid_width=[0.6, 1.2],
    gap_depth=-200,
    pad_width=0.1,
    pad_height=0.5,
    flat=False,
):
    goals = np.zeros((num_gaps + 2, 2))
    # terrain.height_field_raw[:] = -200
    # import ipdb; ipdb.set_trace()
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(
        np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale
    )

    # half_gap_width = round(np.random.uniform(0.6, 1.2) / terrain.horizontal_scale)
    half_valid_width = round(
        np.random.uniform(half_valid_width[0], half_valid_width[1])
        / terrain.horizontal_scale
    )
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth

    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if not flat:
            # terrain.height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            # terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2,
            #                          gap_center-half_gap_width:gap_center+half_gap_width] = gap_depth
            terrain.height_field_raw[
                dis_x - gap_size // 2 : dis_x + gap_size // 2, :
            ] = gap_depth

        terrain.height_field_raw[
            last_dis_x:dis_x, : mid_y + rand_y - half_valid_width
        ] = gap_depth
        terrain.height_field_raw[
            last_dis_x:dis_x, mid_y + rand_y + half_valid_width :
        ] = gap_depth

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def parkour_hurdle_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    stone_len=0.01,
    x_range=[1.5, 2.4],
    y_range=[-0.4, 0.4],
    half_valid_width=[0.4, 0.8],
    hurdle_height_range=[0.2, 0.6],
    pad_width=0.1,
    pad_height=0.5,
    flat=False,
):
    goals = np.zeros((num_stones + 2, 2))
    # terrain.height_field_raw[:] = -200

    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(
        np.random.uniform(half_valid_width[0], half_valid_width[1])
        / terrain.horizontal_scale
    )
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)

    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            terrain.height_field_raw[
                dis_x - stone_len // 2 : dis_x + stone_len // 2,
            ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            terrain.height_field_raw[
                dis_x - stone_len // 2 : dis_x + stone_len // 2,
                : mid_y + rand_y - half_valid_width,
            ] = 0
            terrain.height_field_raw[
                dis_x - stone_len // 2 : dis_x + stone_len // 2,
                mid_y + rand_y + half_valid_width :,
            ] = 0
        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def parkour_step_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_stones=8,
    #    x_range=[1.5, 2.4],
    x_range=[0.2, 0.4],
    y_range=[-0.15, 0.15],
    half_valid_width=[0.45, 0.5],
    step_height=0.2,
    pad_width=0.1,
    pad_height=0.5,
):
    goals = np.zeros((num_stones + 2, 2))
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round((x_range[0] + step_height) / terrain.horizontal_scale)
    dis_x_max = round((x_range[1] + step_height) / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(
        np.random.uniform(half_valid_width[0], half_valid_width[1])
        / terrain.horizontal_scale
    )

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    # stone_width = round(stone_width / terrain.horizontal_scale)

    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    goals[0] = [platform_len - round(1 / terrain.horizontal_scale), mid_y]
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height
        terrain.height_field_raw[dis_x : dis_x + rand_x,] = stair_height
        dis_x += rand_x
        terrain.height_field_raw[
            last_dis_x:dis_x, : mid_y + rand_y - half_valid_width
        ] = 0
        terrain.height_field_raw[
            last_dis_x:dis_x, mid_y + rand_y + half_valid_width :
        ] = 0

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale

    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


import numpy as np


def h_hurdle_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_hurdles=4,
    total_goals=None,
    x_range=[1.5, 2.5],
    y_range=[-0.1, 0.1],
    height_range=[0.2, 0.5],
    pad_width=0.1,
    pad_height=0.5,
    progressive_heights=True,
):
    """
    Generate a terrain with H-shaped hurdles loaded from URDF files.
    This function sets up goals and stores hurdle placement information.
    The actual URDF hurdles will be loaded in legged_robot.py

    Parameters:
        terrain: the terrain object
        platform_len (float): length of the starting platform [meters]
        platform_height (float): height of the starting platform [meters]
        num_hurdles (int): number of hurdles to place
        total_goals (int): total number of goals (for consistent array size)
        x_range (list): distance range between hurdles [meters]
        y_range (list): y-axis offset range for hurdle positions [meters]
        height_range (list): hurdle height range [meters] - chooses from [0.2, 0.3, 0.4, 0.5]
        pad_width (float): width of edge padding [meters]
        pad_height (float): height of edge padding [meters]
        progressive_heights (bool): if True, use progressive heights (0.2, 0.3, 0.4, 0.5)
    """
    # Initialize goals array with consistent size
    if total_goals is None:
        total_goals = num_hurdles + 2
    goals = np.zeros((total_goals, 2))

    # Keep terrain flat
    terrain.height_field_raw[:] = 0

    mid_y = terrain.length // 2  # length is actually y width

    # Convert ranges to pixel units
    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    # Initialize hurdle placement info
    hurdle_positions = []  # Will store (x, y, z, height, urdf_file)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]

    # Available hurdle heights (matching URDF files)
    # Define progressive heights if enabled
    if progressive_heights:
        # Fixed sequence: 20cm, 30cm, 40cm, 50cm
        progressive_sequence = [0.2, 0.3, 0.4, 0.5]
    else:
        progressive_sequence = None

    available_heights = [0.2, 0.3, 0.4, 0.5]

    for i in range(num_hurdles):
        # Calculate next hurdle position
        rand_x = (
            np.random.randint(dis_x_min, dis_x_max)
            if dis_x_max > dis_x_min
            else dis_x_min
        )
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max) if dis_y_max > dis_y_min else 0

        # Select hurdle height
        if progressive_heights and i < len(progressive_sequence):
            # Use progressive height sequence
            hurdle_height = progressive_sequence[i]
        else:
            # Filter available heights based on height_range
            valid_heights = [
                h for h in available_heights if height_range[0] <= h <= height_range[1]
            ]
            if not valid_heights:
                valid_heights = available_heights  # fallback to all heights
            hurdle_height = np.random.choice(valid_heights)

        # Determine URDF file based on height
        if hurdle_height <= 0.2:
            urdf_file = "H_hurdel_200.urdf"
        elif hurdle_height <= 0.3:
            urdf_file = "H_hurdel_300.urdf"
        elif hurdle_height <= 0.4:
            urdf_file = "H_hurdel_400.urdf"
        else:
            urdf_file = "H_hurdel_500.urdf"

        # Store hurdle placement info (in meters, world coordinates)
        hurdle_x = dis_x * terrain.horizontal_scale
        hurdle_y = (mid_y + rand_y) * terrain.horizontal_scale
        hurdle_positions.append(
            (
                hurdle_x,
                hurdle_y,
                0.0,  # Hurdles are placed on the ground, height is determined by URDF
                hurdle_height,
                urdf_file,
            )
        )

        # Set goal position (in pixels)
        goals[i + 1] = [dis_x, mid_y + rand_y]

    # Final goal (placed after last hurdle)
    final_rand_x = (
        np.random.randint(dis_x_min, dis_x_max) if dis_x_max > dis_x_min else dis_x_min
    )
    final_dis_x = dis_x + final_rand_x
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[num_hurdles + 1] = [final_dis_x, mid_y]

    # Fill remaining goals (if total_goals > num_hurdles + 2) with the final goal position
    for i in range(num_hurdles + 2, total_goals):
        goals[i] = [final_dis_x, mid_y]

    # Convert goals to meters
    terrain.goals = goals * terrain.horizontal_scale

    # Store hurdle placement info in terrain object
    terrain.hurdle_positions = hurdle_positions

    # Pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


def crawl_through_hurdle_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_hurdles=4,
    total_goals=None,
    x_range=[2.0, 2.5],
    y_range=[0.0, 0.0],  # 居中对齐，确保阻挡机器人路径
    height_range=[0.2, 0.5],
    pad_width=0.1,
    pad_height=0.5,
    progressive_heights=True,
    post_width=0.12,  # 立柱宽度（米）
    post_depth=0.12,  # 立柱深度（米）
    passage_width=0.7,  # 两立柱之间的通道宽度（米）
):
    """
    生成"钻过"模式的栏杆地形：两根立柱 + 虚拟横杆高度约束

    机器人需要保持低姿态钻过两根立柱之间，虚拟横杆用于惩罚身体过高。
    横杆高度递增：200mm → 300mm → 400mm → 500mm

    Parameters:
        terrain: 地形对象
        platform_len (float): 起始平台长度 [米]
        platform_height (float): 起始平台高度 [米]
        num_hurdles (int): 栏杆数量
        total_goals (int): 目标点总数
        x_range (list): 栏杆间距范围 [米]
        y_range (list): Y轴偏移范围 [米]
        height_range (list): 虚拟横杆高度范围 [米]
        pad_width (float): 边缘填充宽度 [米]
        pad_height (float): 边缘填充高度 [米]
        progressive_heights (bool): 是否使用递进高度
        post_width (float): 立柱宽度 [米]
        post_depth (float): 立柱深度 [米]
        passage_width (float): 通道宽度 [米]
    """
    # 初始化目标点数组
    if total_goals is None:
        total_goals = num_hurdles + 2
    goals = np.zeros((total_goals, 2))

    # 转换单位
    platform_len_px = int(platform_len / terrain.horizontal_scale)
    mid_y = terrain.length // 2
    post_width_px = round(post_width / terrain.horizontal_scale)
    post_depth_px = round(post_depth / terrain.horizontal_scale)
    passage_width_px = round(passage_width / terrain.horizontal_scale)

    # 初始化虚拟横杆信息存储
    if not hasattr(terrain, "virtual_crossbars"):
        terrain.virtual_crossbars = []

    # 可用高度列表
    available_heights = [0.2, 0.3, 0.4, 0.5]

    dis_x = platform_len_px
    goals[0] = [platform_len_px - 1, mid_y]

    for i in range(num_hurdles):
        # 栏杆间距
        rand_x = np.random.uniform(x_range[0], x_range[1])
        rand_x_px = int(rand_x / terrain.horizontal_scale)
        dis_x += rand_x_px

        # Y轴偏移（中心对齐，确保阻挡）
        rand_y = np.random.uniform(y_range[0], y_range[1])
        rand_y_px = int(rand_y / terrain.horizontal_scale)

        # 选择高度
        if progressive_heights:
            # 递进高度：按顺序0.2, 0.3, 0.4, 0.5
            hurdle_height = available_heights[i % len(available_heights)]
        else:
            hurdle_height = np.random.uniform(height_range[0], height_range[1])

        hurdle_height_px = round(hurdle_height / terrain.vertical_scale)

        # 计算立柱位置
        center_y = mid_y + rand_y_px
        half_passage = passage_width_px // 2

        # 左立柱
        left_post_y_start = center_y - half_passage - post_width_px
        left_post_y_end = center_y - half_passage

        # 右立柱
        right_post_y_start = center_y + half_passage
        right_post_y_end = center_y + half_passage + post_width_px

        # 立柱X范围
        post_x_start = max(0, dis_x - post_depth_px // 2)
        post_x_end = min(terrain.width, dis_x + post_depth_px // 2)

        # 在高度场中生成左右两根立柱
        if 0 <= left_post_y_start < terrain.length:
            terrain.height_field_raw[
                post_x_start:post_x_end,
                max(0, left_post_y_start) : min(terrain.length, left_post_y_end),
            ] = hurdle_height_px

        if 0 <= right_post_y_start < terrain.length:
            terrain.height_field_raw[
                post_x_start:post_x_end,
                max(0, right_post_y_start) : min(terrain.length, right_post_y_end),
            ] = hurdle_height_px

        # 记录虚拟横杆信息（用于奖励计算）
        crossbar_info = {
            "x": dis_x * terrain.horizontal_scale,  # 世界坐标
            "y": center_y * terrain.horizontal_scale,
            "height": hurdle_height,  # 虚拟横杆高度（米）
            "width": passage_width,  # 通道宽度（米）
            "depth": post_depth,  # 检测范围深度（米）
        }
        terrain.virtual_crossbars.append(crossbar_info)

        # 设置目标点
        goals[i + 1] = [dis_x, center_y]

    # 最后一个目标点
    final_dis_x = dis_x + np.random.randint(
        int(x_range[0] / terrain.horizontal_scale),
        int(x_range[1] / terrain.horizontal_scale),
    )
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - int(0.5 / terrain.horizontal_scale)
    goals[-1] = [final_dis_x, mid_y]

    # 转换为世界坐标
    terrain.goals = goals * terrain.horizontal_scale

    # 边缘填充
    pad_width_px = int(pad_width / terrain.horizontal_scale)
    pad_height_px = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width_px] = pad_height_px
    terrain.height_field_raw[:, -pad_width_px:] = pad_height_px
    terrain.height_field_raw[:pad_width_px, :] = pad_height_px
    terrain.height_field_raw[-pad_width_px:, :] = pad_height_px


def jump_over_wall_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_walls=4,
    total_goals=None,
    x_range=[2.0, 2.5],
    y_range=[-0.3, 0.3],  # 允许一定Y轴偏移
    height_range=[0.2, 0.5],
    pad_width=0.1,
    pad_height=0.5,
    progressive_heights=True,
    wall_depth=0.15,  # 墙体厚度（米）
    wall_width=1.0,  # 墙体宽度（米）
):
    """
    生成"跳跃"模式的墙体地形：实心墙障碍

    机器人需要跳跃越过墙体。墙体高度递增：200mm → 300mm → 400mm → 500mm

    Parameters:
        terrain: 地形对象
        platform_len (float): 起始平台长度 [米]
        platform_height (float): 起始平台高度 [米]
        num_walls (int): 墙体数量
        total_goals (int): 目标点总数
        x_range (list): 墙体间距范围 [米]
        y_range (list): Y轴偏移范围 [米]
        height_range (list): 墙体高度范围 [米]
        pad_width (float): 边缘填充宽度 [米]
        pad_height (float): 边缘填充高度 [米]
        progressive_heights (bool): 是否使用递进高度
        wall_depth (float): 墙体厚度 [米]
        wall_width (float): 墙体宽度 [米]
    """
    # 初始化目标点数组
    if total_goals is None:
        total_goals = num_walls + 2
    goals = np.zeros((total_goals, 2))

    # 转换单位
    platform_len_px = int(platform_len / terrain.horizontal_scale)
    mid_y = terrain.length // 2
    wall_depth_px = max(round(wall_depth / terrain.horizontal_scale), 2)
    wall_width_px = round(wall_width / terrain.horizontal_scale)

    # 可用高度列表
    available_heights = [0.2, 0.3, 0.4, 0.5]

    dis_x = platform_len_px
    goals[0] = [platform_len_px - 1, mid_y]

    for i in range(num_walls):
        # 墙体间距
        rand_x = np.random.uniform(x_range[0], x_range[1])
        rand_x_px = int(rand_x / terrain.horizontal_scale)
        dis_x += rand_x_px

        # Y轴偏移
        rand_y = np.random.uniform(y_range[0], y_range[1])
        rand_y_px = int(rand_y / terrain.horizontal_scale)

        # 选择高度
        if progressive_heights:
            # 递进高度：按顺序0.2, 0.3, 0.4, 0.5
            wall_height = available_heights[i % len(available_heights)]
        else:
            wall_height = np.random.uniform(height_range[0], height_range[1])

        wall_height_px = round(wall_height / terrain.vertical_scale)

        # 计算墙体位置
        center_y = mid_y + rand_y_px
        half_width = wall_width_px // 2

        wall_y_start = max(0, center_y - half_width)
        wall_y_end = min(terrain.length, center_y + half_width)
        wall_x_start = max(0, dis_x - wall_depth_px // 2)
        wall_x_end = min(terrain.width, dis_x + wall_depth_px // 2)

        # 在高度场中生成实心墙
        terrain.height_field_raw[wall_x_start:wall_x_end, wall_y_start:wall_y_end] = (
            wall_height_px
        )

        # 设置目标点
        goals[i + 1] = [dis_x, center_y]

    # 最后一个目标点
    final_dis_x = dis_x + np.random.randint(
        int(x_range[0] / terrain.horizontal_scale),
        int(x_range[1] / terrain.horizontal_scale),
    )
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - int(0.5 / terrain.horizontal_scale)
    goals[-1] = [final_dis_x, mid_y]

    # 转换为世界坐标
    terrain.goals = goals * terrain.horizontal_scale

    # 边缘填充
    pad_width_px = int(pad_width / terrain.horizontal_scale)
    pad_height_px = int(pad_height / terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width_px] = pad_height_px
    terrain.height_field_raw[:, -pad_width_px:] = pad_height_px
    terrain.height_field_raw[:pad_width_px, :] = pad_height_px
    terrain.height_field_raw[-pad_width_px:, :] = pad_height_px


def h_hurdle_geometric_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_hurdles=4,
    total_goals=None,
    x_range=[1.5, 2.5],
    y_range=[0.0, 0.0],  # 门框居中，不偏移（确保阻挡路径）
    height_range=[0.2, 0.5],
    pad_width=0.1,
    pad_height=0.5,
    progressive_heights=True,
    gate_depth=0.10,  # 门框沿前进方向的厚度（增加到10cm，更明显）
    gate_width=0.8,  # 门框总宽度（增加到80cm，给机器人更多通过空间）
    post_thickness=0.08,  # 立柱厚度（米）
):
    """
    生成门型栏杆地形（使用几何体actors，真正镂空的门框）

    注意：这个函数只记录门框的位置和参数信息，不在高度场中生成
    实际的3D门框结构会在 legged_robot.py 的 _create_envs 中使用
    Isaac Gym的几何体API创建，这样才能实现真正的镂空结构

    门型结构：
    - 两根立柱在左右两侧
    - 上方横梁连接两根立柱（悬空，不接触地面）
    - 中间通道完全镂空，机器人可以通过

    Parameters:
        terrain: 地形对象
        platform_len (float): 起始平台长度 [米]
        platform_height (float): 起始平台高度 [米]
        num_hurdles (int): 栏杆数量
        total_goals (int): 目标点总数（保持数组大小一致）
        x_range (list): 栏杆间距范围 [米]
        y_range (list): Y轴偏移范围 [米]
        height_range (list): 栏杆高度范围 [米] - 从[0.2, 0.3, 0.4, 0.5]中选择
        pad_width (float): 边缘填充宽度 [米]
        pad_height (float): 边缘填充高度 [米]
        progressive_heights (bool): 是否使用递进高度（0.2, 0.3, 0.4, 0.5）
        gate_depth (float): 门框沿前进方向的厚度 [米]
        gate_width (float): 门框总宽度（立柱外侧到外侧）[米]
        post_thickness (float): 立柱厚度 [米]
    """
    # 初始化目标点数组
    if total_goals is None:
        total_goals = num_hurdles + 2
    goals = np.zeros((total_goals, 2))

    # 保持地形平坦（门框通过几何体actors创建，不在高度场中生成）
    terrain.height_field_raw[:] = 0

    mid_y = terrain.length // 2  # length实际上是y方向宽度

    # 注意：为了确保每个环境的actor数量一致（Isaac Gym要求）
    # 我们必须在所有环境中创建相同数量的门框

    # 转换范围到像素单位
    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_int = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len_px, :] = platform_height_int

    dis_x = platform_len_px
    goals[0] = [platform_len_px - 1, mid_y]

    # 可用的栏杆高度（匹配URDF文件）
    available_heights = [0.2, 0.3, 0.4, 0.5]
    if progressive_heights:
        progressive_sequence = [0.2, 0.3, 0.4, 0.5]
    else:
        progressive_sequence = None

    # 初始化门框位置信息列表（用于在legged_robot.py中创建几何体actors）
    gate_obstacles = []

    for i in range(num_hurdles):
        # 计算下一个栏杆位置
        rand_x = (
            np.random.randint(dis_x_min, dis_x_max)
            if dis_x_max > dis_x_min
            else dis_x_min
        )
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max) if dis_y_max > dis_y_min else 0

        # 选择栏杆高度
        if progressive_heights and i < len(progressive_sequence):
            hurdle_height = progressive_sequence[i]
        else:
            valid_heights = [
                h for h in available_heights if height_range[0] <= h <= height_range[1]
            ]
            if not valid_heights:
                valid_heights = available_heights
            hurdle_height = np.random.choice(valid_heights)

        # 计算门框在世界坐标系中的位置（米）
        gate_x = dis_x * terrain.horizontal_scale
        gate_y = (mid_y + rand_y) * terrain.horizontal_scale
        gate_z = 0.0  # 地面高度

        # 存储门框信息（将在legged_robot.py中使用这些信息创建几何体actors）
        gate_info = {
            "x": gate_x,
            "y": gate_y,
            "z": gate_z,
            "height": hurdle_height,
            "gate_width": gate_width,
            "gate_depth": gate_depth,
            "post_thickness": post_thickness,
        }
        gate_obstacles.append(gate_info)

        # 设置目标点（在门框中心）
        goals[i + 1] = [dis_x, mid_y + rand_y]

    # 最后一个目标点（放在最后一个栏杆之后）
    final_rand_x = (
        np.random.randint(dis_x_min, dis_x_max) if dis_x_max > dis_x_min else dis_x_min
    )
    final_dis_x = dis_x + final_rand_x
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[num_hurdles + 1] = [final_dis_x, mid_y]

    # 填充剩余目标点
    for i in range(num_hurdles + 2, total_goals):
        goals[i] = [final_dis_x, mid_y]

    # 转换目标点到米
    terrain.goals = goals * terrain.horizontal_scale

    # 存储门框障碍物信息（供legged_robot.py使用）
    terrain.gate_obstacles = gate_obstacles

    # 边缘填充
    pad_width_px = int(pad_width // terrain.horizontal_scale)
    pad_height_px = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width_px] = pad_height_px
    terrain.height_field_raw[:, -pad_width_px:] = pad_height_px
    terrain.height_field_raw[:pad_width_px, :] = pad_height_px
    terrain.height_field_raw[-pad_width_px:, :] = pad_height_px


def h_hurdle_procedural_terrain(
    terrain,
    platform_len=2.5,
    platform_height=0.0,
    num_hurdles=4,
    total_goals=None,
    x_range=[1.5, 2.5],
    y_range=[0.0, 0.0],
    height_range=[0.2, 0.5],
    pad_width=0.1,
    pad_height=0.5,
    progressive_heights=True,
):
    """
    生成完整的H型栏杆地形（程序化创建复合对象，类似URDF结构）

    这个函数创建类似于URDF文件中描述的H型栏杆：
    1. 顶部横杆（水平圆柱，长0.7m，半径0.008m）
    2. 两根立柱（垂直圆柱，根据高度调整长度，半径0.008m）
    3. 两个底座（长方体，0.35×0.03×0.03m）
    4. 底部连接杆（水平圆柱，长0.6m，半径0.005m）

    在Isaac Gym中，这些组件将作为单个复合对象（composite object）创建，
    而不是使用URDF文件，从而提高效率。

    Parameters:
        terrain: 地形对象
        platform_len (float): 起始平台长度 [米]
        platform_height (float): 起始平台高度 [米]
        num_hurdles (int): 栏杆数量
        total_goals (int): 目标点总数
        x_range (list): 栏杆间距范围 [米]
        y_range (list): Y轴偏移范围 [米]
        height_range (list): 栏杆高度范围 [米]
        pad_width (float): 边缘填充宽度 [米]
        pad_height (float): 边缘填充高度 [米]
        progressive_heights (bool): 是否使用递进高度（0.2, 0.3, 0.4, 0.5）
    """
    # 初始化目标点数组
    if total_goals is None:
        total_goals = num_hurdles + 2
    goals = np.zeros((total_goals, 2))

    # 保持地形平坦
    terrain.height_field_raw[:] = 0

    mid_y = terrain.length // 2

    # 转换范围到像素单位
    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_int = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len_px, :] = platform_height_int

    dis_x = platform_len_px
    goals[0] = [platform_len_px - 1, mid_y]

    # 可用的栏杆高度
    available_heights = [0.2, 0.3, 0.4, 0.5]
    if progressive_heights:
        progressive_sequence = [0.2, 0.3, 0.4, 0.5]
    else:
        progressive_sequence = None

    # 初始化H型栏杆位置信息列表
    h_hurdles = []

    # H型栏杆的组件尺寸（基于URDF文件）
    top_bar_radius = 0.008  # 顶部横杆半径
    top_bar_length = 0.7  # 顶部横杆长度
    leg_radius = 0.008  # 立柱半径
    foot_box_size = [0.35, 0.03, 0.03]  # 底座尺寸 [长, 宽, 高]
    foot_connector_radius = 0.005  # 底部连接杆半径
    foot_connector_length = 0.6  # 底部连接杆长度

    for i in range(num_hurdles):
        # 计算下一个栏杆位置
        rand_x = (
            np.random.randint(dis_x_min, dis_x_max)
            if dis_x_max > dis_x_min
            else dis_x_min
        )
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max) if dis_y_max > dis_y_min else 0

        # 选择栏杆高度
        if progressive_heights and i < len(progressive_sequence):
            hurdle_height = progressive_sequence[i]
        else:
            valid_heights = [
                h for h in available_heights if height_range[0] <= h <= height_range[1]
            ]
            if not valid_heights:
                valid_heights = available_heights
            hurdle_height = np.random.choice(valid_heights)

        # 根据不同高度计算立柱长度（基于URDF文件的结构）
        # H_hurdel_200: 立柱长0.2m, H_hurdel_300: 0.3m, H_hurdel_400: 0.4m, H_hurdel_500: 未指定
        if hurdle_height <= 0.2:
            leg_length = 0.2
        elif hurdle_height <= 0.3:
            leg_length = 0.3
        elif hurdle_height <= 0.4:
            leg_length = 0.4
        else:
            leg_length = 0.5  # 假设50cm高度对应0.5m立柱

        # 计算栏杆在世界坐标系中的位置（米）
        hurdle_x = dis_x * terrain.horizontal_scale
        hurdle_y = (mid_y + rand_y) * terrain.horizontal_scale
        hurdle_z = 0.0  # 地面高度

        # 存储H型栏杆信息
        # 每个H型栏杆由多个几何体组成，在legged_robot.py中会被创建为一个复合对象
        hurdle_info = {
            "x": hurdle_x,
            "y": hurdle_y,
            "z": hurdle_z,
            "height": hurdle_height,
            "leg_length": leg_length,
            # 组件尺寸
            "top_bar": {
                "radius": top_bar_radius,
                "length": top_bar_length,
                "color": [1.0, 1.0, 1.0],  # 白色
            },
            "legs": {
                "radius": leg_radius,
                "length": leg_length,
                "spacing": 0.6,  # 两条立柱之间的间距（Y轴）
                "color": [0.2, 0.2, 0.8],  # 蓝色
            },
            "feet": {
                "box_size": foot_box_size,
                "color": [0.5, 0.5, 0.5],  # 灰色
            },
            "foot_connector": {
                "radius": foot_connector_radius,
                "length": foot_connector_length,
                "color": [0.8, 0.1, 0.1],  # 红色
            },
        }
        h_hurdles.append(hurdle_info)

        # 设置目标点（在栏杆中心）
        goals[i + 1] = [dis_x, mid_y + rand_y]

    # 最后一个目标点
    final_rand_x = (
        np.random.randint(dis_x_min, dis_x_max) if dis_x_max > dis_x_min else dis_x_min
    )
    final_dis_x = dis_x + final_rand_x
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[num_hurdles + 1] = [final_dis_x, mid_y]

    # 填充剩余目标点
    for i in range(num_hurdles + 2, total_goals):
        goals[i] = [final_dis_x, mid_y]

    # 转换目标点到米
    terrain.goals = goals * terrain.horizontal_scale

    # 存储H型栏杆信息（供legged_robot.py使用）
    terrain.h_hurdles = h_hurdles

    # 边缘填充
    pad_width_px = int(pad_width // terrain.horizontal_scale)
    pad_height_px = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width_px] = pad_height_px
    terrain.height_field_raw[:, -pad_width_px:] = pad_height_px
    terrain.height_field_raw[:pad_width_px, :] = pad_height_px
    terrain.height_field_raw[-pad_width_px:, :] = pad_height_px


def demo_terrain(terrain):
    goals = np.zeros((8, 2))
    mid_y = terrain.length // 2

    # hurdle
    platform_length = round(2 / terrain.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / terrain.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / terrain.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + hurdle_depth,
        round(mid_y - hurdle_width / 2) : round(mid_y + hurdle_width / 2),
    ] = hurdle_height

    # step up
    platform_length += round(np.random.uniform(1.5, 2.5) / terrain.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / terrain.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[1] = [platform_length + first_step_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + first_step_depth,
        round(mid_y - first_step_width / 2) : round(mid_y + first_step_width / 2),
    ] = first_step_height

    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / terrain.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length + second_step_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + second_step_depth,
        round(mid_y - second_step_width / 2) : round(mid_y + second_step_width / 2),
    ] = second_step_height

    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / terrain.horizontal_scale)

    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / terrain.horizontal_scale)
    goals[3] = [platform_length + third_step_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + third_step_depth,
        round(mid_y - third_step_width / 2) : round(mid_y + third_step_width / 2),
    ] = third_step_height

    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / terrain.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length + forth_step_depth / 2, mid_y]
    terrain.height_field_raw[
        platform_length : platform_length + forth_step_depth,
        round(mid_y - forth_step_width / 2) : round(mid_y + forth_step_width / 2),
    ] = forth_step_height

    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / terrain.horizontal_scale)
    platform_length += gap_size

    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / terrain.horizontal_scale)

    slope_height = round(np.random.uniform(0.15, 0.22) / terrain.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / terrain.horizontal_scale)
    slope_width = round(1.0 / terrain.horizontal_scale)

    platform_height = slope_height + np.random.randint(0, 0.2 / terrain.vertical_scale)

    goals[5] = [platform_length + slope_depth / 2, left_y]
    heights = (
        np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1))
        * 1
    )
    terrain.height_field_raw[
        platform_length : platform_length + slope_depth,
        left_y - slope_width // 2 : left_y + slope_width // 2,
    ] = (
        heights.astype(int) + platform_height
    )

    platform_length += slope_depth + gap_size
    goals[6] = [platform_length + slope_depth / 2, right_y]
    heights = (
        np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1))
        * -1
    )
    terrain.height_field_raw[
        platform_length : platform_length + slope_depth,
        right_y - slope_width // 2 : right_y + slope_width // 2,
    ] = (
        heights.astype(int) + platform_height
    )

    platform_length += slope_depth + gap_size + round(0.4 / terrain.horizontal_scale)
    goals[-1] = [platform_length, left_y]
    terrain.goals = goals * terrain.horizontal_scale


def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


def half_sloped_terrain(terrain, wall_width=4, start2center=0.7, max_height=1):
    wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    height2width_ratio = max_height_int / wall_width_int
    xs = np.arange(slope_start, terrain_length)
    heights = (
        (height2width_ratio * (xs - slope_start))
        .clip(max=max_height_int)
        .astype(np.int16)
    )
    terrain.height_field_raw[slope_start:terrain_length, :] = heights[:, None]
    terrain.slope_vector = np.array(
        [wall_width_int * terrain.horizontal_scale, 0.0, max_height]
    ).astype(np.float32)
    terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')


def half_platform_terrain(terrain, start2center=2, max_height=1):
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    terrain.height_field_raw[:, :] = max_height_int
    terrain.height_field_raw[-slope_start:slope_start, -slope_start:slope_start] = 0
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')


def stepping_stones_terrain(
    terrain, stone_size, stone_distance, max_height, platform_size=1.0, depth=-1
):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """

    def get_rand_dis_int(scale):
        return np.random.randint(
            int(-scale / terrain.horizontal_scale + 1),
            int(scale / terrain.horizontal_scale),
        )

    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height - 1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance - get_rand_dis_int(0.2))
            terrain.height_field_raw[0:stop_x, start_y:stop_y] = np.random.choice(
                height_range
            )
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = (
                    np.random.choice(height_range)
                )
                start_x += stone_size + stone_distance + get_rand_dis_int(0.2)
            start_y += stone_size + stone_distance + get_rand_dis_int(0.2)
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x:stop_x, 0:stop_y] = np.random.choice(
                height_range
            )
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = (
                    np.random.choice(height_range)
                )
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def convert_heightfield_to_trimesh_delatin(
    height_field_raw, horizontal_scale, vertical_scale, max_error=0.01
):
    mesh = Delatin(
        np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error
    )
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles


def convert_heightfield_to_trimesh(
    height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None
):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += (
            hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        )
        move_x[1:num_rows, :] -= (
            hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        )
        move_y[:, : num_cols - 1] += (
            hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        )
        move_y[:, 1:num_cols] -= (
            hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        )
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1]
            > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols]
            > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return vertices, triangles, move_x != 0
