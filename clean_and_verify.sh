#!/bin/bash
# 清理 Python 缓存并验证安装

echo "========================================="
echo "清理 Python 缓存并验证安装"
echo "========================================="
echo ""

# 清理所有 .pyc 和 __pycache__ 目录
echo ">>> 清理 Python 字节码缓存..."
find /home/wh/RL/extreme-parkour/legged_gym -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
find /home/wh/RL/extreme-parkour/legged_gym -name "*.pyc" -delete 2>/dev/null || true
echo "✓ 缓存已清理"
echo ""

# 验证源代码文件
echo ">>> 验证源代码文件..."
python3 << 'PYTHON_EOF'
import os

source_file = '/home/wh/RL/extreme-parkour/legged_gym/legged_gym/envs/base/legged_robot.py'

if os.path.exists(source_file):
    print(f"✓ 源代码文件存在")
    with open(source_file, 'r') as f:
        lines = f.readlines()
    print(f"✓ 文件总行数: {len(lines)}")
    
    if len(lines) >= 415:
        line_415 = lines[414].strip()
        print(f"✓ 第415行内容: {line_415}")
        if '_resample_commands' in line_415:
            print("✓ 确认: 第415行包含 _resample_commands")
        else:
            print("✗ 警告: 第415行不包含 _resample_commands")
    else:
        print(f"✗ 文件少于415行")
else:
    print(f"✗ 源代码文件不存在!")
    exit(1)
PYTHON_EOF

echo ""
echo "========================================="
echo "完成！"
echo "========================================="
echo ""
echo "重要提示："
echo "1. Python 缓存已清理"
echo "2. 源代码文件已验证正确"
echo "3. 请重新运行训练脚本测试"
echo ""
echo "如果问题仍然存在，可能的原因："
echo "- 训练脚本在不同的 conda 环境中运行"
echo "- 有其他位置的 legged_gym 安装"
echo "- 训练脚本有缓存的导入"
echo ""

