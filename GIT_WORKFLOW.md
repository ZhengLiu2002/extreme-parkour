# Git 日常开发工作流程与常用命令指南

本文档旨在为开发人员提供一个清晰、高效的 Git 工作流程，并详细解释日常开发中常用的 Git 命令。

## 1. Git 核心概念

理解 Git 的四个核心区域是掌握其工作流程的基础：

1.  **工作区 (Workspace)**: 你在电脑上能看到的、正在编辑的项目目录。
2.  **暂存区 (Staging Area / Index)**: 一个临时的文件区域，用于存放你希望在下一次提交中包含的变更。`git add` 命令就是将工作区的变更添加到暂存区。
3.  **本地仓库 (Local Repository)**: 你的项目在你电脑上的完整版本历史记录。`git commit` 命令将暂存区的变更永久保存到本地仓库。
4.  **远程仓库 (Remote Repository)**: 托管在网络服务器上的项目仓库（例如 GitHub、GitLab），用于团队协作和代码备份。`git push` 和 `git pull` 用于本地与远程仓库的数据同步。

---

## 2. 日常开发标准工作流程

这是一个推荐的日常开发流程，适用于个人和团队项目。

### 第一步：开始新任务前，同步远程仓库

在开始任何新的编码任务之前，确保你的本地代码是最新的，以避免不必要的合并冲突。

```bash
# 切换到主分支 (通常是 main 或 master)
git checkout main

# 从远程仓库拉取最新的变更
git pull origin main
```

### 第二步：创建并切换到新分支

为每个新功能、错误修复或任务创建一个独立的分支。这能保持主分支的整洁和稳定，并使代码审查（Code Review）更容易。

```bash
# 创建一个名为 "feature/user-login" 的新分支并立即切换过去
git checkout -b feature/user-login
# 分支命名建议:
# - feature/功能名 (例如: feature/user-login)
# - fix/问题名 (例如: fix/login-bug)
# - chore/琐事名 (例如: chore/update-readme)
```

### 第三步：编码与修改

在你的新分支上进行代码的编写和修改。

### 第四步：暂存和提交变更

在开发过程中，可以分阶段、有逻辑地提交你的代码。一次提交应该只包含一个独立的、完整的功能或修复。

```bash
# 1. 查看当前项目状态（哪些文件被修改了）
git status

# 2. 将你希望提交的变更添加到暂存区
# 添加单个文件
git add path/to/your/file.py
# 添加整个目录
git add path/to/your/directory/
# 添加所有已修改和新创建的文件（常用）
git add .

# 3. 提交暂存区的变更到本地仓库
git commit -m "feat: 实现用户登录功能"
# 提交信息 (commit message) 编写建议:
# - 使用祈使句 (例如: "Add" 而不是 "Added" 或 "Adds")
# - 格式: <类型>: <主题> (例如: feat: 用户登录, fix: 修复密码验证错误)
# - 类型可以是: feat(新功能), fix(修复), docs(文档), style(格式), refactor(重构), test(测试), chore(构建过程或辅助工具变动)
```

### 第五步：推送到远程仓库

当你的功能开发完成或需要他人审查时，将你的分支推送到远程仓库。

```bash
# 将当前分支推送到远程仓库，并设置上游跟踪
git push -u origin feature/user-login
# 第一次推送时使用 `-u` 参数，之后直接使用 `git push` 即可
```

### 第六步：创建合并请求 (Pull Request)

在 GitHub 或 GitLab 等平台上，发起一个从你的功能分支到主分支 (`main`) 的合并请求（Pull Request 或 Merge Request）。这是邀请团队成员审查你的代码、讨论并最终合并代码的正式方式。

### 第七步：合并与清理

代码审查通过后，将你的分支合并到 `main` 分支（通常在远程仓库的网页上完成）。合并后，你可以选择删除远程和本地的功能分支。

```bash
# 1. 切换回主分支
git checkout main

# 2. 拉取最新的远程代码（包含刚刚合并的变更）
git pull origin main

# 3. 删除本地已合并的分支
git branch -d feature/user-login

# 4. (可选) 删除远程分支
git push origin --delete feature/user-login
```

---

## 3. 常用 Git 命令详解

### 查看状态与历史
*   `git status`: 显示工作区和暂存区的状态，这是最常用的命令之一。
*   `git log`: 显示从近到远的提交历史。
    *   `git log --oneline --graph --decorate`: 以更简洁的图形化方式显示历史。
*   `git diff`: 查看工作区与暂存区之间的差异。
    *   `git diff --staged`: 查看暂存区与上一次提交之间的差异。
    *   `git diff <commit1> <commit2>`: 查看两次提交之间的差异。

### 分支管理
*   `git branch`: 列出所有本地分支。
*   `git branch <branch-name>`: 创建一个新分支。
*   `git checkout <branch-name>`: 切换到指定分支。
*   `git merge <branch-name>`: 将指定分支的变更合并到当前分支。

### 撤销与修改
*   `git checkout -- <file>`: 丢弃工作区中对某个文件的修改（危险操作，无法恢复）。
*   `git reset HEAD <file>`: 将文件从暂存区移回工作区（取消 `git add`）。
*   `git commit --amend`: 修改最后一次提交（不要对已经推送到远程的提交使用此命令）。

### 远程协作
*   `git clone <repository-url>`: 克隆一个远程仓库到本地。
*   `git remote -v`: 查看所有远程仓库的地址。
*   `git fetch`: 从远程仓库下载最新的对象，但**不**合并到你当前的工作分支。

---

## 4. 解决合并冲突

当多个人修改了同一个文件的同一部分时，合并就会产生冲突。

1.  运行 `git pull` 或 `git merge` 时，Git 会提示冲突。
2.  打开有冲突的文件，你会看到类似下面的标记：

    ```
    <<<<<<< HEAD
    // 这是你当前分支的代码
    =======
    // 这是你正在合并的分支的代码
    >>>>>>> other-branch-name
    ```
3.  手动编辑文件，解决冲突，决定保留哪部分代码或进行整合。
4.  删除 `<<<<<<<`, `=======`, `>>>>>>>` 这些标记。
5.  保存文件后，使用 `git add <conflicted-file>` 将其标记为已解决。
6.  最后，运行 `git commit` 来完成合并。
