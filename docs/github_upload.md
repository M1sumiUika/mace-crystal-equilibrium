# 上传到 GitHub 私有仓库

本项目只应把代码、文档、必要的实际结果与文献数据提交到 Git。模型权重、下载缓存、`.venv`、密钥、令牌和私密文件由 `.gitignore` 排除。当前工作区原有的空 `1.py` 被保留并忽略。

本次已通过本机 Git Credential Manager 验证用户指定的 `M1sumiUika` 账号，并实际创建 [mace-crystal-equilibrium 私有仓库](https://github.com/M1sumiUika/mace-crystal-equilibrium)；API 返回 `private: true`。认证凭据未写入项目或远端 URL。真实提交记录以 `git log` 和远端 `main` 为准，不补造历史提交、经历或日期。

日后在本工作区更新：`git add .`、`git commit -m "Describe the actual change"`、`git push`。下面保留从其他工作区新建私有仓库的通用操作说明；已存在的仓库无需重复创建。

## 使用 GitHub CLI

从 [GitHub CLI 官方页面](https://cli.github.com/) 安装 `gh`，在项目根目录打开 PowerShell。登录操作在用户自己的终端中进行，不把访问令牌写入项目或聊天。

```powershell
gh auth login
gh auth status
git status --short
git log -1 --oneline
```

上传前确认 `git status` 和已跟踪文件符合预期；下列命令检查常见不应提交的文件，如出现输出应先检查：

```powershell
git ls-files ".venv/*" ".cache/*" "*.model" "*.pt" "*.pth" ".env*"
```

若项目尚未初始化 Git，在确认文件后运行 `git init -b main`、`git add .` 和 `git commit -m "Add reproducible MACE crystal equilibrium study"`；已有仓库和真实提交无需重复初始化。提交需要用户自己有效的 Git 身份，不使用虚构姓名或邮箱。

创建**私有**仓库并推送已有真实提交，参数含义见 [GitHub CLI 官方手册](https://cli.github.com/manual/gh_repo_create)：

```powershell
gh repo create mace-crystal-equilibrium --private --source . --remote origin --push --description "Static-lattice Si and Al equilibrium properties with CUDA MACE and numerical validation"
```

如果同名仓库或 `origin` 已存在，先检查 `git remote -v` 与 GitHub 页面，确认目标后再连接；不要覆盖未知远端。日后更新通常使用 `git add`、`git commit`、`git push`。首次上传后查看 GitHub 的文件页面及 Actions 状态，远端 CPU 测试通过也不能代替本机 GPU 验证。

## 使用网页创建仓库

1. 登录 GitHub，进入 [新建仓库页面](https://github.com/new)。
2. 仓库名填 `mace-crystal-equilibrium`，可见性选择 **Private**。
3. 若准备推送已有本地提交，不要在网页额外生成 README、`.gitignore` 或许可证，以免产生不必要的提交冲突。
4. 创建后复制网页显示的 HTTPS 或 SSH 仓库地址；在本地检查 `git remote -v`，没有 `origin` 时再添加。

```powershell
# 把占位符替换为刚创建的真实地址。
git remote add origin https://github.com/YOUR_ACCOUNT/mace-crystal-equilibrium.git
git push -u origin main
```

HTTPS 认证按 GitHub 或 Git Credential Manager 的交互指引进行，不把令牌放入 URL。GitHub Desktop 也可选择已有本地仓库后发布，并勾选保留私有。最终是否私有，以远端仓库实际设置为准。

公开发布前请再次阅读 [模型记录](model_card.md) 与 [实际报告](report_zh.md)，确保方法、模型来源、验证问题和未完成项目如实可见。项目文件中的运行时间来自真实执行记录；不要为了展示修改历史记录。
