# GitHub 上传与提交身份

项目公开仓库：[M1sumiUika/mace-crystal-equilibrium](https://github.com/M1sumiUika/mace-crystal-equilibrium)。该公开仓库由用户创建；上传时保留用户的网页初始化提交及 README 描述，合入完整项目和实际结果。

## 推送账号与提交作者

GitHub 登录凭据决定有没有权限推送，Git 的 `user.name` 和 `user.email` 决定提交中记录的作者。VS Code 登录账号不会自动覆盖电脑已有的 Git 作者配置；GitHub 根据提交邮箱关联贡献者账号。见 [GitHub 官方说明](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address)。

本仓库使用 `M1sumiUika` 及其 GitHub 隐私邮箱。只设置仓库级配置，不改变其他项目的全局配置：

```powershell
git config --local user.name M1sumiUika
git config --local user.email 274427690+M1sumiUika@users.noreply.github.com
git config --local credential.https://github.com.username M1sumiUika
```

最初的 3 条本地项目提交曾沿用旧全局作者。经用户确认，更正其 author/committer 身份，保留原始文件内容、提交说明和实际提交时间。旧历史备份仅存在本地被忽略的 `.cache/identity-fix/`，不上传。之后合入用户在公开仓库创建的初始提交，通过普通推送发布。

## 日后更新

在项目根目录打开 PowerShell，先检查变化，再提交与推送：

```powershell
git status --short
git diff
git add .
git commit -m "Describe the actual change"
git push
```

当前 `origin` 为 `https://github.com/M1sumiUika/mace-crystal-equilibrium.git`。若在其他电脑克隆，需自行完成 GitHub 登录和仓库级作者配置；不要把令牌放进 URL、项目或聊天。使用 `git log -3 --format="%h %an <%ae>"` 可检查本地提交作者，在 GitHub 的提交页面可核对账号关联。

模型权重、下载缓存、虚拟环境、认证凭据和原有空 `1.py` 均不提交。公开上传前检查：

```powershell
git ls-files ".venv/*" ".cache/*" "*.model" "*.pt" "*.pth" ".env*"
.\.venv\Scripts\python.exe scripts/audit_delivery.py
```

首条命令应无输出。原始科学数据、模型身份、文献来源、图表和实际运行记录可公开复核；模型与第三方数据保留各自许可证。

本机曾因 Git 直连 GitHub 443 超时而使用系统已有代理 `127.0.0.1:7897`，仅在本地 `.git/config` 中设置。若关闭或更换代理，可运行 `git config --local --unset http.proxy`，再按当前网络配置连接。代理配置不上传。
