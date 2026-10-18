# 把这个项目上传到 GitHub

上传用这一份压缩包，不要把整个 `D:\1016homework\orangepiyolo` 传上去。那个目录里还有 68 MB 数据集和已经停掉的手语数字训练，GitHub 仓库不需要它们。

压缩包路径：

`D:\1016homework\orangepiyolo\orangepiyolo_github.zip`

大约 25 MB，31 个文件。最大的单个文件是 ONNX，12.8 MB，没有超过 GitHub 的 100 MB 限制。这台电脑已安装 Git，没有安装 `gh`。下面用浏览器建仓库，再用 Git 推送。

## 1. 在网页上建一个空仓库

1. 打开 https://github.com/new ，登录你的 GitHub 账号。
2. **Repository name** 填 `orangepiyolo`。
3. 选 **Public**。
4. 不要勾选 **Add a README file**，也不要加 `.gitignore` 和 license。压缩包里已经有 `README.md` 和 `.gitignore`。
5. 点 **Create repository**。
6. 创建完成后，页面上会有一行地址，形如：

`https://github.com/你的用户名/orangepiyolo.git`

把这行地址复制下来。下面命令里的地址换成你自己的。

## 2. 解压到一个新目录

在 PowerShell 里执行：

```powershell
Expand-Archive -Path D:\1016homework\orangepiyolo\orangepiyolo_github.zip -DestinationPath D:\1016homework\orangepiyolo-upload -Force
cd D:\1016homework\orangepiyolo-upload\orangepiyolo
dir
```

`dir` 里应能看到 `README.md`、`报告.md`、`记忆.md`、`gesture_infer.py` 和 `weights`。这个新目录和原来的工程分开，原文件不会被删。

## 3. 提交并推送

仍在 `D:\1016homework\orangepiyolo-upload\orangepiyolo` 里执行：

```powershell
git init
git add .
git commit -m "Add five-class gesture detection project."
git branch -M main
git remote add origin https://github.com/你的用户名/orangepiyolo.git
git push -u origin main
```

`git push` 时会弹出 GitHub 登录窗口。用浏览器登录你的账号并授权。密码框如果要 token，到 GitHub 网页 **Settings → Developer settings → Personal access tokens** 生成一个带 `repo` 权限的 token，粘贴进去。token 只显示一次，不要写进任何文件。

推送结束后，打开：

`https://github.com/你的用户名/orangepiyolo`

首页应显示 README 里的项目说明。点进 `报告.md` 可以看训练结果和板端 ATC 记录。

## 4. 以后改了代码再传

只在解压出来的这个目录里改，然后：

```powershell
cd D:\1016homework\orangepiyolo-upload\orangepiyolo
git add .
git commit -m "Describe what changed."
git push
```

如果改的是 `D:\1016homework\orangepiyolo` 里的原工程，需要重新打包，再把新文件复制进 `orangepiyolo-upload\orangepiyolo`，然后执行上面的三条。
