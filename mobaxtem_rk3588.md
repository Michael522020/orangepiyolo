# 用 MobaXterm 把压缩包传到 RK3588 开发板

电脑上的压缩包仍是这一份：

`D:\1016homework\orangepiyolo\orange_pi_deploy.zip`

传到 RK3588 的操作和传到 Orange Pi AI Pro 相同：MobaXterm 的 SSH 加上左边的 SFTP，把 zip 拖进板子家目录再解压。

这块板是瑞芯微 RK3588，NPU 走 RKNN。压缩包里的 `atc_convert.sh`、`aipp.cfg`、`aipp_rgb.cfg`、`gesture_infer.py` 是昇腾 310B4 用的，在 RK3588 上不要执行。解压后实际要用的是：

`weights/gesture_yolov8n.onnx`

输入节点名 `images`，形状 `[1, 3, 640, 640]`，输出 `output0`，形状 `[1, 9, 8400]`。类别顺序固定为：

`five, four, one, three, two`

`.rknn` 要在 x86 的 Ubuntu 电脑上用 RKNN-Toolkit2 转换，目标平台填 `rk3588`。开发板上只跑对应版本的 RKNN Lite，不要在板上 `pip install` 一套新环境。

## 1. 先让电脑和开发板在同一个网络

网线或同一个 Wi-Fi 都可以。开发板接上电，等系统起来。

在开发板自己的终端里查看 IP：

```bash
hostname -I
```

记下第一段地址，例如 `192.168.1.60`。下面用这个地址举例，换成你自己的。

常见账号（按你刷的镜像选一个）：

| 镜像 | 用户名 | 密码 |
| --- | --- | --- |
| Orange Pi 5 / 5 Plus（RK3588） | `orangepi` | `orangepi` |
| Radxa ROCK 5 | `radxa` | `radxa` |
| 许多厂商出厂镜像 | `root` 或 `linaro` | 看该镜像说明 |

## 2. 用 MobaXterm 连上 SSH

1. 打开 MobaXterm。
2. 左上角 **Session** → **SSH**。
3. **Remote host** 填开发板 IP，例如 `192.168.1.60`。
4. 勾选 **Specify username**，填上表里的用户名。
5. **Port** 保持 `22`。
6. 点 **OK**。第一次连接弹出主机指纹时选 **Accept**。
7. 输入密码。密码输入时屏幕不显示字符，这是正常的。

连上以后，窗口左边是 SFTP 文件浏览器，右边是开发板终端。

如果左边没有文件栏：点工具栏 **Sftp browser**，或菜单 **Settings → SSH → SSH browser**，勾选 **Enable SSH browser**，断开后重新连接。

## 3. 在开发板上建目录

用户名是 `orangepi` 时，在右边终端执行：

```bash
mkdir -p ~/gesture
cd ~/gesture
pwd
```

`pwd` 应打印 `/home/orangepi/gesture`。用户名不同时，路径是 `/home/<用户名>/gesture`；`root` 登录则是 `/root/gesture`。

左边 SFTP 点刷新，进入这个 `gesture` 目录。

## 4. 上传压缩包

左边 SFTP 停在刚才的 `gesture` 目录。

把电脑上的这个文件拖进左边窗口：

`D:\1016homework\orangepiyolo\orange_pi_deploy.zip`

拖进去之后等进度条走完。也可以在左边空白处右键 **Upload to current folder**，选中同一个 zip。

上传完成后，在右边终端确认：

```bash
cd ~/gesture
ls -lh orange_pi_deploy.zip
```

大小应接近 11M。若是 0 字节，删掉后重新拖一次：

```bash
rm -f orange_pi_deploy.zip
```

## 5. 解压并只核对 ONNX

```bash
cd ~/gesture
unzip -o orange_pi_deploy.zip
ls -lh weights/gesture_yolov8n.onnx
```

ONNX 大小应是 12,777,110 字节。

如果提示没有 `unzip`，用板上的 Python：

```bash
python3 -m zipfile -e orange_pi_deploy.zip .
```

到这里传输结束。下面这些命令不要在 RK3588 上运行：

```bash
bash atc_convert.sh none weights/gesture_yolov8n.onnx images
python3 gesture_infer.py
```

`atc` 和 `ais_bench` 只存在于 Orange Pi AI Pro 的昇腾环境里。RK3588 上没有这两样，跑了会直接报找不到命令或找不到模块。

## 6. RK3588 上真正要做的转换

在一台 x86 Ubuntu 电脑上安装与板端 Runtime 主版本一致的 RKNN-Toolkit2，然后把这个 ONNX 转成 rknn。转换配置保持和导出一致：

- 目标平台 `rk3588`
- 输入名 `images`
- 输入形状 `[1, 3, 640, 640]`
- 输入是 RGB，数值范围 0 到 1（对应训练时的 `/255`）
- 不要在转换里打开动态形状

转好的 `gesture_yolov8n.rknn` 再用同一次 MobaXterm 会话拖进 `~/gesture`。板端用 RKNN Lite 加载这个 rknn 做推理。后处理仍是 YOLOv8 检测头：8400 个框，5 个类别，没有 objectness，类别顺序是 `five, four, one, three, two`。

电脑上的 `D:\1016homework\orangepiyolo` 原文件保持不动。
