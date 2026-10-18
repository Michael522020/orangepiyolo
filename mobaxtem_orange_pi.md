# 用 MobaXterm 把压缩包传到 Orange Pi AI Pro 8T

压缩包在电脑上的位置：

`D:\1016homework\orangepiyolo\orange_pi_deploy.zip`

大约 11 MB。里面是：

- `gesture_infer.py`
- `atc_convert.sh`
- `aipp.cfg`
- `aipp_rgb.cfg`
- `weights/gesture_yolov8n.onnx`

电脑上的这些原文件不要删，压缩包只是拷过去用的副本。

开发板系统是只读环境约束：传上去之后不要 `pip`、`conda`、`apt`，也不要新建虚拟环境。

## 1. 先让电脑和开发板在同一个网络

这块板当前已经连上 Wi-Fi。终端提示符是：

`(base) HwHiAiUser@orangepiaipro:~$`

所以登录信息是：

- 用户名 `HwHiAiUser`
- 主机名 `orangepiaipro`
- 家目录 `/home/HwHiAiUser`

查地址用 `ip a`。`ip -a` 会直接打印用法说明，不会给出地址。

`ip a` 里各网卡的状态：

| 网卡 | 状态 | 地址 | 用途 |
| --- | --- | --- | --- |
| `lo` | UP | `127.0.0.1` | 本机回环，不能用来 SSH |
| `docker0` | DOWN | `172.17.0.1/16` | Docker 网桥，不能用来 SSH |
| `bond0` | DOWN | 无 | 未启用 |
| `can2` | DOWN | 无 | CAN 口，不是网口 |
| `eth0` | DOWN，`NO-CARRIER` | 无 | 网线没插好或没插，现在走不了网线 |
| `wlan0` | UP | `172.26.49.9/16` | 当前可用地址 |

MobaXterm 和 `scp` 都填 `172.26.49.9`。电脑要能访问 `172.26.49.0/16` 这个网段，一般就是连开发板正在用的那个 Wi-Fi。

`wlan0` 的地址是 DHCP 分配的，租约大约还有 42471 秒。如果以后连不上，在板子上再执行一次 `ip a`，看 `wlan0` 的 `inet` 有没有变。

提示符前面的 `(base)` 表示板上原来的 conda base 已经激活。继续用这个环境，不要 `conda create`，也不要 `pip`、`apt`。

## 2. 用 MobaXterm 连上 SSH

1. 打开 MobaXterm。
2. 左上角 **Session** → **SSH**。
3. **Remote host** 填 `172.26.49.9`。
4. 勾选 **Specify username**，填 `HwHiAiUser`。
5. **Port** 保持 `22`。
6. 点 **OK**。第一次连接弹出主机指纹时选 **Accept**。
7. 输入这块板的密码。密码输入时屏幕不显示字符，这是正常的。

也可以在电脑 PowerShell 里直接传：

```powershell
ssh HwHiAiUser@172.26.49.9 "mkdir -p ~/gesture"
scp D:\1016homework\orangepiyolo\orange_pi_deploy.zip HwHiAiUser@172.26.49.9:~/gesture/
```

连上以后，窗口左边会出现一个文件浏览器，那就是 SFTP。右边是开发板的终端。

如果左边没有文件栏：点工具栏 **Sftp browser**，或菜单 **Settings → SSH → SSH browser**，勾选 **Enable SSH browser**，断开后重新连接。

## 3. 在开发板上建目录

在右边终端执行：

```bash
mkdir -p ~/gesture
cd ~/gesture
pwd
```

`pwd` 应打印 `/home/HwHiAiUser/gesture`。左边 SFTP 点刷新，进入这个 `gesture` 目录。

## 4. 上传压缩包

左边 SFTP 停在 `/home/HwHiAiUser/gesture`。

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

## 5. 解压

```bash
cd ~/gesture
unzip -o orange_pi_deploy.zip
find . -type f -printf '%s %p\n'
```

应能看到：

- `gesture_infer.py`
- `atc_convert.sh`
- `aipp.cfg`
- `aipp_rgb.cfg`
- `weights/gesture_yolov8n.onnx`（大约 12,777,110 字节）

如果提示没有 `unzip`，不要 apt 安装，用板上的 Python：

```bash
python3 -m zipfile -e orange_pi_deploy.zip .
```

给转换脚本执行权限：

```bash
sed -i 's/\r$//' atc_convert.sh
chmod +x atc_convert.sh
```

## 6. 转 OM 再开摄像头

先加载板上已经装好的 CANN，不要重新安装：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
which atc
```

`which atc` 要打印出 `atc` 的路径。如果这个 `set_env.sh` 不存在，在板上找现成的：

```bash
find /usr/local/Ascend -name set_env.sh 2>/dev/null
```

找到后 `source` 那一条，仍然不要重装 CANN。

然后：

```bash
cd ~/gesture
bash atc_convert.sh none weights/gesture_yolov8n.onnx images
```

成功后当前目录会多出 `gesture_yolov8n.om`。`gesture_infer.py` 顶部已经是 `input_mode = float32`、`model_path = gesture_yolov8n.om`，不用改。

USB 摄像头插上后：

```bash
python3 gesture_infer.py
```

窗口里按 `q` 或 `Esc` 退出。

USB 摄像头这条路径确认有框之后，再转更快的 RGB AIPP 模型：

```bash
bash atc_convert.sh aipp_rgb weights/gesture_yolov8n.onnx images
```

然后把 `gesture_infer.py` 里 `CONFIG` 改成：

- `model_path`: `gesture_yolov8n_aipp_rgb.om`
- `input_mode`: `aipp_rgb`

两个字段要一起改。`float32` 的 OM 和 `aipp_rgb` 的 OM 不能混用。
