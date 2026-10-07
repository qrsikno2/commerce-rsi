# 采样程序隔离：实测和边界

日期：2026-10-07；服务器 kernel `5.15.0-86-generic`，Linux x86_64，容器内 root。

初始探测：`unshare --user --map-root-user true` 返回 EPERM；`docker` / `bwrap` 不在 PATH。当前进程已经有一层宿主 seccomp filter。Landlock ABI 探测 `syscall(444, NULL, 0, LANDLOCK_CREATE_RULESET_VERSION)` 返回 **1**，可叠加限制。没有修改驱动、安装 Docker、启动守护进程或要求宿主特权。

## 已实现路径

`src/commerce_rsi/openrsi_adapter/sandbox.py` 提供 `run_sampler(...)` 和固定训练器可调用的 `SandboxedSampler.sample_negatives(...)`。

1. 新 Python interpreter 使用 `-P -s -S`（safe path、无 user site、无 site.py），清理环境，仅保留 locale/hash seed；通过 `exec` 建立新的地址空间，没有 fork 后直接执行用户函数。这里不使用 `-I`，因为它会忽略 `PYTHONHASHSEED`、破坏 hash 可复现性。
2. 父子间只用 JSON 管道传送显式白名单；`close_fds=True` 关闭父进程文件描述符。公共依赖 `math`、`random`、`json` 和 sandbox 控制所需标准库在限制前加载。
3. worker 在生成代码的模块级语句、compile/exec 之前设置 wall/CPU/address-space/FD/process/file-size/core limits。root 启动时清 supplementary groups 并 drop 到 uid/gid 65534；设置不可 dump 与 `no_new_privs`。
4. 安装空 allow-rules 的 Landlock ruleset：所有 ABI 1 支持的文件读取、列目录、写入、执行和创建删除访问均禁止。
5. 叠加 seccomp BPF syscall allowlist，只允许 Python 的基本内存、信号、时间与已有管道读写。没有 open/openat、socket、fork/clone/exec、kill、ptrace、process_vm、ioctl、io_uring、mount、getrandom 或提权操作。拒绝其他 architecture；x32 syscall 不在 allowlist。ABI 1 不覆盖的 truncate 等文件操作由 seccomp 拒绝。
6. 然后才 exec 候选代码并调用 `sample_negatives`。结果 JSON 由可信父进程解析；业务训练器独立检查形状、商品存在、正例/同文本商品排除、无重复。worker 无法传回 Python 对象或修改训练器权重、评分函数。

任何隔离安装错误返回 `isolation_unavailable`，绝不降级为普通子进程。语法/接口/程序错误、超时、OOM/worker exit、过量输出均作为失败，没有随机策略兜底。父进程使用非阻塞读写防止管道死锁，并监控 wall time 与 stdout+stderr 总量。

默认 `run_sampler` 为 wall 5 秒、CPU 2 秒、256 MiB address space、16 MiB 输入、1 MiB 输出、64 KiB 源码。trainer bridge 使用 512 MiB/32 MiB 输入；仍是 CPU 2 秒、wall 5 秒。每次训练采样调用都建立新 worker，避免跨 batch 持久化；其开销计入业务执行成本。

## 公开给采样器的数据

- batch：`qid/query/item_id/known_positive_ids`。已知正例由可信训练器的 `known_positive_item_ids` 转换，并与显式传入的已知正例合并。原始评论、评分、user ID、outer/inner split 标签和隐藏答案不传入。
- candidate_view：全部固定候选库 `item_ids`，对齐 `texts`（前 256 字符）和 `categories`（前 128 字符）。该截断只限制采样器传输，不修改固定业务模型的输入协议；每个候选策略拿到相同视图。
- model_view：有限数值 `step/total_steps/progress` 中实际提供的字段；当前训练器提供 step/total_steps。没有模型对象、路径、embedding tensor 或任意 callback。
- rng：父进程按记录的训练随机状态派生 seed，再创建标准库 `random.Random`；同时固定 hash seed 与全局 random seed。通过 seed 的正常采样逻辑可重放，故意使用时钟的候选仍可能不确定，需要配对重复实验识别。

这意味着初版可学习文字/品类/查询/阶段相关采样，但没有动态模型 embedding hard-mining 能力。后续需要通过额外只读 JSON snapshot 显式增加，同时复查安全与公平预算。

## 真实检查

`tests/test_sandbox.py` 使用手写可信攻击 fixture，不是模型生成的未知程序。检查：

- 正常函数和预加载 math/random 可运行，给定 seed 可重放。
- 读取私有文件、覆盖 trainer、truncate、unlink 均失败，宿主文件内容不变；额外用 `/tmp` 下所有用户可读写的文件确认拒绝不是仅依赖 root 文件权限。
- ctypes 直接调用 socket/fork/kill syscall 返回 EPERM，subprocess 执行失败。
- 无限循环按 wall limit 终止，输出洪泛触发输出限制。
- 失败/非法输出不会换成随机采样；secret 环境和刻意设置 inheritable 的父 FD 不泄漏。
- trainer bridge 丢弃禁用字段、截断公开商品文本，结果由宿主 validator 检查。

与真实上游接口和 Parquet 测试一起运行得到 **24 passed**（2026-10-07）。这是实际启用了 Landlock/seccomp 的内核执行测试，不是 AST 过滤或模拟 sandbox。

## 适用范围

此实现针对当前 Linux x86_64 和已有内核特性，未声称是容器/虚拟机边界或抵御未知内核漏洞。没有 network/PID namespace，但网络与进程控制 syscall 被拒绝、文件访问被禁止、FD/env/address-space 隔离并实测。sampler 可查看它自己的内存、输入和预加载标准库；其中必须始终不含隐藏答案。它可能自行打印或主动退出来制造格式错误，可信评分器只能接受合法采样输出，不能信任程序打印的指标。

该 worker 不能加载 numpy/torch、联网、GPU 访问或读取公共数据文件。未来扩大这些权限需要重新设计并实测，不得直接在 allowlist 中随意开放。CPU/wall/memory 限制针对本 worker；整个代际实验仍须由可信控制器执行独立总预算管理。
