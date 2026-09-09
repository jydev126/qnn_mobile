# 最小 C++ runtime 源码导读

源码只有 [qnn_context_runner.cpp](qnn_context_runner.cpp) 一个翻译单元。先从底部 `main()` 看六段流程，再回头读它调用的 helper。不要一开始从所有 QNN typedef 读起。

## 先回答：这个程序的输入是什么

它读取已经 build 好的 context binary 和 native raw tensor 文件。它不读取 DLC、不构建 RF-DETR 算法、不做 resize/画框，也不调用 `qnn-net-run`。DLC → context 的编译边界由前一阶段的 Qualcomm 工具验证。

```text
qnn-context-runner
  --backend /设备路径/libQnnHtp.so
  --system /设备路径/libQnnSystem.so
  --context /设备路径/rf_detr.bin
  --input image=/设备路径/image.raw
  --output-dir /设备路径/新的结果目录
  --runs 6
```

`--input` 可重复，但当前只支持一个 graph。tensor 名字、顺序、shape 和 dtype 都来自 binary metadata，没有在 C++ 中写死 `image`、`boxes` 或 `300`。float32/int32、dense、静态 shape 是明确的支持范围，不做自动量化或反量化。

## 对照 main 的六段流程

| main 中的阶段 | 核心调用 | 结果是什么 |
| --- | --- | --- |
| 1. load / initialize | dlopen、getProviders、logCreate、backendCreate、deviceCreate | API 函数表、backend/device 句柄 |
| 2. file / metadata | readFile、systemContextCreate、systemContextGetMetaData | binary 字节、graph/tensor 的描述 |
| 3. restore | contextCreateFromBinary、graphRetrieve | 可以执行的 context 与 graph 句柄 |
| 4. buffers | TensorBuffers、readInputs | 有正确名字、id、shape、dtype、clientBuf 的 Qnn_Tensor_t 数组 |
| 5. execute | graphExecute | CPU 可读取的输出 buffer |
| 6. release | contextFree、deviceFree、backendFree、logFree、systemContextFree | 释放 QNN 所有者对象；graph 跟随 context 失效 |

可对应 TensorRT 的经验：provider/backend/device 初始化类似 runtime 环境准备；context restore 类似反序列化 engine；准备 QNN tensor 与绑定 buffer 类似设置 tensor 地址；同步 graphExecute 类似提交推理并等待完成。但这里没有 CUDA stream，QNN context/graph 和 TensorRT 的 engine/execution context 也不是逐一等价的类型。

## 为什么先 dlopen，再拿一张函数表

`libQnnHtp.so` 在运行时动态加载。`dlsym` 查找导出的 `QnnInterface_getProviders`，它返回提供者列表；程序按 headers 的 API major/minor 选择兼容表，再通过 `session.api.graphExecute(...)` 调用。

这么做让程序编译时依赖 QNN headers，运行时选择 backend 库。链接命令没有 `-lQnnHtp`；`readelf -d` 的 DT_NEEDED 因而不会列出 QNN 库，但程序仍然在运行时需要它们。

SDK 产品版本 2.45 和打印的 QNN core API 2.34.0 属于不同版本标识，不是拿错了 SDK。实际库文件哈希、工具 build version 与 headers 一起记录更可靠。

System API 有另一张函数表。2.45 的 headers 将旧 `systemContextGetBinaryInfo` 标为 deprecated，本程序使用 `systemContextGetMetaData`。它只读取描述信息，并不会替你创建 HTP 中可执行的 context。

## 三种“context”不要混起来

- 磁盘的 `rf_detr.bin`：序列化字节，`readFile` 后存在 `session.binary`。
- `metadataContext`：System API 的对象，拥有 metadata 中的指针。
- `session.context`：HTP backend 中恢复后的 QNN context，拥有真正执行所需的图与资源。

只读到 metadata 不代表 HTP 已加载成功。真正恢复发生在 `contextCreateFromBinary`；真正找到 graph handle 发生在 `graphRetrieve`。

`singleGraph()` 处理本 SDK 的 binary/graph metadata V1/V2/V3 差异，拒绝多图。`withTensor()` 将 tensor V1/V2 的分支集中起来，其他代码直接访问共有字段。版本处理集中在 helper 中，是为了保持 main 的调用顺序清楚。

## buffer 是谁分配、谁拥有的

```text
Session
├── binary：拥有文件字节
├── metadataContext：拥有 graph/tensor metadata 指针
└── context：拥有 graph handle 的有效期

TensorBuffers
├── tensors[]：复制的 Qnn_Tensor_t 描述（部分字段借用 metadata 指针）
└── buffers[]：拥有真正的输入/输出字节
        ↑
        └── tensors[i].clientBuf.data 指向这里
```

`TensorBuffers` 先复制 tensor 描述，保留真实 id、名字、量化信息等，再设置 `memType=RAW` 和 `clientBuf`。raw buffer 用 C++ vector 分配，执行期间不 resize、不移动其存储。

复制 `Qnn_Tensor_t` 不等于深拷贝 metadata。名字、dimensions、部分 quantization 描述仍可能指向 System context 的内存。因此本例保留 `metadataContext` 直到所有执行结束；如果以后想提前释放 System context，就必须深拷贝所有引用字段，不能只复制 struct。

对这两种 32 位类型，buffer 字节数是 `4 × 各维度乘积`。分配前检查维度非零、乘法不溢出、总长可由 client buffer 的 uint32 表示。动态维度和其他 dtype 明确拒绝。

`readInputs` 按名称匹配命令行文件，并验证字节数，然后拷贝到已经分配好的 buffer。它不会把 NCHW 变成 NHWC，不会把 float 转成量化整数。raw 数据的正确含义由 prepare 和模型 contract 保证。

`graphExecute` 同步返回成功后，输出 buffer 可读取；程序每次另建 `Result_i` 保存 native bytes。这里没有 NMS、sigmoid 或画框；这些算法只应按模型 recipe 在后处理阶段做。

## 为什么释放资源后仍保留库映射

首次设备测试发生了一个值得保留的失败：6 次 execute、写文件和所有 QNN free 都成功，但动态库卸载后进程退出 SIGSEGV，crash 栈指向未映射地址。未定位到 SDK 内部具体符号，因此不能断言是哪个内部回调。

本例对 QNN 库使用 `RTLD_NODELETE`，将代码映射保持到进程结束，防止退出阶段访问已卸载代码；`dlclose` 仍释放加载引用，但不提前解除映射。QNN 的 context/device/backend/logger/System context 均显式 free，并检查错误。这个选择适用于当前一次启动、一次退出的命令行工具；它不是可热卸载的长运行服务设计。

`Session::close()` 是正常路径，返回是否释放成功；析构再调用 close 是异常路径兜底。句柄被设为 nullptr 后不会再次 free。任何失败抛异常，main 捕获、打印 `[FAIL]`、退出 1；部署脚本还检查整个 adb shell 的退出码，不能只看日志中的“完成”。

## 时间从哪里来

`Timings::measure` 用 `std::chrono::steady_clock` 包围函数。`execute_i` 只包围同步 `graphExecute`；`read_inputs`、`write_outputs_i`、context 文件读取和恢复另计。

这使你能亲自修改计时边界。例如想测包含输出写盘的吞吐，需要新增更外层计时，而不是把现有 execute 数字改称端到端延迟。buffer 分配、参数解析、少量日志和进程加载并未全部纳入一个总计时，不能简单把 CSV 相加当完整启动时延。

第 0 次单独输出，后续保留每次样本。当前 C++ 日志为 warn，没有启用 backend profiling；与工具的 verbose/basic 数字只能说明量级与阶段，不能直接归因性能差异。

## 编译是怎样变成 Android executable 的

[build-cpp.sh](../scripts/build-cpp.sh) 实际调用本地 NDK 的 `aarch64-linux-android28-clang++`，使用 C++17、QNN headers 和 `-ldl`；没有 CMake。`-static-libstdc++` 将 NDK 的 C++ 标准库静态带进程序，避免额外部署 `libc++_shared.so`，并不意味着 QNN 或 Android libc 也被静态链接。

`artifacts/cpp/qnn-context-runner` 是 Android ARM64 ELF，不能直接在 Fedora x86_64 上运行。构建脚本用 readelf 展示架构与依赖，设备脚本 push 后 chmod，再由 adb shell 执行。API 28 是构建的最低 Android API 目标；本次实际手机是 Android 16 / API 36。

学习时可以先在 main 中给每一段写一句自己的解释，再尝试增加一个仅打印 metadata 的模式。当前工程没有提前加入异步执行、共享内存或性能配置；先把已有每个句柄和 buffer 的所有者说清楚。
