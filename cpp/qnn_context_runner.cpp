// 阅读顺序：main → Session → TensorBuffers。只做 QNN runtime，不做 RF-DETR 后处理。
// SDK 头文件来自 QAIRT_ROOT；本仓库不复制 SDK 实现或 SampleApp 源码。
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Bytes = std::vector<uint8_t>;
using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check(Qnn_ErrorHandle_t error, const char* operation) {
    if (error != QNN_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed, QNN error=" +
                                 std::to_string(error));
}

Bytes readFile(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(file.is_open(), "Cannot open: " + path.string());
    const auto size = file.tellg();
    require(size > 0, "Empty/unreadable file: " + path.string());
    Bytes bytes(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    require(file.good(), "Incomplete read: " + path.string());
    return bytes;
}

// 计时只包围传入函数；execute 的区间不包含文件读写。它是 CPU 侧同步 API
// wall time（含 RPC 等），并不等于 HTP 内部 kernel 时间。
struct Timings {
    std::vector<std::pair<std::string, double>> rows;
    template<class Function> void measure(const std::string& name, Function function) {
        const auto start = Clock::now();
        function();
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        rows.emplace_back(name, ms);
        std::cout << "[TIME] " << name << " " << std::fixed << std::setprecision(3)
                  << ms << " ms\n";
    }
    void save(const fs::path& path) const {
        std::ofstream file(path);
        file << "phase,milliseconds\n" << std::fixed << std::setprecision(6);
        for (const auto& row : rows) file << row.first << ',' << row.second << '\n';
        file.close();
        require(!file.fail(), "Cannot write timings: " + path.string());
    }
};

// dlopen 得到库句柄，dlsym 得到 provider 查询函数；真正的 QNN API 从表中取。
struct Library {
    void* handle = nullptr;
    Library() = default;
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    void open(const std::string& path) {
        // 本 demo 将 runtime 映射保留到进程结束。该 SDK/设备组合在实际卸载
        // 库后发生过退出 SIGSEGV；NODELETE 避免退出回调访问已卸载代码。
        // QNN context/device/backend 等资源仍由 close() 显式释放。
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
        if (!handle) throw std::runtime_error("dlopen " + path + ": " + dlerror());
    }
    template<class Function> Function symbol(const char* name) {
        dlerror();
        void* address = dlsym(handle, name);
        const char* error = dlerror();
        if (error) throw std::runtime_error(std::string("dlsym ") + name + ": " + error);
        return reinterpret_cast<Function>(address);
    }
    ~Library() { if (handle) dlclose(handle); }
};

void qnnLog(const char* format, QnnLog_Level_t, uint64_t, va_list args) {
    std::fputs("[QNN] ", stderr);
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
}

// System context 拥有 metadata 指针；QNN context 拥有恢复出来的 graph。
// 保留 System context 与 binary buffer 直到结束，避免浅拷贝 tensor 的悬空指针。
struct Session {
    Library backendLibrary, systemLibrary;
    QNN_INTERFACE_VER_TYPE api{};
    QNN_SYSTEM_INTERFACE_VER_TYPE system{};
    Qnn_LogHandle_t logger = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    QnnSystemContext_Handle_t metadataContext = nullptr;
    Bytes binary;

    void load(const std::string& backendPath, const std::string& systemPath) {
        backendLibrary.open(backendPath);
        const QnnInterface_t** providers = nullptr;
        uint32_t count = 0;
        auto getProviders = backendLibrary.symbol<decltype(&QnnInterface_getProviders)>(
            "QnnInterface_getProviders");
        check(getProviders(&providers, &count), "QnnInterface_getProviders");
        bool found = false;
        for (uint32_t i = 0; i < count; ++i) {
            const auto& v = providers[i]->apiVersion.coreApiVersion;
            if (v.major == QNN_API_VERSION_MAJOR && v.minor >= QNN_API_VERSION_MINOR) {
                api = providers[i]->QNN_INTERFACE_VER_NAME;
                std::cout << "QNN core API " << v.major << '.' << v.minor << '.' << v.patch << '\n';
                found = true;
                break;
            }
        }
        require(found, "No compatible QNN provider; use the matching QAIRT SDK/runtime");

        systemLibrary.open(systemPath);
        const QnnSystemInterface_t** systemProviders = nullptr;
        auto getSystemProviders = systemLibrary.symbol<decltype(&QnnSystemInterface_getProviders)>(
            "QnnSystemInterface_getProviders");
        check(getSystemProviders(&systemProviders, &count), "QnnSystemInterface_getProviders");
        found = false;
        for (uint32_t i = 0; i < count; ++i) {
            const auto& v = systemProviders[i]->systemApiVersion;
            if (v.major == QNN_SYSTEM_API_VERSION_MAJOR && v.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
                system = systemProviders[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
                found = true;
                break;
            }
        }
        require(found, "No compatible QNN System provider");
        require(api.logCreate && api.logFree && api.backendCreate && api.backendFree &&
                api.deviceCreate && api.deviceFree && api.contextCreateFromBinary &&
                api.contextFree && api.graphRetrieve && api.graphExecute &&
                system.systemContextCreate && system.systemContextGetMetaData &&
                system.systemContextFree, "Required QNN API entry is null");
    }

    // 显式 close 让正常路径能检查释放错误；析构是异常路径的兜底。
    bool close() noexcept {
        bool ok = true;
        auto released = [&](Qnn_ErrorHandle_t error, const char* name) {
            if (error != QNN_SUCCESS) {
                std::fprintf(stderr, "[FAIL] %s: %llu\n", name,
                             static_cast<unsigned long long>(error));
                ok = false;
            }
        };
        if (context) { released(api.contextFree(context, nullptr), "contextFree"); context = nullptr; }
        if (device) { released(api.deviceFree(device), "deviceFree"); device = nullptr; }
        if (backend) { released(api.backendFree(backend), "backendFree"); backend = nullptr; }
        if (logger) { released(api.logFree(logger), "logFree"); logger = nullptr; }
        if (metadataContext) {
            released(system.systemContextFree(metadataContext), "systemContextFree");
            metadataContext = nullptr;
        }
        return ok;
    }
    ~Session() { close(); }
};

struct GraphInfo {
    const char* name;
    uint32_t inputCount, outputCount;
    Qnn_Tensor_t *inputs, *outputs;
};

// 不猜测图名；目前限定一个 graph，版本差异集中在这两个小函数。
GraphInfo singleGraph(const QnnSystemContext_BinaryInfo_t& binary) {
    uint32_t count = 0;
    QnnSystemContext_GraphInfo_t* graphs = nullptr;
    auto unpack = [&](const auto& info) { count = info.numGraphs; graphs = info.graphs; };
    switch (binary.version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: unpack(binary.contextBinaryInfoV1); break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: unpack(binary.contextBinaryInfoV2); break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: unpack(binary.contextBinaryInfoV3); break;
        default: throw std::runtime_error("Unsupported context metadata version");
    }
    require(count == 1 && graphs, "This demo requires exactly one graph");
    auto view = [](const auto& g) -> GraphInfo {
        return {g.graphName, g.numGraphInputs, g.numGraphOutputs, g.graphInputs, g.graphOutputs};
    };
    switch (graphs[0].version) {
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1: return view(graphs[0].graphInfoV1);
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2: return view(graphs[0].graphInfoV2);
        case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3: return view(graphs[0].graphInfoV3);
        default: throw std::runtime_error("Unsupported graph metadata version");
    }
}

template<class Function> decltype(auto) withTensor(Qnn_Tensor_t& tensor, Function function) {
    switch (tensor.version) {
        case QNN_TENSOR_VERSION_1: return function(tensor.v1);
        case QNN_TENSOR_VERSION_2: return function(tensor.v2);
        default: throw std::runtime_error("Unsupported tensor version");
    }
}

bool safeName(const std::string& name) {
    return !name.empty() && name != "." && name != ".." &&
           name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") == std::string::npos;
}

// vector 拥有 raw buffer；Qnn_Tensor_t.clientBuf 仅借用其地址。
// float32 / int32 已覆盖本模型，其他类型报错，避免暗中量化或误读。
struct TensorBuffers {
    std::vector<Qnn_Tensor_t> tensors;
    std::vector<Bytes> buffers;
    std::vector<std::string> names;

    TensorBuffers(Qnn_Tensor_t* source, uint32_t count, const char* direction,
                  std::ofstream& metadata) {
        require(source && count, "Missing input/output metadata");
        tensors.assign(source, source + count);
        buffers.resize(count);
        names.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto& tensor = tensors[i];
            if (tensor.version == QNN_TENSOR_VERSION_2 && tensor.v2.isDynamicDimensions) {
                for (uint32_t d = 0; d < tensor.v2.rank; ++d)
                    require(!tensor.v2.isDynamicDimensions[d], "Dynamic dimensions are not supported");
            }
            withTensor(tensor, [&](auto& t) {
                require(t.name && safeName(t.name), "Tensor name cannot be used as a filename");
                names[i] = t.name;
                require(t.dataFormat == QNN_TENSOR_DATA_FORMAT_DENSE, "Only dense tensors are supported");
                const char* dtype = nullptr;
                if (t.dataType == QNN_DATATYPE_FLOAT_32) dtype = "float32";
                if (t.dataType == QNN_DATATYPE_INT_32) dtype = "int32";
                require(dtype, "Unsupported tensor dtype for " + names[i] + ": " + std::to_string(t.dataType));
                require(t.rank == 0 || t.dimensions, "Missing dimensions");
                size_t bytes = 4;
                std::string shape;
                for (uint32_t d = 0; d < t.rank; ++d) {
                    require(t.dimensions[d] && bytes <= std::numeric_limits<uint32_t>::max() / t.dimensions[d],
                            "Invalid/oversized tensor dimensions");
                    bytes *= t.dimensions[d];
                    shape += (d ? "x" : "") + std::to_string(t.dimensions[d]);
                }
                buffers[i].resize(bytes);
                t.memType = QNN_TENSORMEMTYPE_RAW;
                t.clientBuf = {buffers[i].data(), static_cast<uint32_t>(bytes)};
                metadata << direction << '\t' << names[i] << '\t' << dtype << '\t'
                         << shape << '\t' << bytes << '\n';
                std::cout << "[TENSOR] " << direction << ' ' << names[i] << ' ' << dtype
                          << " [" << shape << "] " << bytes << " bytes\n";
            });
        }
    }

    void readInputs(const std::map<std::string, fs::path>& files) {
        require(files.size() == tensors.size(), "Supply exactly one --input name=file per input tensor");
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto it = files.find(names[i]);
            require(it != files.end(), "Missing input: " + names[i]);
            const Bytes data = readFile(it->second);
            require(data.size() == buffers[i].size(), "Wrong input byte count for " + names[i]);
            std::copy(data.begin(), data.end(), buffers[i].begin());
        }
    }

    void writeOutputs(const fs::path& directory) const {
        require(fs::create_directory(directory), "Output directory already exists: " + directory.string());
        for (size_t i = 0; i < buffers.size(); ++i) {
            std::ofstream file(directory / (names[i] + ".raw"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(buffers[i].data()), buffers[i].size());
            file.close();
            require(!file.fail(), "Cannot write output: " + names[i]);
        }
    }
};

struct Options {
    std::string backend, system;
    fs::path context, output;
    std::map<std::string, fs::path> inputs;
    int runs = 6;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        require(i + 1 < argc, "Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--backend") o.backend = value;
        else if (key == "--system") o.system = value;
        else if (key == "--context") o.context = value;
        else if (key == "--output-dir") o.output = value;
        else if (key == "--runs") {
            require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                    "--runs must be an integer");
            o.runs = std::stoi(value);
            require(o.runs >= 1 && o.runs <= 1000, "--runs must be 1..1000");
        } else if (key == "--input") {
            const auto split = value.find('=');
            require(split != std::string::npos && split > 0 && split + 1 < value.size(), "Use --input name=file");
            require(o.inputs.emplace(value.substr(0, split), value.substr(split + 1)).second,
                    "Duplicate input name");
        } else throw std::runtime_error("Unknown option: " + key);
    }
    require(!o.backend.empty() && !o.system.empty() && !o.context.empty() && !o.output.empty() &&
            !o.inputs.empty(), "Required: --backend --system --context --input name=file --output-dir");
    return o;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "qnn-context-runner --backend libQnnHtp.so --system libQnnSystem.so\n"
                     "  --context model.bin --input name=file.raw [--input name=file.raw ...]\n"
                     "  --output-dir NEW_DIR [--runs 6]\n"
                     "Single graph; static dense float32/int32 tensors; native raw files.\n";
        return 0;
    }
    try {
        const Options o = parse(argc, argv);
        fs::create_directories(o.output);
        require(!fs::exists(o.output / "tensors.tsv") && !fs::exists(o.output / "Result_0"),
                "Use a fresh output directory");
        Timings timings;
        Session session;

        // 1. 初始化运行库与 HTP device。这部分不是模型 Compose/Finalize。
        timings.measure("load_libraries", [&] { session.load(o.backend, o.system); });
        timings.measure("backend_device_create", [&] {
            check(session.api.logCreate(qnnLog, QNN_LOG_LEVEL_WARN, &session.logger), "logCreate");
            check(session.api.backendCreate(session.logger, nullptr, &session.backend), "backendCreate");
            check(session.api.deviceCreate(session.logger, nullptr, &session.device), "deviceCreate");
        });

        // 2. 读取序列化 context 的字节与 metadata；getMetaData 不执行模型。
        timings.measure("read_context_file", [&] { session.binary = readFile(o.context); });
        const QnnSystemContext_BinaryInfo_t* metadata = nullptr;
        timings.measure("read_metadata", [&] {
            check(session.system.systemContextCreate(&session.metadataContext), "systemContextCreate");
            check(session.system.systemContextGetMetaData(session.metadataContext, session.binary.data(),
                  session.binary.size(), &metadata), "systemContextGetMetaData");
            require(metadata, "No binary metadata");
        });
        const GraphInfo info = singleGraph(*metadata);
        require(info.name && *info.name, "Missing graph name");
        std::cout << "[GRAPH] " << info.name << '\n';

        // 3. 恢复已经 finalize 的 context，取回 graph handle；这里不再 graphFinalize。
        timings.measure("context_create_from_binary", [&] {
            check(session.api.contextCreateFromBinary(session.backend, session.device, nullptr,
                  session.binary.data(), session.binary.size(), &session.context, nullptr),
                  "contextCreateFromBinary");
        });
        Qnn_GraphHandle_t graph = nullptr;
        timings.measure("graph_retrieve", [&] {
            check(session.api.graphRetrieve(session.context, info.name, &graph), "graphRetrieve");
        });

        // 4. metadata 决定 buffer 的名字、类型与字节数，输入 raw 在 CPU 内存中准备。
        std::ofstream tensorFile(o.output / "tensors.tsv");
        tensorFile << "direction\tname\tdtype\tshape\tbytes\n";
        TensorBuffers inputs(info.inputs, info.inputCount, "input", tensorFile);
        TensorBuffers outputs(info.outputs, info.outputCount, "output", tensorFile);
        tensorFile.close();
        require(!tensorFile.fail(), "Cannot write tensors.tsv");
        timings.measure("read_inputs", [&] { inputs.readInputs(o.inputs); });

        // 5. 相同 input、相同 context 连续执行；第 0 次单独看，后续也不能武断称稳态。
        for (int i = 0; i < o.runs; ++i) {
            timings.measure("execute_" + std::to_string(i), [&] {
                check(session.api.graphExecute(graph, inputs.tensors.data(), info.inputCount,
                      outputs.tensors.data(), info.outputCount, nullptr, nullptr), "graphExecute");
            });
            timings.measure("write_outputs_" + std::to_string(i), [&] {
                outputs.writeOutputs(o.output / ("Result_" + std::to_string(i)));
            });
        }

        // 6. context 释放时 graph 随之失效；runtime 代码映射保持到进程退出。
        timings.measure("free_resources", [&] { require(session.close(), "QNN cleanup failed"); });
        timings.save(o.output / "timings.csv");
        std::cout << "[OK] completed " << o.runs << " inferences; native outputs saved\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
