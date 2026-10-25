# NVIDIA 設備之間與 NVIDIA／AMD 之間的可攜性

查核日期：2026-10-25。核心區分：**原始碼可重用、模型可交換、二進位可直接執行、效能相近，是四件不同的事。** 以下為依官方文件整理的部署建議，未進行硬體測試。

## 跨 NVIDIA 設備：哪些資產值得共用？

| 資產 | Orin → 其他 NVIDIA GPU | 要做的工作 |
|---|---|---|
| Python 業務邏輯與標準 PyTorch operator | 通常可重用原始碼 | 安裝目標 CPU／OS／CUDA 的 framework；測試 operator 與精度 |
| CUDA C++ 原始碼、cuBLAS／cuFFT 等呼叫 | 通常可重新編譯移植 | 選正確 toolkit、GPU target、host compiler；查 library 平台支援 |
| CPU 執行檔與 `.so`／Python native extension | 不能把 Orin 的 Arm64 檔直接當 x86_64 檔使用 | 各 CPU 架構重建；核對 libc、C++ ABI、依賴與 OS |
| GPU cubin／fatbin | 有條件 | 必須包含目標可執行的 SASS 或可 JIT 的 PTX |
| ONNX 模型 | 是較好的交換資產 | 逐後端檢查 opset、operator、shape、量化與 custom op |
| TensorRT `.engine`／`.plan` | 預設不作通用發布資產 | 原則上依 GPU 型號、TensorRT 與部署環境重建 |
| DLA 編譯結果 | 不能泛用到一般 RTX／資料中心 GPU | 需有相容 DLA 與 runtime；另建 GPU 推論路徑 |
| Docker image | 可共用配置策略，通常需要不同映像變體 | 區分 `linux/arm64`、`linux/amd64` 及 Jetson／server 套件 |
| 模型驗證資料、輸入輸出規格、基準測試 | 高度適合共用 | 保留誤差容許值、前後處理、版本與測試條件 |

技術依據：[CUDA for Tegra](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-for-tegra-appnote/index.html)、[CUDA GPU 相容性](https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html)、[TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)、[ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/)、[Docker 多平台映像](https://docs.docker.com/build/building/multi-platform/)。

## CPU 架構、GPU 架構、平台不是同一個座標

Jetson AGX Orin 的 GPU compute capability 為 **8.7**，編譯目標通常寫作 `sm_87`；CPU host 是 Arm64。桌機 RTX 常見 x86_64 host，而 Arm 伺服器即使同樣是 `aarch64`，也不能直接假定等同 Jetson/Tegra 的驅動、BSP、camera 與 multimedia 環境。[NVIDIA compute capability 表](https://developer.nvidia.com/cuda/gpus)；[CUDA for Tegra](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-for-tegra-appnote/index.html)

查核時 JetPack 7.2.1 已列 Orin family；下載頁中統一 Arm CUDA 安裝與 SBSA 的文字明確以 **Thor** 說明，不能據此宣稱 AGX Orin 已成為一般 SBSA 伺服器，或所有 server Arm64 wheel 都可安裝。[JetPack 7.2.1 下載與支援說明](https://developer.nvidia.com/embedded/jetpack/downloads)

跨平台建置至少記錄：

```text
host: aarch64 / x86_64
OS + Jetson Linux/BSP（若適用）
CUDA toolkit + runtime + driver
GPU compute capability
TensorRT / cuDNN / framework / Python ABI
模型版本、精度、shape、plugin、編譯旗標
```

## cubin、PTX 與 fatbin 的正確理解

一般 CUDA cubin 的相容性在相同 compute capability 主版本內，允許目標 GPU 的次版本相同或更高；PTX 可由驅動 JIT 成目標 GPU 的機器碼，提供向較新 GPU 的前向相容機制。若二進位沒有相容 cubin，也沒有可用 PTX，kernel 便不能執行。[Ampere compatibility guide](https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html)

例子：只有 `sm_87` cubin **不等於**可在 `sm_80` A100 上執行；8.0 低於 8.7。也不能把「有 PTX」理解成可在 AMD GPU 上執行。較新的 architecture-specific `a` target 另有例外，不保證一般前向相容。[PTX 與二進位相容說明](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/)；[architecture-specific targets](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)

示意建置：若同一演算法確實只使用 compute 8.0 可支援的特性，可在各 host 平台建置時放入多個 GPU target：

```bash
nvcc -O3 kernel.cu -o application \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_87,code=sm_87 \
  -gencode arch=compute_80,code=compute_80
```

此範例說明 fatbinary 配置，並非提供已存在的 `kernel.cu`，也未實際編譯；需使用支援這些 targets 的 toolkit。它不會把 host code 變成跨 x86／Arm 通用二進位，不會補上 library 缺少的 target，也不保證新的 GPU 特性或最佳效能。驅動亦需理解所產生的 PTX ISA。[NVCC GPU code generation](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-compiler-driver-nvcc/index.html)

## TensorRT：共用模型，依部署目標產生 engine

TensorRT engine 是針對部署配置最佳化的產物。官方 engine compatibility 文件說明預設與建立時的裝置類型相關；硬體相容模式會限制可選 tactics，且查核文件明載 **NVIDIA JetPack 不支援 hardware compatibility mode**。不能因 Orin 屬於 Ampere 就套用桌機 `AMPERE_PLUS` 的跨卡發布策略。[TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)

TensorRT 另有 version compatibility，但它與 hardware compatibility 分屬不同條件，亦不是任意升降版本的保證；外部 plugins 和 runtime 仍需配套。穩妥的 Orin 發布策略為保留 ONNX／原始模型、建置程式和參數，在目標環境建立並快取 engine。[TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)

建議快取鍵包含：模型雜湊、GPU 型號、TensorRT 版本、CUDA／BSP 環境、精度、dynamic-shape profiles、plugin 版本。更換 GPU、升級 JetPack 或修改模型時重新驗證；不要以副檔名相同判定可沿用。

**版本證據邊界：** 查核時 JetPack 7.2.1 列 TensorRT 10.16.2，但 TensorRT `latest` 文件已顯示其他版本；部分舊文件網址也會重新導向 `latest`。上述採保守的每目標重建原則，沒有將 latest 所有新功能宣稱為 JetPack 10.16.2 已提供。實際發布須對照已安裝版本的 release notes 與支援矩陣。[JetPack 套件表](https://developer.nvidia.com/embedded/jetpack/downloads)

## Jetson 特定硬體加速不能只靠 CUDA API 概括

Tegra 的 CPU 與 iGPU 共享 SoC DRAM，桌機 dGPU 則通常有獨立 VRAM；記憶體配置、快取、一致性與拷貝成本不同。可移植的 CUDA 程式仍要重新量測 pinned／managed／device memory 策略；共享實體 RAM 不代表所有 CUDA API 的資料路徑都零拷貝。[CUDA for Tegra 記憶體說明](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-for-tegra-appnote/index.html)

DLA、PVA、相機 ISP、V4L2／Jetson multimedia pipeline 是特定硬體與軟體環境能力。一般 RTX 卡不能執行 Orin 的 DLA loadable；相機擷取與影片 codec 路徑應另包成 adapter。將共同的模型／張量計算與板級 I/O 分離，可以保留最多可移植部分。[JetPack 硬體相關 SDK 表](https://developer.nvidia.com/embedded/jetpack/downloads)；[TensorRT DLA 文件](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html)

DLA 尤其要核對版本：查核時 TensorRT latest 文件另記錄 11.x 的 JetPack／DriveOS DLA 限制，不能把它當作 JetPack 7.2.1 所附 10.16.2 的完整相容矩陣。若產品依賴 DLA，應在選定 JetPack 上實際完成建模、執行與 fallback 驗證後再鎖版。

## 容器：封裝相依性，不消除平台差異

Docker 的 multi-platform manifest 可以讓同一 tag 對應不同 CPU 架構映像，並不代表映像內的 native binary 自動通用。Jetson 應選與其軟體堆疊相容的底層映像與套件；還需要 host 的 GPU 驅動與 NVIDIA Container Toolkit 配置。[Docker 多平台建置](https://docs.docker.com/build/building/multi-platform/)；[NVIDIA Container Toolkit 安裝](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

建議將 CI 拆成 `jetson-arm64`、`nvidia-x86_64`、`amd-x86_64` 等實際部署組合，各自建置與驗證。可共享上層 Dockerfile 步驟及應用來源；driver、GPU library、Python wheel 和 framework 後端需依平台選擇。

## NVIDIA ↔ AMD：可行路徑與限制

| 路徑 | 可重用程度 | 不會自動處理的部分 |
|---|---|---|
| PyTorch 高階模型 | 通常較高，AMD ROCm 仍使用 `torch.cuda` API | custom CUDA extension、特定 fused op、硬體精度差異、支援矩陣 |
| ONNX + 各家推論後端 | 可共用模型檔與部分應用流程 | opset、custom op、量化、shape、CPU fallback、provider 套件 |
| CUDA → HIP／HIPIFY | 可協助轉換 kernel 與 API 原始碼 | 未支援 API、inline PTX、warp 假設、library 語義和效能調校 |
| SYCL | 共用 C++ 運算描述的候選路徑 | 仍需 NVIDIA／AMD backend、編譯器與 library；不保證 Jetson 套件支援 |
| TensorRT engine／CUDA binary | 不可直接作 AMD 發布檔 | 需從模型或原始碼重建 AMD 路徑 |

依據：[PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html)、[ONNX Runtime providers](https://onnxruntime.ai/docs/execution-providers/)、[AMD HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.0/how-to/hip_porting_guide.html)、[Intel DPC++ migration guide](https://cdrdv2-public.intel.com/834754/dpcpp-compatibility-tool_developer-guide-reference_2025.0-768918-834754.pdf)。

HIPIFY 應視為遷移工具，不是功能與效能認證。尤其不要硬寫 warp size 為 32 或 64；要查 runtime／架構特性，重新檢查 lane mask、shuffle、同步、atomic、shared memory 與 launch 配置。已大量依賴 TensorRT／DeepStream／Jetson camera 的應用，替換工作通常超過 kernel 轉譯。[HIP 6.4 porting guide](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.0/how-to/hip_porting_guide.html)

ONNX Runtime 查核頁已將舊 ROCm execution provider 標為 deprecated，AMD GPU 新路徑應優先核對 MIGraphX EP 的版本／平台支援，不宜直接複製舊教學。TensorRT EP 則建議同時註冊 CUDA EP，讓不支援的節點可由 CUDA 處理；仍要檢查實際分配，避免以為全部都在預期加速器上執行。[Providers 列表](https://onnxruntime.ai/docs/execution-providers/)；[MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)；[TensorRT EP](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

## 建議軟體結構

```text
應用規則、資料格式、模型與驗證資料（共用）
    ↓
Inference / NumericBackend / VideoSource 介面（共用）
    ├── Jetson：TensorRT + CUDA + 板級影音 I/O
    ├── NVIDIA server：TensorRT / PyTorch + CUDA + server I/O
    ├── AMD：ROCm PyTorch / MIGraphX + HIP + 平台 I/O
    └── CPU：正確性參考實作或允許時的降級路徑
```

這是設計建議，並非既有程式碼架構。優先抽象真正跨平台的能力與資料生命週期，避免用一個巨型介面假裝所有加速器都支援相同功能。

## 移植驗收流程

1. 固定模型版本、輸入尺寸、batch、精度、前後處理與測試資料。
2. 先驗證正確性及容許誤差，再確認每一段實際跑在 CPU／GPU／DLA 的哪裡。
3. 在各目標重建 native library、framework extension 與推論 engine。
4. 分別量測冷啟動、暖機後端到端延遲、吞吐量與記憶體峰值。
5. 記錄功耗模式、溫度、降頻、I/O、provider fallback 與長時間穩定性。
6. 把可用平台列成明確矩陣；未測組合標為未驗證，不以「CUDA 支援」推定可部署。

AMD 硬體與軟體選擇另見 [AMD 運算方案](05-amd-compute.md)，語言與小型環境檢查見 [語言選擇](03-language-selection.md)。
