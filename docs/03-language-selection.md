# AGX Orin 開發語言選擇

查核日期：2026-10-25。範圍為 NVIDIA Jetson AGX Orin；以下選型是工程建議，並非本機硬體實測。

## 建議：Python 作研究與控制，C++／CUDA C++ 作需要精確控制的熱點

若尚未有既定語言，先以 **Python + PyTorch／TensorRT** 完成模型與資料流程；量測後才把影像資料路徑、CPU 密集處理或自訂 GPU kernel 移至 **C++／CUDA C++**。TensorRT 有 Python 與 C++ API，而 Python 也可透過 NVIDIA CUDA Python 存取 CUDA；使用 Python 不代表 GPU 計算由 Python 直譯器逐元素執行。[TensorRT 文件](https://docs.nvidia.com/deeplearning/tensorrt/latest/)；[CUDA Python](https://developer.nvidia.com/cuda/python)

對既有 C++ 機器人、影音或工控產品，直接採 C++ 主程式可能更省整合成本。對資料科學團隊，先全程 Python 通常較容易驗證模型；是否需要重寫應由延遲、CPU 使用量與維護需求決定，而不是假設 C++ 必定讓 GPU kernel 更快。

## 依工作選語言

| 工作 | 優先選擇 | 使用方式 | 主要取捨 |
|---|---|---|---|
| 模型實驗、資料前處理、推論原型 | Python | PyTorch、TensorRT Python API | 開發快；需管理 wheel、Python ABI 與 JetPack 相容性 |
| 數值演算法原型 | Python | GPU 陣列／張量運算，必要時 CUDA Python | 盡量批次運算，避免逐元素 Python 迴圈和頻繁同步 |
| 長時間攝影機串流、低延遲服務 | C++ | TensorRT C++、GStreamer、CUDA library | 容易控制 buffer、執行緒及生命週期；開發成本較高 |
| 自訂 stencil、濾波、幾何或融合 kernel | CUDA C++ | `nvcc` 編譯 `.cu`，以 C API 或 Python binding 對外 | NVIDIA 工具鏈整合直接；跨 AMD 需改後端或 HIP 化 |
| NVIDIA 與 AMD 都必須支援的自訂運算 | C++ + 後端抽象 | CUDA／HIP，或評估 SYCL | 建置與效能驗證需兩套；不能以單一來源等同單一執行檔 |
| Rust 已是產品主語言 | Rust 主程式 + C ABI | 呼叫 C++／CUDA 封裝；將資源所有權收斂在邊界 | 可保留 Rust 服務架構；FFI 仍需處理 pointer、stream、同步與錯誤 |
| Julia 數值研究團隊 | Julia + CUDA.jl | GPU 陣列與 kernel | 有 Jetson 支援路徑；特定 CUDA、JetPack 和附加 library 要另驗證 |
| 既有 Fortran 科學程式 | Fortran CPU + C/CUDA 介面，或在支援的伺服器用 CUDA Fortran | 保留既有數值核心 | 不應將 HPC SDK 的 Arm server 支援視為 Orin 官方支援 |

上表的適用性為分析建議；底層能力分別見 [CUDA 編譯器](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-compiler-driver-nvcc/index.html)、[HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.0/how-to/hip_porting_guide.html)、[CUDA.jl 安裝](https://cuda.juliagpu.org/stable/installation/overview/)、[CUDA Fortran](https://developer.nvidia.com/cuda-fortran)。

## Python：最適合作為初始主語言

NVIDIA 提供針對 Jetson／JetPack 的 PyTorch wheel 與相容矩陣。選包時必須同時匹配 JetPack、CUDA、Python 版本、`aarch64` 及相關依賴；不能因桌機上 `pip install torch` 成功，就認定相同指令在 Orin 會裝到 GPU 版。[Jetson PyTorch 安裝](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/)；[Jetson PyTorch 相容矩陣](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)

CUDA Python 包含高階 runtime 操作與低階 API bindings；各子套件獨立版本化。查核時部分 `latest` 文件明示來自開發分支，所以應依鎖定版本安裝，而非直接照最新網頁升級整套環境。[官方原始碼與套件結構](https://github.com/NVIDIA/cuda-python)；[cuda.bindings 文件狀態](https://nvidia.github.io/cuda-python/cuda-bindings/latest/)

實作建議：

1. 先使用 PyTorch 等已最佳化 operator，減少自行維護 kernel。
2. 保留 GPU 上的中間張量，降低 CPU↔GPU 往返和隱性同步。
3. 以 profiler 區分 Python／CPU 時間、kernel 時間、資料搬移與等候輸入。
4. 只有確認瓶頸後才選擇 C++ extension、CUDA kernel 或額外編譯工具。
5. 正式部署鎖定完整套件版本與下載來源；另存模型、輸入規格與精度測試。

最小環境檢查範例；此段為供目標硬體執行的診斷程式，未在本次研究執行：

```python
import platform
import torch

print("CPU architecture:", platform.machine())
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("HIP build:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(128, 128, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("Result:", y.shape, y.device)
```

AMD 的 ROCm PyTorch **也沿用 `torch.cuda` 與 `device="cuda"`**；以 `torch.version.hip` 分辨後端。這種 API 共用有利於原始碼移植，但自訂 CUDA extension、TensorRT plugin 或特定低精度 kernel 不因此自動相容。[PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html)

## C++／CUDA C++：需要控制時再深入

CUDA C++ 可呼叫 cuBLAS、cuFFT 等 library，也可撰寫 `__global__` kernel。`nvcc` 負責 CUDA 編譯流程並搭配 host compiler；同一份 `.cu` 可以為不同 GPU 編譯，但 CPU 架構、工具鏈、GPU target 和外部 library 都需要一致的建置配置。[NVCC 文件](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-compiler-driver-nvcc/index.html)

建議讓公開介面只傳輸入描述、buffer、stream／執行上下文與結果，不向產品邏輯暴露 TensorRT engine 建置細節。NVIDIA／AMD 的記憶體管理與 kernel 放在各自實作內，較容易替換或比對。

值得使用 C++ 的典型觸發條件：

- CPU 前後處理占掉大部分端到端時間。
- 每筆輸入很小但呼叫頻率極高，語言／框架調度成本顯著。
- 需要與既有 C++ 相機、感測器或串流系統整合。
- 必須明確控制 buffer 回收、非同步執行與服務關閉順序。

這些條件並不構成硬即時保證；仍需量測排程、溫度、I/O 與最差延遲。

## Rust、Julia、Fortran 的邊界

**Rust：** `rust-cuda` 是可追蹤的開源 CUDA 工具生態，但 Rust kernel 工具鏈與 C/C++ ABI 封裝是不同方案。若團隊已有 Rust，建議先以 FFI 使用已驗證 CUDA／C++ 庫；若想全 Rust kernel，先針對 Orin 的 `aarch64` host、SM 8.7、編譯器與 CUDA 版本做小型驗證，再決定是否承擔工具鏈維護。不要把一般 Rust 能在 Arm64 執行視為 CUDA ecosystem 已全面驗證。[Rust-CUDA 專案](https://github.com/Rust-GPU/rust-cuda)

**Julia：** CUDA.jl 官方文件明列 Tegra／Jetson 路徑，也說明主要開發與 CI 平台為 x86_64 Linux。查核頁面的 JetPack 6 Orin 組合另標「not validated on hardware」，因此適合有 Julia 經驗的研究團隊，但不能省略目標板測試；CUDA.jl 可用也不代表其所有 cuDNN／cuTENSOR 等附加套件都在同一 Jetson 組合可用。[CUDA.jl 平台與 Jetson 說明](https://cuda.juliagpu.org/stable/installation/overview/)

**Fortran：** NVIDIA HPC SDK 26.9 列出 Arm server／SBSA 平台需求；這不是 AGX Orin 的直接支援承諾。既有 Orin／JetPack 6 問題中 NVIDIA 人員也指出 HPC SDK 不支援 Jetson。因此本研究不將 CUDA Fortran／NVHPC 當作 Orin 的預設生產路徑。可保留 CPU Fortran，將 GPU 部分封裝為 C ABI，或在明確支援的 NVIDIA 伺服器執行原 HPC 工作負載；新 JetPack 分支仍需逐項重新確認。[HPC SDK 26.9 平台需求](https://docs.nvidia.com/hpc-sdk/archive/26.9/release-notes/index.html)；[NVIDIA Orin 支援回覆](https://forums.developer.nvidia.com/t/problems-with-nvfortran-on-jetson-agx-orin/323324)

## 版本選擇與驗收

查核時 JetPack 下載頁列出 **7.2.1 支援 Orin family**，並搭配 CUDA 13.2.2、TensorRT 10.16.2。JetPack 6.2.1 的既有專案需保留自己的鎖定環境；語言工具、PyTorch wheel、ROS 與 camera driver 的升級應一起驗證。不能把「支援 Orin」延伸成任何 Python 或 Arm64 套件都相容。[JetPack 下載頁快照](https://developer.nvidia.com/embedded/jetpack/downloads)

建議驗收清單：相同測試資料的誤差、暖機後 p50／p95／p99 延遲、冷啟動、峰值記憶體、CPU 使用率、功耗模式與長時間熱降頻。比較語言時保持模型、精度、batch、輸入尺寸與 GPU kernel 路徑相同，否則測到的常是後端差異。更多建置與部署限制見 [跨設備可攜性](04-portability.md)。
