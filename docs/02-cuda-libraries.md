# AGX Orin 可用的 CUDA 函式庫與加速 SDK

查核日期：2026-10-25。本文以 **Jetson AGX Orin + Jetson Linux** 為對象；各表的用途建議是工程判斷，平台與版本資訊則連結至 NVIDIA 官方資料。本次未在實機安裝或執行效能測試。

## 1. 先固定版本，再挑函式庫

CUDA 核心數學函式庫很適合在 Orin、工作站與伺服器共用原始碼；相機、DLA、PVA 與影片編解碼則要另外處理平台差異。不要只看套件名稱含 `arm64` 就安裝。

| 軟體層 | 歷史可重現基線：JetPack 6.2.1 | 查閱時官方下載頁：JetPack 7.2.1 |
|---|---|---|
| Jetson Linux | 36.4.4 | 39.2.1 |
| Ubuntu / Kernel | 22.04 / 5.15 | 24.04 / 6.8 |
| CUDA | 12.6；6.2.1 細項頁列 12.6.10 | 13.2.2 |
| cuDNN | 9.3 | 9.20.0 |
| TensorRT | 10.3 | 10.16.2 |
| VPI | 3.2 | 4.1.3 |
| Orin 支援 | 有 | 下載頁列出 Jetson Orin Family |

來源：[JetPack 6.2.1 元件說明](https://developer.nvidia.com/embedded/jetpack-sdk-621)、[查閱時 JetPack 下載頁](https://developer.nvidia.com/embedded/jetpack/downloads)。右欄是動態網頁快照，不能據此推定所有舊專案、相機驅動與附加 SDK 均能直接升級；6.2.1 也不代表目前最新版本。

以下核心 library 的 **Jetson 平台可用性以 CUDA 12.6 元件矩陣為明確證據**。CUDA 12.6 明確區分 `x86_64`、`arm64-sbsa`、`aarch64-jetson`；較新的統一 Arm 發行策略也不等於所有既有 Orin 軟體可混裝 SBSA 套件。[CUDA 12.6 元件及平台矩陣](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html)、[JetPack 7.2.1 平台說明](https://developer.nvidia.com/embedded/jetpack/downloads)

## 2. 核心 CUDA libraries：數值、訊號、影像與自訂運算

下表各項均列於 CUDA 12.6 的 `aarch64-jetson` 支援矩陣。它們屬於 CUDA Toolkit；是否已安裝 runtime、開發標頭或範例，仍取決於 JetPack 安裝選項。

| 函式庫 / 元件 | 能做什麼 | Orin 邊緣場景與選用判斷 |
|---|---|---|
| **cuBLAS / cuBLASLt** | 向量及稠密矩陣運算；Lt 提供較彈性的 GEMM 資料格式、型別與演算法選擇 | 大量矩陣乘法、特徵轉換、批次線性運算。先用 library，再考慮自寫 kernel。[API](https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html) |
| **cuFFT / cuFFTW** | 1D / 2D / 3D、實數與複數 FFT；cuFFTW 協助移植 FFTW 使用情境 | 雷達、聲音頻譜、振動異常偵測、影像頻域處理。重用 FFT plan 並比較批次大小。[API](https://docs.nvidia.com/cuda/archive/12.6.0/cufft/index.html) |
| **cuSPARSE** | 稀疏矩陣向量乘法、矩陣乘法、三角解法及格式轉換 | 稀疏圖計算、有限元素與線性運算子；資料稀疏程度和格式會影響效益。新程式優先評估 Generic API。[API](https://docs.nvidia.com/cuda/archive/12.6.0/cusparse/index.html) |
| **cuSOLVER** | LU、QR、Cholesky、SVD、特徵值及線性系統求解 | 點雲配準、最小平方法、狀態估測中的矩陣分解。大問題先估 workspace；小矩陣不必然比 CPU 快。[API](https://docs.nvidia.com/cuda/archive/12.6.0/cusolver/index.html) |
| **cuRAND** | CPU 呼叫及 device 端的偽亂數／準亂數產生 | 蒙地卡羅、粒子濾波、模擬採樣；固定 seed 與 generator，另驗證重現性。[API](https://docs.nvidia.com/cuda/archive/12.6.0/curand/index.html) |
| **NPP** | CUDA 加速影像與訊號基礎操作、色彩轉換、濾波等 | AI 前處理、影像校正及批次影像處理；避免每一步都搬回 CPU。[API](https://docs.nvidia.com/cuda/archive/12.6.0/npp/index.html) |
| **Thrust** | C++ 高階平行演算法，例如排序、scan、reduce、transform | 點雲整理、資料篩選與統計，適合先實作正確版本。[CCCL](https://github.com/NVIDIA/cccl) |
| **CUB** | device / block / warp 層級的平行運算原語 | 客製 kernel 的排序、scan、reduce 等熱點；比直接用 Thrust 更需理解執行與記憶體模型。[CCCL](https://github.com/NVIDIA/cccl) |
| **libcu++** | CUDA C++ 標準函式庫設施 | 同步與 device 端 C++ 元件；通常搭配自訂 CUDA kernel。[CCCL](https://github.com/NVIDIA/cccl) |
| **CUDA Runtime / Driver API、NVRTC、nvJitLink** | 裝置、stream、記憶體管理；執行期編譯與連結 | 自訂運算與框架整合；屬於基礎設施，不是現成影像或 AI 演算法。[元件表](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html) |
| **nvJPEG** | CUDA JPEG 處理 | JPEG 批次輸入候選路徑；不要把「函式庫存在」等同「所有專用 JPEG 硬體 backend 均支援 Orin」。[平台表](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html)、[backend 說明](https://docs.nvidia.com/cuda/archive/12.8.2/nvjpeg/index.html) |

「可在 GPU 上計算 FP64」與「適合高吞吐 FP64 HPC」不同。若需求是高精度科學計算，應以實際矩陣大小、精度、延遲與功耗測試選型，不由 INT8 TOPS 推算成果。

## 3. AI、視覺與多媒體：不全是 CUDA library

| 軟體 | 類型與基線供應方式 | 應用與限制 |
|---|---|---|
| **cuDNN** | 深度學習 primitives library；JetPack 6.2.1 包含 9.3 | 提供神經網路底層運算；一般由 PyTorch 或推論後端間接使用。Jetson 與 SBSA 在支援表中分列。[cuDNN 9.3 矩陣](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.3.0/reference/support-matrix.html) |
| **TensorRT** | 模型最佳化與推論 SDK；JetPack 6.2.1 包含 10.3 | 適合部署已訓練模型。模型算子、精度與 engine 建置環境需驗證；不是完整訓練框架。[JetPack 元件](https://developer.nvidia.com/embedded/jetpack-sdk-621)、[TensorRT 10.3 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.3.0.html) |
| **cuDLA / DLA 路線** | DLA 執行與互通 API；CUDA 12.6 列 Jetson 支援 | 用於 DLA loadable 與 GPU 協作，DLA 不是可任意執行 CUDA kernel 的另一顆 GPU。新 JetPack / TensorRT 組合須另外確認 DLA runtime，不能只看硬體有 DLA。[cuDLA API](https://docs.nvidia.com/cuda/archive/12.6.0/cudla-api/index.html) |
| **VPI** | 跨 CPU / CUDA / 專用視覺加速器的 library；基線包含 3.2 | 濾波、校正、光流、特徵追蹤、立體視差。每個演算法的 backend 不同；PVA / VIC / OFA 路線不能原樣搬去一般 RTX。[VPI 3.2 演算法矩陣](https://docs.nvidia.com/vpi/3.2/algorithms.html) |
| **Jetson Multimedia API / libargus / NVIDIA GStreamer plugins** | Jetson 平台的相機與媒體 API / plugin | CSI 相機、ISP、硬體影片編解碼、轉色與縮放。AGX Orin 的 JetPack 6.x 管線可由 `nvv4l2decoder`、編碼 plugin 及 `nvvidconv` 組合。[Jetson GStreamer](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html) |
| **DeepStream** | 基於 GStreamer 的影像分析 SDK；另行選擇相容版本安裝 | 多路串流、批次推論、追蹤與結果輸出。7.1 的官方矩陣列 JetPack 6.1 / CUDA 12.6；不能因 CUDA 同為 12.6，就宣稱對所有 6.2.x 組合均經官方驗證。[7.1 安裝與矩陣](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Installation.html) |
| **PyTorch** | 機器學習 framework；另外安裝相符 wheel 或容器 | 研究、模型驗證與部分微調。匹配 JetPack、Python ABI、CUDA、cuDNN；桌面 x86 wheel 不能給 Jetson 使用。[NVIDIA Jetson 安裝指引](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/) |

**DLA 版本提醒：** 本文確認的是 6.2.1 配套含 DLA 的歷史基線。查閱時最新 TensorRT 文件對 Jetson / DLA 有個別版本限制，舊文件 URL 亦可能重導向最新版本；實作前必須讀目標版本矩陣並在實機測試 DLA engine，不能套用最新頁對所有 10.x 的描述。[TensorRT 官方文件與版本說明](https://docs.nvidia.com/deeplearning/tensorrt/latest/)

## 4. 需另查條件，不應當作 Orin 預設可用

| 項目 | 為何不能直接承諾 | 建議 |
|---|---|---|
| **NCCL** | 用於多 GPU / 多節點 collective；通用 NCCL 專案說明不能取代特定 Jetson、網路與框架版本的測試 | 單台 Orin 通常無需使用。多 Orin 專案先驗證 build、transport 與 `nccl-tests`，不預設具有 NVLink 或伺服器級互連。[NCCL](https://developer.nvidia.com/nccl)、[原始碼](https://github.com/NVIDIA/nccl) |
| **cuSPARSELt** | 和 cuSPARSE 不同，是結構化稀疏矩陣乘法庫；需另裝且發行平台策略曾改變 | 某些 NVIDIA PyTorch for Jetson 套件需要它；依該框架安裝指引固定版本。v0.9.0 release notes 指出專用 Jetson 平台支援終止、Orin / Thor 應轉 Arm64 SBSA；這不是允許舊 JetPack 任意混裝的保證。[PyTorch 安裝](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/)、[cuSPARSELt release notes](https://docs.nvidia.com/cuda/cusparselt/release_notes.html) |
| **cuTENSOR、cuQuantum、RAPIDS、TensorRT-LLM、NVSHMEM / Mp 系列** | 各自有硬體、CUDA、OS、封裝與版本矩陣；名稱含 CUDA 或支援 Arm 不能證明此 Orin 配置受支援 | 本研究未逐一確認，故不列入可直接採用清單；若應用確實需要，再查目標 release 的 Jetson / SM / OS 支援，另做 PoC |
| **cuFile / GPUDirect Storage** | CUDA 元件表存在 aarch64-jetson 套件，不代表儲存堆疊、檔案系統和資料路徑都滿足 direct I/O 條件 | 不由「套件可安裝」推論「可零拷貝」。先以普通 I/O 測試，再核對完整產品支援與部署需求。[CUDA 元件表](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html) |
| **Video Codec SDK 與 Jetson Multimedia API** | API 與平台版本不同；新版 Video Codec SDK 已有 Thor 支援，也不代表舊版 Orin 管線自動相容 | Orin 6.x 優先走該版 Jetson Multimedia API / GStreamer；跨 RTX 則隔離媒體後端。[Jetson 文件](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html)、[Video Codec SDK](https://developer.nvidia.com/video-codec-sdk) |

## 5. 哪些可以跨 NVIDIA 裝置共用？

| 共用對象 | 可攜程度 | 要重做的部分 |
|---|---|---|
| cuBLAS、cuFFT、cuSPARSE、cuSOLVER、cuRAND、NPP、Thrust / CUB 原始碼 | 高，前提是目的平台的該版本支援 GPU 與 API | 分別為 CPU 架構及目標 GPU 編譯、連結相符 library，重新調校批次與記憶體配置 |
| PyTorch 模型與大部分張量程式碼 | 高於直接依賴平台 API 的程式碼 | 換相符套件 / backend；自訂 CUDA extension、特定精度與算子另驗證 |
| TensorRT 模型建置流程 | 中至高 | 以各目標環境重建 engine；不把 `.engine` 當通用模型格式 |
| VPI CUDA / CPU backend | 部分可共用 | 核對目的平台發行版本與演算法；改寫僅有專用引擎的路線 |
| libargus、DLA、PVA、Jetson 特有媒體 buffer 管線 | 低至中 | 以平台 adapter 替換相機、媒體與加速器；保留上層演算法與資料契約 |

以上是依官方平台表和 API 分工做的工程判斷。Tegra CPU 與 iGPU 共用實體 DRAM，離散 GPU 常有自己的 VRAM；共用 API 不代表兩者最適合相同搬移策略。[CUDA for Tegra](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-for-tegra-appnote/index.html)

## 6. 實際選擇順序與驗證

建議先依需求選一條最短管線：

1. **多攝影機 AI：** 相容 DeepStream + TensorRT；必要時加入 VPI / NPP 前處理。
2. **單模型推論：** Python 驗證模型 → TensorRT 建置 → C++ 或 Python 執行；先確認端到端延遲再決定改寫。
3. **雷達、音訊、振動：** cuFFT + cuBLAS / 自訂 CUDA；低批量先和 CPU 比較。
4. **稀疏 / 數值運算：** cuSPARSE / cuSOLVER；先確認資料規模、精度與 workspace。
5. **客製資料整理：** Thrust → 必要時 CUB → 最後才自寫融合 kernel。

下列僅是待於 Jetson 執行的唯讀盤點命令，**本研究未執行**。SDK Manager 的安裝選項及 runtime / development 套件差異，可能使部分命令或標頭不存在。

```bash
cat /etc/nv_tegra_release
uname -m
/usr/local/cuda/bin/nvcc --version
dpkg-query -W 'nvidia-jetpack*' 'cuda-*' 'libcublas*' 'libcufft*' 'libcusparse*' 'libcusolver*' 'libcurand*' 'libnpp*' 'libcudnn*' 'libnvinfer*' 'libnvvpi*'
apt-cache depends nvidia-jetpack
```

取得版本後，再從對應版本的官方 samples 選一個代表性運算，檢查輸出誤差、持續負載延遲、最大記憶體、溫度與功耗。不能以 `nvcc` 存在證明 GPU kernel 可執行，也不能以 import 成功證明推論沒有退回 CPU。

安裝方法可參考 [JetPack 6.2.1 安裝文件](https://docs.nvidia.com/jetson/jetpack/6.2.1/install-setup/index.html)。該頁部分套件清單仍含早期 `12-2` 範例，因此查詢實機套件與正式 release notes 比照抄套件版本更可靠。
