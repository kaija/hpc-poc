# Jetson AGX Orin：邊緣運算能力與適用場景

查核日期：2026-10-25（Asia/Taipei）。本文將「NVIDIA AGX Orin」解讀為 **Jetson AGX Orin**，不是採用 DriveOS 的 DRIVE AGX Orin。尚未取得實際模組型號、RAM、載板及散熱條件，因此不承諾 FPS、攝影機路數或 LLM tokens/s。

## 1. 適合怎麼用

AGX Orin 適合把攝影機、感測器與 AI 推論放在現場處理：機器視覺、機器人感知、多路影片分析、語音介面，以及受記憶體與延遲限制的本地生成式 AI。它也能執行一般 CUDA 數值運算；選用理由應是現場 I/O、功耗與整合需求，而非把它視為大型訓練伺服器的等效替代品。前半是官方定位，後半為本研究的工程選型建議。[NVIDIA Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

| 工作負載 | 可實作內容 | 建議起點 | 首要驗證項目 |
| --- | --- | --- | --- |
| 工業機器視覺 | 瑕疵檢測、物件分類、OCR、分割、尺寸量測前處理 | TensorRT + VPI／NPP；多路管線用 DeepStream | 曝光與光源、解析度、漏檢率、每件工件端到端延遲 |
| 多路影片分析 | 偵測、追蹤、人流統計、事件摘要 | 硬體解碼 + DeepStream + TensorRT | 解碼與推論是不同負載；測同時路數、掉幀、事件延遲 |
| 機器人／AMR | 視覺里程計、SLAM、深度重建、障礙物地圖 | ROS 2 + 相容版 Isaac ROS Visual SLAM／nvblox | 感測器同步、定位誤差、地圖記憶體、感知更新率 |
| 訊號處理 | FFT 頻譜、濾波、雷達前處理、振動異常分析 | cuFFT + cuBLAS + 自訂 CUDA kernel | 輸入批次是否夠大、資料搬移、數值誤差 |
| 語音與本地助理 | ASR、TTS、喚醒詞、語音問答 | 支援 Jetson 的模型 runtime；部分工作可用 TensorRT | 串流首回應延遲、同時跑視覺時的資源競爭 |
| LLM／VLM／RAG | 本地文件問答、影像問答、機器狀態摘要 | 小模型／量化模型；核對 TensorRT Edge-LLM 或其他引擎的 Orin 支援 | 模型權重 + KV cache + workspace；首 token 與持續產生速度 |
| 現場數值計算 | 矩陣乘法、稀疏解算、蒙地卡羅、批次最佳化 | cuBLAS、cuSPARSE、cuSOLVER、cuRAND | FP32／FP64需求、問題規模、實測是否勝過CPU |
| 輕量微調／資料前處理 | 小型模型實驗、少量更新、現場資料篩選 | 相容版 PyTorch | 訓練記憶體與耗時；大型訓練優先放工作站或伺服器 |

這張表是基於能力的應用建議，不是每個模型的認證清單。函式庫依據見 [CUDA 函式庫](02-cuda-libraries.md)。機器人的具體官方實作見 [Visual SLAM 3.2](https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html) 與 [nvblox 3.2](https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_nvblox/index.html)；後者將深度與姿態轉成 3D 重建及導航 costmap。

## 2. 硬體能力要分開看

以下以官方 Technical Brief v1.2 的標準量產 32GB／64GB 模組為規格基準。早期 Developer Kit、Industrial SKU 與新版功率設定可能不同，採購或量測必須以實際料號及其資料表為準。

| 項目 | AGX Orin 32GB | AGX Orin 64GB |
| --- | --- | --- |
| GPU | Ampere，1,792 CUDA cores、56 Tensor Cores | Ampere，2,048 CUDA cores、64 Tensor Cores |
| CPU | 8 核 Arm Cortex-A78AE | 12 核 Arm Cortex-A78AE |
| 記憶體 | 32GB LPDDR5，共享系統記憶體 | 64GB LPDDR5，共享系統記憶體 |
| 標稱記憶體頻寬 | 204.8 GB/s | 204.8 GB/s |
| 專用加速器 | 2 個 NVDLA v2、PVA v2 | 2 個 NVDLA v2、PVA v2 |
| 該版規格的功耗範圍 | 15–40W | 15–60W |
| 標稱 AI 峰值 | 200 sparse INT8 TOPS | 275 sparse INT8 TOPS |

來源：[NVIDIA Technical Brief v1.2，規格表與 GPU／DLA 章節](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)。功耗欄是模組規格，不能當成含相機、NVMe、載板與電源損耗的整機牆上功耗。

### 不要把 275 TOPS 當成所有程式的速度

64GB 版本的 275 sparse INT8 TOPS 是 GPU 與 DLA 的合計峰值，技術簡報分別列 GPU 170、DLA 105；一般 CUDA FP32 計算是另一種指標。稀疏模型須滿足指定的結構化稀疏條件，不能把任何 dense 模型、FFT 或 FP64 解算器直接換算成這個數字。[Technical Brief，GPU、Sparsity、DLA](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)

本研究建議：比較裝置時先固定模型、精度、輸入大小、batch、精度門檻與功耗模式，再比較端到端延遲和每瓦吞吐量。未測量前，不能從 NVIDIA／AMD 的 TOPS 廣告數字推算實際優劣。

### GPU、DLA、PVA 與影片引擎的分工

| 單元 | 適合工作 | 軟體入口與限制 |
| --- | --- | --- |
| CPU | I/O、通訊、流程控制、不規則與小型工作 | C++、Python；小任務不一定值得送GPU |
| CUDA GPU／Tensor Cores | 一般平行運算、深度學習、矩陣計算 | CUDA、數學庫、cuDNN、TensorRT；Tensor Cores須符合運算和資料型別條件 |
| DLA | 可映射到其支援算子的神經網路推論 | 舊版可確認路線為 TensorRT FP16／INT8；它不是可執行任意CUDA kernel的GPU |
| PVA／VIC | 支援的視覺前處理、轉換等 | VPI或Jetson多媒體介面；依演算法支援表選backend |
| NVDEC／NVENC／ISP | 影片解碼、編碼、相機影像處理 | Jetson多媒體與相機API；codec、位元深度、解析度需另查 |

DLA 的硬體存在不等於每版 JetPack 都提供同樣 runtime。本研究以 JetPack 6.2.1 的 DLA 3.1 組合作為已記錄基線；7.x 必須再核對該版發行說明與實際套件。GPU、PVA與影片API細節見 [函式庫分層](02-cuda-libraries.md)。[JetPack 6.2.1 Release Notes](https://docs.nvidia.com/jetson/jetpack/6.2.1/release-notes/index.html)

## 3. 記憶體是主要設計條件

Orin 的 CPU 與 GPU 共用 SoC DRAM；64GB 並不是「64GB RAM 再加 64GB VRAM」。CUDA device、host、managed allocations 的存取與快取行為仍不同，不能因共用實體記憶體就移除同步、直接共享任意指標，或假設資料格式轉換沒有成本。[CUDA 12.6 for Tegra：Memory Management](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-for-tegra-appnote/index.html)

模型容量可先作粗估：

```text
未含額外開銷的權重大小 ≈ 參數數量 × 每參數位元數 / 8
7B × 4 bit / 8 ≈ 3.5 GB（十進位）

實際需求 = 權重 + 量化metadata + KV cache + activation/workspace
         + 影片buffer + 其他模型 + OS與應用程式
```

這是容量算式，不是 7B 模型的實測記憶體或效能。長 context、並行請求、多相機與 VLM 影像輸入會增加需求；swap 也不能當成具有相同延遲與頻寬的額外 GPU 記憶體。

生成式 AI 的新路線可參考 TensorRT Edge-LLM。查閱時官方矩陣列出 **Orin + JetPack 7.2 + CUDA 13.2**，Orin runtime precision 為 FP16／INT8／INT4，不支援 FP8／FP4 model engines；不能照搬 Thor 範例的精度選項。7.2.1 或其他小版本仍應以所選 release 的矩陣與安裝測試確認。[TensorRT Edge-LLM Support Matrix，頁面標示更新於2026-09-29](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html)

## 4. 現場部署的限制

以下為工程判斷與驗證建議：

- **功耗與散熱**：短測峰值不能代表密閉機箱中的長時間效能；用部署功耗模式做持續負載測試。
- **相機與I/O**：SoC規格、載板接頭、感測器驅動和可用頻寬是不同層次；先確認相機模組及同步方式。
- **硬即時控制**：感知與決策適合放Orin；馬達保護、急停或嚴格截止時間的閉環，應由另行驗證的控制器承擔。
- **HPC數值需求**：能跑FP64不代表適合大量FP64科學模擬；先測實際問題及誤差，再選工作站或資料中心GPU。
- **多節點**：多台Orin可以分工處理相機或不同服務；不能把它們的RAM當成一張透明的大顯示卡，也不能假設有資料中心NVLink／NCCL的部署條件。
- **離線運作**：模型、tokenizer與必要資產預先放在設備上；用斷網測試驗證服務真的不依賴雲端。

功率模式、時脈與溫控的實際機制見 [Jetson Linux r36.4.4 Power and Performance](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)；這份文件可用於6.x基線，不應直接套用其模式ID到所有SKU或7.x。

## 5. 建議先做的三個PoC

1. **單路視覺**：固定影片輸入，測前處理 + TensorRT推論 + 後處理；先驗證正確率，再增加路數。
2. **FFT或矩陣運算**：拿實際感測器資料，對照CPU、CUDA library與端到端耗時，確認GPU加速是否有收益。
3. **本地小型LLM／VLM**：固定模型版本、量化與context，記錄首token、tokens/s、峰值RAM，並測與視覺服務共存。

通過後再決定是否需要DLA、C++熱路徑、跨NVIDIA或AMD後端。執行與量測清單見 [06：驗證計畫](06-validation-plan.md)。
