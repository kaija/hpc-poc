# AMD 硬體如何進行加速運算

研究日期：2026-10-25（Asia/Taipei）。本篇區分 GPU、APU 內的 GPU、NPU、CPU 與可程式化 SoC；它們不是安裝同一套軟體就能互換的運算裝置。

## 先選運算單元，再選軟體

| 硬體類別 | 適合的工作（工程建議） | 主要開發路徑 | 不能直接假設的事 |
| --- | --- | --- | --- |
| AMD Instinct GPU | 大模型訓練／推論、科學計算、資料中心多 GPU | ROCm、HIP C++、PyTorch ROCm | 任一 Instinct 都受最新版支援；資料中心卡適合低功耗無風扇邊緣盒 |
| Radeon／Radeon PRO 獨立 GPU | 工作站推論、影像運算、數值分析 | 精確符合支援矩陣時使用 ROCm／HIP | 所有 Radeon、筆電版、相同 RDNA 世代都支援同一套工具 |
| Ryzen APU 的 Radeon iGPU | 小型邊緣電腦、共享記憶體下的本地 AI | 受支援型號使用 ROCm GPU 路徑 | Ryzen 品牌、內顯存在或記憶體大，就代表 ROCm 可用 |
| Ryzen AI 的 XDNA NPU | 合適模型的低功耗推論、常駐感知 | Ryzen AI Software、模型量化／編譯與 runtime | HIP kernel 能跑在 NPU；所有 ONNX operators 都能卸載 |
| EPYC／Ryzen CPU | 控制流程、小批次運算、CPU fallback、CPU 數值工作 | C/C++、Python、AOCL 等 CPU 函式庫 | AMD CPU 需要 ROCm 才能運算 |
| Versal／Zynq／Kria 等 adaptive SoC／FPGA | 特殊感測 I/O、硬體資料流、嵌入式推論 | 對應硬體世代的 Vitis／Vitis AI／XRT 流程 | ROCm 或 Ryzen AI 的模型編譯產物可直接使用 |

ROCm 官方矩陣分別列出 Instinct、Radeon、Ryzen 的組合；Ryzen AI 與 Vitis AI 各有不同工具鏈。上表的選型建議是根據這些產品定位所作的工程判斷，並非效能實測。[ROCm 相容矩陣](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/latest/compatibility/compatibility-matrix.html)、[Ryzen AI 安裝](https://ryzenai.docs.amd.com/en/latest/inst.html)、[Vitis AI](https://www.amd.com/en/developer/resources/vitis-ai.html)、[AOCL](https://www.amd.com/en/developer/aocl.html)。

## 版本與支援範圍

本篇的 ROCm 版本基線為 **10.1.0**，官方固定版 release notes 標示 **2026-10-05**。這是文件查核，不是本機安裝或硬體驗證。其他 `latest` 連結為查閱時內容，後續可能更新。[ROCm 10.1.0 release notes](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)。

購買或部署前，保存以下相容組合，不能只寫「ROCm 支援 AMD」：

```text
產品完整 SKU + GPU gfx target + 作業系統版本 + kernel
+ GPU driver + ROCm 版本 + library/framework 版本
+ Python 版本／wheel 架構 + 必要工具功能
```

查核時，先選硬體家族與確切型號，再選 OS；最後查 framework 與 library 的個別支援。相同 `gfx` target 是有用線索，但不能代替官方對產品 SKU 的認證。[ROCm 相容矩陣](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/latest/compatibility/compatibility-matrix.html)。

| 環境 | 本篇採用的判讀 |
| --- | --- |
| 原生 Linux | 優先作為 HPC、profiling、多 GPU 的評估環境；仍須配對硬體與發行版／kernel |
| 原生 Windows | 不能再概括成「沒有 ROCm／不能跑 PyTorch」；必須依當版官方套件與 framework 指引選擇。10.1.0 的 HIP、許多數學庫及 MIOpen 列有 Windows 支援 |
| WSL2 | 10.1.0 安裝指南標為 Technical Preview；不是原生 Linux 完整功能的替代品 |

10.1.0 元件表中的 RCCL、ROCgdb、ROCprofiler-SDK 等仍列為 Linux；各 GPU 的工具覆蓋不同。WSL2 指南另外明列 profiling、GPU debugging 與依賴 KFD 的工具不可用。[10.1.0 元件與平台表](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html#rocm-core-sdk-components)、[10.1.0 安裝與 WSL 限制](https://rocm.docs.amd.com/en/docs-10.1.0/install/rocm.html)。

**實務建議：**初次評估選擇官方明列的 Linux + GPU + ROCm + PyTorch 組合。若產品必須使用 Windows，先用同一模型驗證所需 operators、自訂 extensions、profiling 與部署工具，再決定是否採用。

### 具體硬體例子

下列 SKU 均出現在 ROCm 10.1.0 的官方硬體支援名單，可作為評估起點；用途欄是工程建議。

| 具體產品 | 官方列出的 GPU target | 可評估的用途 | 本篇確認範圍 |
| --- | --- | --- | --- |
| Instinct MI300X | `gfx942`／CDNA 3 | 機房大型模型與 HPC；不視為 Orin 同功耗級替代品 | 已確認 10.1.0 硬體名單；尚未選定主機 OS／driver／framework 組合 |
| Radeon RX 7900 XTX | `gfx1100`／RDNA 3 | 桌機 AI 推論、影像與數值工作站 | 已確認 10.1.0 硬體名單；原生 Linux、Windows、WSL 的功能不可混為一談 |
| Ryzen AI Max+ 395（Radeon 8060S） | `gfx1151`／Strix Halo、RDNA 3.5 | 小型電腦上的本地 AI／共享記憶體 GPU 運算 | 已確認 10.1.0 的 **iGPU** 名單；NPU 需另查 Ryzen AI Software |

這不是三套已驗證的部署組合：本次讀取的互動矩陣未提供完整「指定 SKU × OS × driver × framework」選擇結果，因此沒有把分開列出的硬體表與 OS 表直接組合成相容保證。選定產品後，仍須補齊上面的精確組合與實機驗收。[ROCm 10.1.0 硬體支援名單](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html#amd-hardware-support)、[精確組合查核入口](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/latest/compatibility/compatibility-matrix.html)。

## CUDA library 對應到 AMD 的路徑

下表是功能對照與移植起點，**不代表 ABI 相容，也不保證 API、資料型別、演算法或速度完全相同**。

| NVIDIA 元件 | AMD 對應方向 | 用途與注意事項 |
| --- | --- | --- |
| CUDA Runtime／CUDA C++ | HIP Runtime／HIP C++ | GPU 記憶體、stream、event、自訂 kernel；需重新編譯與測試 |
| cuBLAS | hipBLAS／rocBLAS | 密集線性代數；hipBLAS 明列 rocBLAS 與 cuBLAS backend |
| cuBLASLt | hipBLASLt | 可調整 layout、演算法等的矩陣乘法；逐項確認資料型別與 epilogue |
| cuFFT | hipFFT／rocFFT | FFT；重新驗證 transform layout、batch 與數值誤差 |
| cuRAND | hipRAND／rocRAND | 亂數；不可假設相同 seed 就有逐位相同的序列 |
| cuSPARSE | hipSPARSE／rocSPARSE | 稀疏矩陣計算；格式與 API 覆蓋需核對 |
| cuSOLVER | hipSOLVER／rocSOLVER | 線性代數求解與分解 |
| CUB／Thrust | hipCUB／rocPRIM／rocThrust | reduction、scan、sort 等平行 primitives |
| cuDNN | MIOpen | 神經網路 primitives；不是替換動態庫檔名即可使用 |
| NCCL | RCCL | 多 GPU／多節點 collective communication；不等於可直接混用 NVIDIA 與 AMD 的單一 collective 群組 |
| TensorRT | MIGraphX／ONNX Runtime 的 MIGraphX EP | 重新匯入／編譯模型；不能讀取 TensorRT engine 當通用模型 |
| NPP／部分視覺前處理 | RPP 或 HIP 自訂實作 | 功能範圍不同，需要依演算法重新選 API |

數學庫名稱與分工依 [ROCm Core SDK 元件](https://rocm.docs.amd.com/en/latest/components/core.html)；hipBLAS 的 backend 行為依 [hipBLAS 文件](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/)。神經網路與通訊分別參考 [MIOpen](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)、[RCCL](https://rocm.docs.amd.com/projects/rccl/en/latest/)。推論 compiler 參考 [MIGraphX](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/)、[ONNX Runtime MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)。

`hip*` 與 `roc*` 名稱可幫助理解抽象層，但仍要看各函式庫版本。不要由 hipBLAS 的雙 backend 支援推論「每個 HIP library、每個 OS、Jetson ARM64 都有同等套件」。例如目前 hipCUB 首頁的描述較聚焦 rocPRIM，而總覽頁仍描述 CUB／rocPRIM；採用前應以指定版本的 build 文件和實際 CI 為準。[hipCUB 文件](https://rocm.docs.amd.com/projects/hipCUB/en/latest/)。

## 如何開始使用 AMD GPU

### Python：先讓實際模型跑起來

安裝與硬體相符的官方 ROCm PyTorch 套件／容器後，先確認 backend，再做小型計算。**下面是提供給目標機執行的 smoke test，本研究未在 AMD 硬體實測。**

```python
import torch

assert torch.version.hip is not None, "目前不是 ROCm 版 PyTorch"
assert torch.cuda.is_available(), "ROCm GPU 尚不可用"
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("Device:", torch.cuda.get_device_name(0))

x = torch.arange(4096, dtype=torch.float32, device="cuda")
y = x * 2 + 1
torch.cuda.synchronize()
expected = torch.arange(4096, dtype=torch.float32) * 2 + 1
torch.testing.assert_close(y.cpu(), expected)
print("GPU arithmetic check passed")
```

ROCm 版 PyTorch 刻意沿用 `torch.cuda` 與 `device="cuda"` API 名稱；這裡的字串不代表它正在執行 NVIDIA CUDA binary。用 `torch.version.hip` 辨識 HIP backend。[PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html)。

工程驗收建議：此測試通過後，繼續執行真實模型、驗證輸出品質、記錄 warm-up 後的 P50／P95 延遲與顯示記憶體峰值。不要把「成功列出 GPU」當作應用相容性完成。

### C++：需要自訂 GPU kernel 時使用 HIP

HIP 是 C++ runtime API 與 kernel 語言。一般流程為：CPU 配置／準備資料 → GPU buffer → 啟動 kernel／library → 同步 → 讀回或交給下一個 GPU stage。AMD 官方提供 HIPIFY 將許多 CUDA 呼叫改寫為 HIP；仍需要人工修正與效能調校。[What is HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html)。

安裝完成後，可在**原生 Linux 目標機**先檢查工具鏈；安裝前綴與 PATH 依套件而異：

```bash
rocminfo
hipcc --version
# compute.hip 是自己的 HIP 原始碼；以下只是編譯形式。
hipcc -O3 compute.hip -o compute
./compute
```

跨機建置或產生多個 GPU 目標時，依該版 compiler 文件設定對應 `gfx` target；不要把為某張卡建置的 binary 視為跨所有 AMD GPU 的保證。`rocminfo` 可協助確認 runtime 偵測到的 GPU；HIP 編譯入口可使用 `hipcc` 或官方說明的 `amdclang++`。[ROCm 安裝驗證](https://rocm.docs.amd.com/en/docs-10.1.0/install/rocm.html)、[HIP 移植指南](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)。

### CPU：不一定要上 GPU

AOCL 提供面向 AMD CPU 的 BLAS、LAPACK、FFT、數學與其他函式庫。對小輸入、分支密集或資料搬移佔比高的工作，**建議先量 CPU baseline，再決定是否搬到 GPU**；這是工程評估原則，不代表 AOCL 一律較快。[AMD Optimizing CPU Libraries](https://www.amd.com/en/developer/aocl.html)。

## Ryzen AI NPU：與 ROCm GPU 分開評估

Ryzen AI Software 的 NPU 流程以模型準備、量化／編譯、runtime 執行為核心。Windows 安裝頁與 Linux 安裝頁是分開的；查閱時頁面版本為 **Ryzen AI Software 1.8.0**。Linux 文件明列較新 consumer APU 從 Strix Point 起、embedded APU 從 Ryzen AI Embedded P100 起的支援範圍，因此不能說「Ryzen AI 只能 Windows」，也不能說「所有舊 Ryzen AI 的 NPU 都支援 Linux」。[Windows 安裝](https://ryzenai.docs.amd.com/en/latest/inst.html)、[Linux 安裝](https://ryzenai.docs.amd.com/en/latest/linux.html)。

實作建議：

1. 確認 CPU 完整型號與 NPU 世代；另外確認 iGPU 是否受 ROCm 支援。
2. 選 NPU software、OS 與 driver 的匹配組合，先跑官方 quick test。
3. 用目標模型檢查 operator、shape、量化精度與編譯支援。
4. 量測整條 pipeline；CPU fallback、資料轉換與 NPU 編譯快取都算進產品成本。
5. 預留 CPU／GPU fallback，但不可把 fallback 成功誤認為 NPU 加速成功。

Linux 1.8.0 文件提供 XRT／NPU driver 安裝與 `xrt-smi examine` 驗證；此指令是 NPU 路徑，不能拿 `rocminfo` 取代。[Ryzen AI Linux 指南](https://ryzenai.docs.amd.com/en/latest/linux.html)。

## 嵌入式 AMD：Versal、Zynq、Kria 不等於 Radeon

Vitis AI 的工具鏈包含量化、compiler、NPU IP、runtime 與參考設計；查閱時官方頁面的 General Access 聚焦 Versal AI Edge 與 AI Edge Gen 2，舊 DPU 技術另有 legacy 文件。Zynq／Kria 的專案必須查自己的板卡、DPU／NPU IP 與工具版本，不可直接套用最新 Versal 的支援說明。[Vitis AI Developer Hub](https://www.amd.com/en/developer/resources/vitis-ai.html)。

工程選型上，若需求是特殊 camera／工業介面、固定資料流或可程式化硬體加速，可評估此路線；若主要是熟悉的 Python AI／通用 GPU kernel，先評估受支援的 Radeon／Ryzen GPU 或 Jetson，通常較接近既有軟體工作方式。此處是開發成本判斷，沒有聲稱哪一款產品效能較好。

## 從 NVIDIA 移植到 AMD，哪些能保留？

| 資產 | 可保留程度 | 必要工作 |
| --- | --- | --- |
| 純 Python 業務邏輯、標準 PyTorch ops | 通常較高 | 換成匹配的 ROCm build；測試所有模型 operators |
| 原始模型權重、標準 ONNX graph | 模型層可重用 | 重新選 EP、處理不支援 ops、校正與編譯 |
| CUDA C++ 自訂 kernel | 可做來源層移植 | HIPIFY + 人工修正 + 重新編譯／效能調校 |
| cuBLAS／cuFFT 等 library 呼叫 | 有對應替換方向 | 核對 API、資料型別與計算結果 |
| TensorRT engine、自訂 TensorRT plugin | 不可當通用產物 | 從模型／原始碼重建 AMD backend |
| Jetson camera／DLA／DeepStream 裝置整合 | 平台相依性高 | 重新設計採集、媒體與推論整合層 |

移植 CUDA kernel 時尤其要處理 warp／wavefront 寬度、lane mask、內嵌 PTX、同步與資源配置假設。AMD 官方明確要求不要固定假設 `warpSize` 為 32 或 64；能編譯不代表演算法與效能已正確。[HIP 移植指南](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)。模型推論則使用 AMD graph compiler 或其 EP 的流程，並依套件支援表選版本。[MIGraphX](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/)、[MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)。

**建議的跨廠牌架構：**以 Python／C++ 定義相同的輸入、輸出與數值驗收；模型保留訓練來源與 ONNX；GPU 執行層分成 NVIDIA 與 AMD backend。先讓兩邊得到可接受的結果，再分別優化。容器、binary、engine cache 與硬體專用前處理模組分開打包。

## 尚待具體硬體與工作負載確認

- 使用者尚未指定 AMD SKU、OS、模型、吞吐量、延遲及功耗限制，本篇不能替某一台 AMD 電腦保證相容或效能。
- 沒有在 AMD GPU／NPU 進行 benchmark；文中的用途與推薦路徑是來源整理加工程分析。
- 採購前應補一份具體矩陣截取與實機驗收：精確 SKU、版本、模型／資料、品質、延遲、記憶體、整機功耗及長時間穩定性。
