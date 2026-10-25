# 選型、安裝與實機驗證計畫

查核日期：2026-10-25。這是建議的實驗流程；本研究未連線到Orin或AMD設備，以下指令與模型效能均未在目標硬體執行。只完成文件研究及文件本身的檢查。

## 1. 選定軟體組合再安裝

| 目標 | 先選擇的條件 | 應保留的產物 |
| --- | --- | --- |
| Orin視覺／機器人 | 相機驅動、ROS／Isaac ROS／DeepStream與JetPack的交集 | BSP版本、套件清單、容器digest、相機設定 |
| Orin生成式AI | 引擎release、Orin支援、模型格式／精度與JetPack交集 | 原始模型版本、量化設定、tokenizer、建置紀錄 |
| NVIDIA工作站／伺服器 | GPU compute capability、driver、CUDA、框架版本 | 每種CPU架構的build、每類GPU的engine |
| AMD GPU／APU | 精確SKU、gfx target、OS／kernel、ROCm及框架矩陣 | GPU識別、driver／ROCm版本、HIP build與測試結果 |
| AMD NPU | Ryzen AI處理器型號、NPU driver、Ryzen AI Software與模型支援 | ONNX、量化／編譯設定、NPU分派紀錄 |

不要只因套件版本數字更大就升級。查閱時NVIDIA下載頁列7.2.1支援Orin，但Isaac ROS欄仍為「Coming soon」；相對地，Isaac ROS 3.2的nvblox文件明列JetPack 6.1／6.2。需要機器人套件時應按其發行分支的矩陣選版本，而不是混裝這兩套環境。[JetPack下載表](https://developer.nvidia.com/embedded/jetpack/downloads)、[Isaac ROS 3.2平台矩陣](https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_nvblox/index.html)

AMD的安裝與硬體矩陣入口、框架驗證程式見 [05：AMD運算](05-amd-compute.md)。本文件不提供一條假設所有AMD裝置都適用的安裝指令。

## 2. Orin環境盤點

以下於**Jetson終端機**執行，不是在開發用Mac或x86主機。屬於6.x常見盤點方式；若換7.x，依該版工具文件核對路徑及參數。

```bash
uname -m
cat /etc/os-release
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core nvidia-jetpack
nvcc --version
python3 --version
sudo /usr/sbin/nvpmodel -q
```

解讀方式：

- `aarch64`確認CPU架構；仍需另外辨識Orin SKU。
- `nvidia-jetpack`是套件層的線索；精簡映像可能沒有此meta-package，不能僅因查無套件就判定沒有CUDA。
- `nvcc`是compiler；runtime-only容器可能沒有它，應另查runtime與framework是否能用GPU。
- `nvpmodel -q`只查當前模式；不要把其他SKU的模式ID複製過來更改。

`nvpmodel`依據：[Power Mode Controls](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)。SDK與L4T版本配對依據：[JetPack Archive](https://developer.nvidia.com/embedded/jetpack-archive)。

若已裝相容版PyTorch，可用以下最小檢查確認GPU可見、實際執行矩陣乘法並同步：

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("HIP build:", torch.version.hip)
assert torch.cuda.is_available(), "GPU不可用：檢查驅動、框架build與裝置權限"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((512, 512), device="cuda", dtype=torch.float32)
y = x @ x
torch.cuda.synchronize()
assert torch.allclose(y, torch.full_like(y, 512.0))
print("GPU matmul smoke test passed")
```

這不是效能benchmark。PyTorch ROCm版本也沿用`torch.cuda`介面；辨識AMD後端要看`torch.version.hip`，不能看變數名稱猜硬體。安裝適配與來源見 [語言選擇](03-language-selection.md) 與 [AMD運算](05-amd-compute.md)。

## 3. 四層驗證

1. **硬體可見**：檢查GPU／NPU型號與driver；CUDA／HIP跑最小運算。
2. **功能正確**：以CPU或已驗證模型作參考，檢查數值容許誤差、偵測正確率或任務品質。
3. **確實加速**：查看profiler、runtime log與算子placement，排除非預期CPU fallback；GPU利用率本身不足以證明整個模型都在GPU。
4. **部署可用**：同時開啟相機、網路、磁碟與其他模型，測長時間延遲、記憶體、功耗與熱降頻。

這是本研究提出的驗證方法；測試門檻要由實際產品需求決定。

## 4. 最小benchmark紀錄表

| 欄位 | 記錄內容 |
| --- | --- |
| 硬體 | 精確SKU、RAM、載板、散熱、電源、儲存 |
| 軟體 | OS、kernel、driver／BSP、CUDA／ROCm、library／framework、容器digest |
| 工作負載 | 模型或演算法版本、資料集、解析度、batch、context、並行數 |
| 精度 | FP32／FP16／INT8／INT4、量化方式、校正集、容許誤差 |
| 延遲 | p50／p95／p99、冷啟動、engine建置時間；推論和端到端分別量測 |
| 吞吐 | FPS／samples/s／tokens/s；清楚說明是單請求或總吞吐 |
| LLM | prompt長度、output長度、TTFT、decode速度；prefill與decode分開 |
| 記憶體 | 峰值、穩態、CPU／GPU共享或獨立、OOM／swap情況 |
| 電力 | 功耗模式、模組sensor值、整機量測方式；分開報告 |
| 穩定性 | 掉幀率、失敗率、溫度、時脈、長測期間 |
| 品質 | 與基準模型的誤差／準確率／召回率；不能只看速度 |

建議先暖機，再做至少三輪相同輸入測試；另外做一次足以觀察熱穩態的持續負載測試。此為實驗建議，非硬體廠商規定的認證流程。

Orin可用`tegrastats`觀察RAM、引擎使用率、溫度與電力欄位；實際欄位依裝置而異。下面以前景模式執行，完成後用Ctrl-C停止：

```bash
tegrastats --interval 1000
```

來源：[Tegrastats Utility，r36.4.4](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)。CPU/GPU時間線可進一步用相容版Nsight Systems；AMD選ROCm對應的profiler，Windows／WSL功能限制見AMD文件。

## 5. 建議的跨裝置實驗順序

1. 先在一台Orin完成單一工作負載，保存原始模型、輸入與正確性基準。
2. 搬到NVIDIA工作站：重建x86_64應用／容器與TensorRT engine，驗證同樣資料的品質與端到端效能。
3. 以同一PyTorch程式或可交換模型，驗證一張明確受支援的AMD GPU。
4. 只有遇到實測瓶頸才寫自訂CUDA／HIP kernel，並保留CPU參考實作。
5. 若目標是Ryzen AI NPU，另建立量化與編譯路線；不要把GPU的HIP測試當作NPU驗證。

建議的可交付目錄設計如下；這是後續實作範例，本次沒有建立這些程式或模型：

```text
models/       原始模型、ONNX、版本與校正資料說明
src/core/     與廠牌無關的業務邏輯與資料格式
src/backends/ cpu、cuda、hip、tensorrt、npu的適配層
deploy/       每平台獨立的依賴鎖定與容器建置
benchmarks/   固定輸入、品質檢查與量測工具
results/      原始log、環境資訊、比較報表
```

## 6. 選型判斷

- **相機／機器人整合為主**：先評估AGX Orin及其所需SDK版本交集。
- **通用PC應用加GPU運算**：NVIDIA或受支援AMD GPU都可評估；以現有程式依賴決定移植成本。
- **大型訓練或高FP64需求**：另評估對應工作站／資料中心GPU及系統，不從Orin TOPS推算。
- **低功耗固定模型推論**：評估DLA或Ryzen AI NPU，但將算子、形狀、精度與runtime支持作為前置條件。
- **同時支援兩家廠商**：優先Python/PyTorch或標準模型格式，將CUDA/HIP及專用影像API隔離。

以上是研究建議，不含採購價格或特定SKU的效能排名；最終選型應依本表的實測結果。
