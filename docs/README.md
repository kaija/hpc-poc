# NVIDIA Jetson AGX Orin 與 AMD 邊緣運算研究

研究日期：**2026-10-25（Asia/Taipei）**。以 NVIDIA、AMD、PyTorch、ONNX Runtime 等官方文件及專案文件為主要來源；各主題內有來源連結、版本條件與工程建議。

本研究中的「AGX Orin」指 **Jetson AGX Orin**。目前尚未指定32GB／64GB／Industrial SKU、載板、AMD型號與目標工作負載，故提供選型與驗證方法，未進行目標硬體benchmark。

## 閱讀索引

| 文件 | 回答的問題 |
| --- | --- |
| [01 — AGX Orin 邊緣運算能力](01-agx-orin-edge-computing.md) | 能做哪些視覺、機器人、訊號、生成式AI與數值運算？硬體和記憶體有哪些限制？ |
| [02 — CUDA libraries 與加速SDK](02-cuda-libraries.md) | cuBLAS、cuFFT、cuSPARSE、cuSOLVER、cuDNN、TensorRT、VPI、DeepStream等如何選？哪些有Jetson版本證據？ |
| [03 — 開發語言選擇](03-language-selection.md) | Python、C++／CUDA、Rust、Julia、Fortran各適合什麼？ |
| [04 — 跨NVIDIA及跨廠牌可攜性](04-portability.md) | 原始碼、PTX、ONNX、engine、container哪些能共用？跨RTX／伺服器／AMD需重做什麼？ |
| [05 — AMD硬體與運算軟體](05-amd-compute.md) | Instinct、Radeon、Ryzen iGPU、Ryzen AI NPU、CPU、FPGA如何使用？CUDA函式庫有哪些對應？ |
| [06 — 實機驗證計畫](06-validation-plan.md) | 如何盤點環境、驗證GPU執行、量測延遲／記憶體／功耗並比較平台？ |

若只想先做決策，讀本頁後接03與04；若要開始實驗，先按02／05確認版本，再照06驗證。

## 主要結論

1. **Orin適合現場感知與推論。** 多路影像分析、機器視覺、機器人感知、訊號處理與適當規模的本地LLM／VLM都是可評估方向；GPU、DLA、PVA與影片引擎要各自選軟體路徑。275 sparse INT8 TOPS不是任何程式都能取得的速度。[能力與依據](01-agx-orin-edge-computing.md)
2. **語言預設選Python + C++／CUDA。** Python負責模型實驗與應用流程，現成library執行GPU工作；只有量測發現瓶頸後才改寫熱點。既有C++產品可直接以C++整合。[語言分析](03-language-selection.md)
3. **跨NVIDIA以原始碼與原始模型為核心。** CUDA數學庫、標準張量程式與模型通常有較高重用性；CPU執行檔、TensorRT engine與Jetson專用I/O需要分平台建置或替換。[可攜性矩陣](04-portability.md)
4. **AMD GPU主要走ROCm／HIP。** 高階PyTorch程式及模型較容易重用；CUDA自訂kernel可評估HIPIFY。Ryzen AI NPU、CPU AOCL與Vitis是不同運算路線，不能用一套HIP程式涵蓋。[AMD分析與函式庫對照](05-amd-compute.md)
5. **選硬體前先選可驗證的軟體組合。** 精確SKU、OS、driver／BSP、library、framework和模型必須一起核對；最後用相同品質門檻比較端到端效能。[驗證流程](06-validation-plan.md)

## 版本與來源的使用方式

| 證據 | 本研究如何使用 |
| --- | --- |
| JetPack 6.2.1固定版文件 | 保留CUDA12.6、TensorRT10.3等歷史配套，作為具體可核對的基線，不稱為最新版 |
| 查閱時JetPack下載頁 | 列7.2.1／Jetson Linux39.2.1及Orin family；採用前仍查附加SDK和板級驅動 |
| ROCm 10.1.0固定版release notes | 頁面日期2026-10-05，作為AMD章節的版本基線；產品支持另查矩陣 |
| `latest`文件／動態產品頁 | 表示查閱當時內容，可能後續改版；不能外推到所有舊版 |
| 工程建議、算式與流程 | 為本研究分析，非廠商認證、價格比較或實測效能 |

主要版本來源：[JetPack 6.2.1](https://docs.nvidia.com/jetson/jetpack/6.2.1/release-notes/index.html)、[JetPack下載與SDK配套](https://developer.nvidia.com/embedded/jetpack/downloads)、[JetPack歷史支援](https://developer.nvidia.com/embedded/jetpack-archive)、[ROCm 10.1.0](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)。

特別注意：舊TensorRT網址可能重導新版，DLA、hardware compatibility、精度及plugin支持應按實際版本閱讀。文件有提供版本邊界；沒有把「套件可下載」「硬體有加速器」「模型能在加速器執行」視為同一件事。

## 後續實驗需要的輸入

- Orin模組SKU、RAM、JetPack／L4T版本、載板、相機與散熱。
- AMD完整型號、OS，以及要用GPU、NPU還是CPU。
- 代表性模型或演算法、輸入尺寸、目標精度、batch／並行數。
- 延遲、吞吐、功耗、成本與離線需求。

取得上述條件後，可依06建立具體的支援矩陣與PoC；本次已完成研究文件，尚無實機效能結果。
