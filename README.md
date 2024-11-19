# 金融文件檢索增強生成系統

本專案是透過與 Claude (Anthropic) AI 助理的協作開發完成。Claude 協助了系統架構設計、程式碼實作、錯誤處理邏輯以及文檔撰寫等關鍵部分。這種人機協作的方式讓我們能夠快速開發並改進強化系統功能與效能。

## 協作亮點

- 系統架構：與 Claude 共同設計模組化架構
- 程式碼實作：結合人類專業知識與 AI 最佳實踐
- 錯誤處理：完善的異常處理機制
- 文檔撰寫：清晰的文檔和註解說明
- 持續優化：反覆討論和改進系統設計

## 主要特點

- 結合向量搜索和 BM25 的混合檢索
- 針對繁體中文優化的 CKIP 文本處理
- 強化的快取和批次處理
- 多模態文件處理（PDF、文本）
- 針對金融和保險領域的專業優化
- 詳細的錯誤分析和日誌記錄

## 系統需求

- Python 3.8+
- PyTorch 
- transformers
- langchain
- CKIP Transformers
- FAISS
- OpenAI API 金鑰

## 安裝方式

```bash
pip install -r requirements.txt
```

設置環境變數：
```bash
export OPENAI_API_KEY=你的金鑰
```

## 使用方式

基本用法：
```bash
python complete_rag_openai.py \
  --question_path questions.json \
  --source_path documents/ \
  --output_path results.json \
  --retrieval_mode hybrid
```

重要參數：
- `retrieval_mode`: vector | bm25 | hybrid
- `device`: GPU 設備 ID（-1 表示使用 CPU）
- `chunk_size`: 文本分塊大小
- `chunk_overlap`: 分塊重疊度

## 配置說明

系統可通過 `RAGConfig` 配置以下參數：

- 文本處理（分塊大小、重疊度）
- 模型選擇（CKIP、OpenAI）
- 硬體使用（GPU/CPU）
- 快取行為
- 處理批次大小

## 系統架構

主要組件：

1. 文件處理器
   - PDF 提取
   - 文本正規化
   - 快取管理
  
2. CKIP 處理器
   - 中文斷詞
   - 文本分割
   - 批次處理

3. 查詢處理器
   - 查詢擴展
   - 模式匹配
   - 結果重排序

4. 向量存儲管理器
   - FAISS 整合
   - 嵌入管理
   - 存儲驗證

## 錯誤分析

包含全面的錯誤分析功能：

- 類別特定模式分析
- 詞彙重疊分析
- 上下文評估
- 詳細日誌記錄

## License

MIT License

## 參與貢獻

歡迎提交 Issue 和 Pull Request。請確保測試通過並符合專案程式碼風格。

## 開發團隊

- 人類開發者 Steve：負責專案規劃、需求分析、錯誤分析、程式實作和系統整合
- Claude (Anthropic)：協助系統設計、程式實作和文檔撰寫
- README 由 Claude (Anthropic) 協助生成

## 鳴謝

特別感謝 Claude (Anthropic) 在本專案開發過程中提供的寶貴建議和協助。這種人機協作的開發模式展現了 AI 輔助開發的潛力。
