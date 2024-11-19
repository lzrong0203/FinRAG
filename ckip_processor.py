import hashlib
import logging
import multiprocessing
import os
import pickle
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Any

import msgpack
import torch
import transformers
from ckip_transformers.nlp import CkipWordSegmenter
from tqdm import tqdm

# 設置 transformers 的日誌級別
transformers.logging.set_verbosity_error()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ckip_processor_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CKIPProcessorError(Exception):
    """CKIP處理器基礎異常類"""
    pass


class ModelInitializationError(CKIPProcessorError):
    """模型初始化異常"""
    pass


class ProcessingError(CKIPProcessorError):
    """文本處理異常"""
    pass


@dataclass
class ProcessingMetrics:
    """處理指標數據類"""
    total_texts: int
    processed_texts: int
    cache_hits: int
    processing_errors: int
    start_time: float
    end_time: Optional[float] = None

    @property
    def processing_time(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.total_texts if self.total_texts > 0 else 0

    @property
    def error_rate(self) -> float:
        return self.processing_errors / self.total_texts if self.total_texts > 0 else 0


class CacheManager:
    """緩存管理器"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "segmentation_cache.pkl"
        self.cache: Dict[int, str] = {}
        self.modified = False
        self._load_cache()

    def _load_cache(self) -> None:
        """載入緩存"""
        if self.cache_file.exists():
            try:
                with self.cache_file.open('rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} items from cache")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Starting with empty cache.")
                self.cache = {}

    def save_cache(self) -> None:
        """安全地保存緩存"""
        if not self.modified:
            return

        temp_file = self.cache_file.with_suffix('.tmp')
        try:
            with temp_file.open('wb') as f:
                pickle.dump(self.cache, f)
            temp_file.replace(self.cache_file)
            self.modified = False
            logger.info(f"Saved {len(self.cache)} items to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def get(self, key: int) -> Optional[str]:
        """獲取緩存項"""
        return self.cache.get(key)

    def set(self, key: int, value: str) -> None:
        """設置緩存項"""
        self.cache[key] = value
        self.modified = True


class CKIPProcessor:
    """CKIP文本處理器"""
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> 'CKIPProcessor':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
            self,
            cache_dir: str = "ckip_cache",
            batch_size: int = 32,
            num_processes: Optional[int] = None,
            model_name: str = "bert-base",
            device: int = 0,
            use_delim: bool = True,
            max_retries: int = 3
    ) -> None:
        """
        初始化CKIP處理器
        Args:
            cache_dir: 緩存目錄
            batch_size: 批次大小
            num_processes: CPU進程數
            model_name: 模型名稱
            device: GPU設備ID
            use_delim: 是否使用分隔符
            max_retries: 最大重試次數
        """
        if hasattr(self, 'initialized'):
            return

        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.num_processes = num_processes or max(1, multiprocessing.cpu_count() - 1)
        self.device = device
        self.use_delim = use_delim
        self.max_retries = max_retries

        # 檢查CUDA可用性
        if self.device >= 0 and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU.")
            self.device = -1

        # 創建緩存目錄
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manager = CacheManager(self.cache_dir)

        # 初始化模型
        self._initialize_model()

        self.initialized = True
        logger.info(
            f"Initialized CKIP Processor with {self.num_processes} processes "
            f"on {'GPU:' + str(self.device) if self.device >= 0 else 'CPU'}"
        )

    def _initialize_model(self) -> None:
        """初始化CKIP模型"""
        try:
            # 清理可能損壞的緩存
            for cache_path in [
                self.cache_dir / "hub",
                Path.home() / ".cache" / "huggingface",
                Path.home() / ".cache" / "torch"
            ]:
                if cache_path.exists():
                    try:
                        import shutil
                        shutil.rmtree(str(cache_path))
                    except Exception as e:
                        logger.warning(f"Failed to clean cache at {cache_path}: {e}")

            # 模型名稱映射
            model_mapping = {
                'bert-base': 'bert-base',
                'bert-tiny': 'bert-tiny',
                'albert-base': 'albert-base',
                'albert-tiny': 'albert-tiny'
            }

            model_name = model_mapping.get(self.model_name, 'bert-base')

            for attempt in range(self.max_retries):
                try:
                    self.ws_driver = CkipWordSegmenter(model=model_name, device=self.device)

                    # 驗證模型
                    test_result = self.ws_driver(["測試句子"])
                    if not test_result or not isinstance(test_result[0], list):
                        raise ValueError("Model validation failed")

                    logger.info(f"Successfully initialized CKIP model: {model_name}")
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise ModelInitializationError(
                            f"Failed to initialize model after {self.max_retries} attempts: {e}")
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    continue

        except Exception as e:
            raise ModelInitializationError(f"Fatal error initializing CKIP model: {e}")

    def _process_batch(self, texts: List[str], metrics: ProcessingMetrics) -> List[str]:
        """處理單一批次的文本"""
        results = []
        uncached_texts = []
        uncached_indices = []

        # 檢查快取
        for i, text in enumerate(texts):
            if not text.strip():
                results.append("")
                continue

            cached_result = self.cache_manager.get(text)
            if cached_result is not None:
                results.append(cached_result)
                metrics.cache_hits += 1
            else:
                results.append("")  # 佔位符
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 處理未快取的文本
        if uncached_texts:
            try:
                segmented = self.ws_driver(uncached_texts, use_delim=self.use_delim)

                # 更新結果和快取
                for i, seg_result in zip(uncached_indices, segmented):
                    result = " ".join(seg_result)
                    results[i] = result
                    self.cache_manager.set(uncached_texts[i], result)

                metrics.processed_texts += len(uncached_texts)

            except Exception as e:
                logger.error(f"Error in word segmentation: {e}")
                metrics.processing_errors += len(uncached_texts)
                for i, text in zip(uncached_indices, uncached_texts):
                    results[i] = text

        # 定期保存快取
        if metrics.processed_texts % 100 == 0:  # 減少存檔頻率
            self.cache_manager.save_cache()

        return results

    def segment_parallel(self, texts: List[str]) -> List[str]:
        """平行處理多個文本"""
        if not texts:
            return []

        # 過濾空文本
        texts = [text for text in texts if text and text.strip()]
        if not texts:
            return []

        # 初始化指標
        metrics = ProcessingMetrics(
            total_texts=len(texts),
            processed_texts=0,
            cache_hits=0,
            processing_errors=0,
            start_time=datetime.now().timestamp()
        )

        try:
            # 分批處理
            batches = [
                texts[i:i + self.batch_size]
                for i in range(0, len(texts), self.batch_size)
            ]

            results = []

            # GPU處理
            if self.device >= 0:
                # for batch in tqdm(batches, desc="Processing batches on GPU"):
                for batch in batches:
                    results.extend(self._process_batch(batch, metrics))
            # CPU多進程處理
            else:
                with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                    futures = []
                    for batch in batches:
                        future = executor.submit(self._process_batch, batch, metrics)
                        futures.append(future)

                    for future in tqdm(futures, desc="Processing batches on CPU"):
                        results.extend(future.result())

            # 更新並記錄指標
            metrics.end_time = datetime.now().timestamp()
            # logger.info(
            #     f"Processing completed: {metrics.processed_texts}/{metrics.total_texts} texts processed, "
            #     f"cache hit rate: {metrics.cache_hit_rate:.2%}, "
            #     f"error rate: {metrics.error_rate:.2%}, "
            #     f"processing time: {metrics.processing_time:.2f}s"
            # )

            # 定期保存緩存
            if metrics.processed_texts % 1000 == 0:
                self.cache_manager.save_cache()

            return results

        except Exception as e:
            logger.error(f"Error in parallel segmentation: {e}")
            metrics.end_time = datetime.now().timestamp()
            metrics.processing_errors = len(texts)
            raise ProcessingError(f"Failed to process texts: {e}")

        finally:
            # 確保緩存被保存
            self.cache_manager.save_cache()

    def __del__(self) -> None:
        """確保緩存被保存"""
        if hasattr(self, 'cache_manager'):
            self.cache_manager.save_cache()


from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """快取統計資訊"""
    total_hits: int = 0
    total_misses: int = 0
    total_items: int = 0
    last_cleanup: Optional[datetime] = None

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0


class CacheManager:
    """改進版快取管理器"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "segmentation_cache.pkl"
        self.stats_file = self.cache_dir / "cache_stats.json"
        self.cache: Dict[str, str] = {}
        self.modified = False
        self.stats = CacheStats()
        self._lock = threading.Lock()

        # 確保目錄存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_cache()

    def get_cache_key(self, text: str) -> str:
        """使用 MD5 生成穩定的快取金鑰"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    @lru_cache(maxsize=1000)
    def get(self, text: str) -> Optional[str]:
        """從快取獲取結果，先檢查記憶體快取"""
        with self._lock:
            key = self.get_cache_key(text)
            result = self.cache.get(key)
            if result is not None:
                self.stats.total_hits += 1
            else:
                self.stats.total_misses += 1
            return result

    def set(self, text: str, value: str) -> None:
        """設置快取值"""
        with self._lock:
            key = self.get_cache_key(text)
            self.cache[key] = value
            self.stats.total_items += 1
            self.modified = True

            # 當快取數量達到閾值時進行清理
            if self.stats.total_items > 100000:
                self._cleanup_cache()

    def _load_cache(self) -> None:
        """載入快取和統計資訊"""
        try:
            if self.cache_file.exists():
                with self.cache_file.open('rb') as f:
                    try:
                        self.cache = msgpack.unpack(f)
                    except:
                        # 如果 msgpack 載入失敗，嘗試使用 pickle
                        f.seek(0)
                        self.cache = pickle.load(f)

                self.stats.total_items = len(self.cache)
                logger.info(f"Loaded {len(self.cache)} items from cache")

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Starting with empty cache.")
            self.cache = {}

    def save_cache(self) -> None:
        """安全地保存快取"""
        if not self.modified:
            return

        with self._lock:
            temp_file = self.cache_file.with_suffix('.tmp')
            try:
                with temp_file.open('wb') as f:
                    msgpack.pack(self.cache, f)

                # 安全地替換檔案
                temp_file.replace(self.cache_file)
                self.modified = False
                logger.info(f"Saved {len(self.cache)} items to cache")

            except Exception as e:
                logger.error(f"Failed to save cache: {e}")
                if temp_file.exists():
                    temp_file.unlink()

    def _cleanup_cache(self) -> None:
        """清理過期的快取項目"""
        with self._lock:
            if len(self.cache) > 50000:  # 保留最新的 50000 項
                sorted_items = sorted(
                    self.cache.items(),
                    key=lambda x: len(x[1])  # 根據值的長度排序
                )
                self.cache = dict(sorted_items[-50000:])
                self.stats.total_items = len(self.cache)
                self.stats.last_cleanup = datetime.now()
                self.modified = True
                logger.info(f"Cleaned up cache, remaining items: {len(self.cache)}")


class CKIPEnhancedTextSplitter(RecursiveCharacterTextSplitter):
    """整合 CKIP 處理器的增強型文本分割器"""

    def __init__(
            self,
            ckip_processor: 'CKIPProcessor',
            chunk_size: int = 800,
            chunk_overlap: int = 400,
            category: str = None,  # 新增類別參數
            **kwargs: Any,
    ) -> None:
        # 基礎分隔符
        separators = [
            "\n\n",  # 段落分隔
            "。\n",  # 句號換行
            "。",  # 句號
            "；",  # 分號
            "，",  # 逗號
            " ",  # 空格
            ""  # 字符級別
        ]

        # 根據不同類別設置不同的保護模式
        self.protect_patterns = self._get_protect_patterns(category)

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            is_separator_regex=True,
            **kwargs
        )
        self.ckip_processor = ckip_processor

    def _get_protect_patterns(self, category: str) -> List[str]:
        """獲取需要保護的內容模式"""
        common_patterns = [
            r"第[一二三四五六七八九十]+條[\s\S]*?(?=第[一二三四五六七八九十]+條|$)"  # 完整條款
        ]

        if category == 'finance':
            finance_patterns = [
                # 財報表頭
                r"合併資產負債表[\s\S]*?單位：新台幣仟元",
                # 財務項目行
                r"現金及約當現金.*?(?=\n)",
                r"資產總計.*?(?=\n)",
                # 具體數字段落
                r"\$\s*[\d,]+\s*\d+%?\s*(?=\n)",
                # 報表時期
                r"民國\s*\d+\s*年\s*\d+\s*月\s*\d+\s*日"
            ]
            return common_patterns + finance_patterns

        return common_patterns

    def split_text(self, text: str) -> List[str]:
        """改進的文本分割方法"""
        try:
            # 1. 標記需要保護的部分
            protected_parts = []
            for pattern in self.protect_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    protected_parts.append((match.start(), match.end(), match.group()))

            # 2. 如果找到需要保護的部分，優先按這些部分分割
            if protected_parts:
                protected_parts.sort(key=lambda x: x[0])
                chunks = []
                last_end = 0

                for start, end, content in protected_parts:
                    # 處理保護部分之前的文本
                    if start > last_end:
                        interim_text = text[last_end:start]
                        if interim_text.strip():
                            chunks.extend(super().split_text(interim_text))

                    # 處理保護內容
                    if len(content) > self._chunk_size:
                        # 如果保護內容過長，使用更保守的分割方式
                        sub_chunks = self._split_long_protected_content(content)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(content)

                    last_end = end

                # 處理最後剩餘的部分
                if last_end < len(text):
                    remaining = text[last_end:]
                    if remaining.strip():
                        chunks.extend(super().split_text(remaining))

            else:
                # 如果沒有需要保護的部分，使用基本分割方法
                chunks = super().split_text(text)

            # 3. 後處理：清理和驗證
            return self._postprocess_chunks(chunks)

        except Exception as e:
            logger.error(f"Error in enhanced splitting: {e}")
            return super().split_text(text)

    def _split_long_protected_content(self, content: str) -> List[str]:
        """處理過長的保護內容"""
        # 針對不同類型的內容使用不同的分割策略
        if re.match(r"第[一二三四五六七八九十]+條", content):
            # 條款的分割
            return self._split_by_sentences(content)
        elif re.search(r"[0-9,]+(?:元|%)", content):
            # 數字相關內容的分割
            return self._split_preserving_numbers(content)
        else:
            # 一般內容的分割
            return self._split_by_delimiters(content)

    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子分割，保持句意完整"""
        sentences = re.split(r'(。|；)', text)
        current_chunk = []
        chunks = []
        current_length = 0

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]  # 加回分隔符

            if current_length + len(sentence) > self._chunk_size and current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunks.append(''.join(current_chunk))

        return chunks

    def _postprocess_chunks(self, chunks: List[str]) -> List[str]:
        """後處理分割後的文本塊"""
        processed_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # 確保條款編號不會出現在末尾
            if re.search(r'第[一二三四五六七八九十]+條\s*$', chunk):
                continue

            # 檢查並修復可能被分割的關鍵片段
            if re.search(r'^[，。；：]', chunk):
                if processed_chunks:
                    processed_chunks[-1] += chunk
                    continue

            processed_chunks.append(chunk)

        return processed_chunks
