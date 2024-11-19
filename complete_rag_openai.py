"""
RAG (Retrieval-Augmented Generation) System
基於 CKIP Transformers 與 OpenAI 的檢索增強生成系統
"""
import argparse
import gc
import hashlib
import io
import json
import logging
import os
import pickle
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import fitz
import pdfplumber
import pytesseract
import unicodedata
from PIL import Image
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from tqdm import tqdm

from ckip_processor import CKIPProcessor, CKIPEnhancedTextSplitter

# 載入環境變數
load_dotenv()

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'rag_system_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 異常類定義
class RAGSystemError(Exception):
    """RAG系統基礎異常類"""
    pass


class DocumentProcessingError(RAGSystemError):
    """文檔處理異常"""
    pass


class ModelError(RAGSystemError):
    """模型相關異常"""
    pass


class QueryProcessingError(RAGSystemError):
    """查詢處理異常"""
    pass


# 資料模型定義
from dataclasses import dataclass
import multiprocessing
import torch
import logging

logger = logging.getLogger(__name__)

from typing import Dict, List, Optional, Union


@dataclass
class RAGConfig:
    """
    RAG系統配置。包含所有必要的系統配置參數。

    Attributes:
        cache_dir (str): CKIP和向量存儲的緩存目錄
        vector_store_path (str): 向量存儲的保存路徑
        batch_size (int): 批處理大小
        num_processes (Optional[int]): CPU進程數，默認為(CPU核心數-1)
        device (int): GPU設備ID，-1表示使用CPU
        ckip_model (str): 使用的CKIP模型名稱
        model_name (str): 使用的OpenAI模型名稱
        temperature (float): OpenAI模型的temperature參數
        chunk_size (int): 文本分塊大小
        chunk_overlap (int): 文本分塊重疊大小
        use_delim (bool): 是否在CKIP處理中使用分隔符
        max_retries (int): 操作失敗時的最大重試次數
        timeout (int): 操作超時時間（秒）

    Notes:
        - 如果num_processes為None，將自動設置為(CPU核心數-1)
        - 如果指定的GPU設備不可用，將自動回退到CPU
        - chunk_size和chunk_overlap的值會影響檢索性能
    """
    # 必需參數
    cache_dir: str
    vector_store_path: str
    batch_size: int
    device: int
    ckip_model: str
    model_name: str
    temperature: float
    chunk_size: int
    chunk_overlap: int
    use_delim: bool
    max_retries: int
    timeout: int
    rebuild_stores: bool = False  # New parameter with default value

    # 可選參數
    num_processes: Optional[int] = None
    use_jieba: bool = False  # 新增屬性

    def __post_init__(self):
        """
        初始化後的處理，包括：
        1. 設置默認的進程數
        2. 檢查CUDA可用性
        3. 驗證參數有效性
        """
        # 設置進程數
        if self.num_processes is None:
            self.num_processes = max(1, multiprocessing.cpu_count() - 1)

        # 檢查CUDA可用性
        if self.device >= 0 and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU.")
            self.device = -1

        # 驗證參數
        self._validate_config()

    def _validate_config(self):
        """驗證配置參數的有效性"""
        # 驗證批次大小
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")

        # 驗證進程數
        if self.num_processes <= 0:
            raise ValueError(f"Number of processes must be positive, got {self.num_processes}")

        # 驗證塊大小和重疊
        if self.chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"Chunk overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"Chunk overlap ({self.chunk_overlap}) must be less than chunk size ({self.chunk_size})")

        # 驗證CKIP模型名稱
        valid_ckip_models = ['bert-base', 'bert-tiny', 'albert-base', 'albert-tiny']
        if self.ckip_model not in valid_ckip_models:
            raise ValueError(f"Invalid CKIP model. Must be one of {valid_ckip_models}")

        # 驗證溫度參數
        if not 0 <= self.temperature <= 1:
            raise ValueError(f"Temperature must be between 0 and 1, got {self.temperature}")

        # 驗證重試和超時參數
        if self.max_retries < 0:
            raise ValueError(f"Max retries must be non-negative, got {self.max_retries}")
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {self.timeout}")

    @property
    def device_name(self) -> str:
        """返回設備名稱（用於日誌和顯示）"""
        return f"cuda:{self.device}" if self.device >= 0 else "cpu"

    def to_dict(self) -> dict:
        """將配置轉換為字典格式"""
        return {
            'cache_dir': self.cache_dir,
            'vector_store_path': self.vector_store_path,
            'batch_size': self.batch_size,
            'num_processes': self.num_processes,
            'device': self.device,
            'device_name': self.device_name,
            'ckip_model': self.ckip_model,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'use_delim': self.use_delim,
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }

    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return (
            f"RAGConfig(\n"
            f"    cache_dir: {self.cache_dir}\n"
            f"    vector_store_path: {self.vector_store_path}\n"
            f"    batch_size: {self.batch_size}\n"
            f"    num_processes: {self.num_processes}\n"
            f"    device: {self.device_name}\n"
            f"    ckip_model: {self.ckip_model}\n"
            f"    model_name: {self.model_name}\n"
            f"    temperature: {self.temperature}\n"
            f"    chunk_size: {self.chunk_size}\n"
            f"    chunk_overlap: {self.chunk_overlap}\n"
            f"    use_delim: {self.use_delim}\n"
            f"    max_retries: {self.max_retries}\n"
            f"    timeout: {self.timeout}\n"
            f")"
        )


class QueryExpansion(BaseModel):
    """查詢擴展結果模型"""
    expanded_query: str = Field(description="擴充後的查詢語句")
    keywords: List[str] = Field(description="關鍵字列表")


class RetrievalResult(BaseModel):
    """檢索結果模型"""
    file_id: int = Field(description="最相關的檔案ID")
    confidence: float = Field(description="信心分數 (0-1)")


@dataclass
class ProcessingMetrics:
    """處理指標數據類"""
    start_time: float
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    end_time: Optional[float] = None

    @property
    def processing_time(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def success_rate(self) -> float:
        return self.processed_documents / self.total_documents if self.total_documents > 0 else 0

    @property
    def average_chunks_per_doc(self) -> float:
        return self.total_chunks / self.processed_documents if self.processed_documents > 0 else 0

    @property
    def average_tokens_per_doc(self) -> float:
        return self.total_tokens / self.processed_documents if self.processed_documents > 0 else 0


# 提示模板定義
QUERY_EXPANSION_TEMPLATE = """您是一位專精於台灣金融市場的資深顧問，具備完整的保險、理財、銀行實務知識。
請協助擴展並優化以下查詢。

原始查詢: {query}

請特別注意以下面向：
1. 金融相關：
   - 金融商品特性和銷售規範
   - 理財投資相關規定
   - 資產配置原則
   - 財務報表的內容

2. 保險相關：
   - 保單條款標準用語
   - 核保與理賠規則
   - 保險法規要求

3. 銀行相關：
   - 內部作業規範
   - 法令遵循要求
   - 風險管理規定

請提供：
1. 擴充後的查詢敘述：使用台灣金融市場的標準用語及專業術語
2. 關鍵字列表：標注核心概念和專業術語

{format_instructions}"""

RERANKING_TEMPLATE = """您是一位專業的金融文件審查專家，特別專精於台灣金融市場法規及實務。
請仔細評估下列文件與查詢的關聯性。

查詢問題：{query}

候選文件段落：
{contexts}

評估標準：
1. 內容相關性：
   - 是否直接回答查詢重點
   - 專業術語使用是否正確
   - 內容是否完整且具體

2. 實務應用性：
   - 是否符合實務作業需求
   - 是否包含具體執行方式
   - 是否有明確的流程說明

請根據以上標準，選出最適合的文件，並提供：
selected_id: [最相關的文件ID]
reason: [選擇原因，請簡要說明符合上述哪些標準]

注意：
1. 必須從上述文件中選擇，不能選擇其他文件
2. 如遇內容相似的情況，優先選擇：
   - 描述更具體完整的文件
   - 符合現行法規的版本
   - 包含實務執行細節的內容
"""


class DocumentProcessor:
    """文檔處理器基類"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.metrics = ProcessingMetrics(start_time=datetime.now().timestamp())

    def process_document(self, file_path: str) -> Optional[str]:
        """處理單個文檔"""
        raise NotImplementedError

    def process_documents(self, file_paths: List[str]) -> Dict[int, str]:
        """批次處理多個文檔"""
        raise NotImplementedError


# ... [前面的導入部分保持不變]

class EnhancedPDFProcessor(DocumentProcessor):
    """增強型PDF處理器，支援更好的快取追蹤和對照功能"""

    def __init__(self, config: RAGConfig):
        """初始化時確保目錄結構和快取"""
        super().__init__(config)

        # 基本設置
        self.min_text_length = 50
        self.ocr_confidence_threshold = 60
        self.ocr_batch_size = 5
        self.manifest_lock = threading.Lock()

        # 初始化快取相關屬性
        self.cache_manifest = {}  # 確保屬性存在
        self._initialize_cache_structure(config)

    def _initialize_cache_structure(self, config: RAGConfig):
        """初始化快取結構和manifest"""
        try:
            # 設置路徑
            self.cache_root = Path(config.cache_dir).resolve()
            self.text_cache_dir = self.cache_root / "text_cache"
            self.manifest_dir = self.cache_root / "manifests"
            self.manifest_file = self.manifest_dir / "cache_manifest.json"

            # 創建目錄結構
            for directory in [self.cache_root, self.text_cache_dir, self.manifest_dir]:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")

            # 載入或創建manifest
            self.cache_manifest = self._load_or_create_manifest()

        except Exception as e:
            logger.error(f"Error initializing cache structure: {e}")
            # 確保cache_manifest至少是一個空字典
            self.cache_manifest = self._create_empty_manifest()
            raise

    def _create_empty_manifest(self) -> Dict:
        """創建空的manifest結構"""
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "cache_format_version": "1.0",
            "files": {}
        }

    def _load_or_create_manifest(self) -> Dict:
        """載入或創建manifest"""
        try:
            if self.manifest_file.exists():
                try:
                    with open(self.manifest_file, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    if self._validate_manifest_file(self.manifest_file):
                        logger.info("Successfully loaded existing manifest")
                        return manifest
                except Exception as e:
                    logger.error(f"Error loading manifest file: {e}")

            # 如果加載失敗或文件不存在，創建新的
            logger.info("Creating new manifest file")
            return self._create_new_manifest()

        except Exception as e:
            logger.error(f"Error in manifest loading/creation: {e}")
            return self._create_empty_manifest()

    def _create_new_manifest(self) -> Dict:
        """創建新的manifest文件"""
        with self.manifest_lock:
            try:
                new_manifest = self._create_empty_manifest()
                temp_file = self.manifest_file.with_suffix(f'.tmp.{os.getpid()}')

                # 寫入臨時文件
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(new_manifest, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                # 驗證臨時文件
                if not self._validate_manifest_file(temp_file):
                    raise ValueError("Failed to validate temporary manifest file")

                # 安全地移動文件
                self._safe_rename(temp_file, self.manifest_file)
                logger.info("Successfully created new manifest file")

                return new_manifest

            except Exception as e:
                logger.error(f"Error creating new manifest: {e}")
                if 'temp_file' in locals() and temp_file.exists():
                    temp_file.unlink()
                return self._create_empty_manifest()

    def process_document(self, file_path: str) -> Optional[str]:
        """處理單個PDF文檔"""
        logger.info(f"Processing PDF: {file_path}")

        try:
            # 檢查快取
            cached_content = self.load_from_cache(file_path)
            if cached_content:
                logger.info(f"Using cached content for {file_path}")
                return cached_content

            # 處理文檔
            content = None
            for method in [self._extract_text_with_pdfplumber, self._extract_text_with_pymupdf]:
                try:
                    content = method(file_path)
                    if content and len(content.strip()) >= self.min_text_length:
                        break
                except Exception as e:
                    logger.warning(f"Extraction method failed: {str(e)}")
                    continue

            if content and len(content.strip()) >= self.min_text_length:
                # 保存到快取
                cache_path = self.save_to_cache(file_path, content)
                if cache_path:
                    logger.info(f"Successfully cached content to {cache_path}")
                return content.strip()

            raise DocumentProcessingError(f"No valid text extracted from PDF: {file_path}")

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return None

    def process_documents(self, file_paths: List[str]) -> Dict[int, str]:
        """批次處理多個PDF文檔"""
        self.metrics = ProcessingMetrics(start_time=datetime.now().timestamp())
        self.metrics.total_documents = len(file_paths)
        results: Dict[int, str] = {}

        try:
            with ThreadPoolExecutor(max_workers=self.config.num_processes) as executor:
                future_to_path = {
                    executor.submit(self.process_document, path): int(Path(path).stem)
                    for path in file_paths
                }

                for future in tqdm(future_to_path, desc="Processing documents"):
                    file_id = future_to_path[future]
                    try:
                        content = future.result()
                        if content:
                            results[file_id] = content
                            self.metrics.processed_documents += 1
                        else:
                            self.metrics.failed_documents += 1
                    except Exception as e:
                        logger.error(f"Error processing file {file_id}: {str(e)}")
                        self.metrics.failed_documents += 1
                        continue

            # 生成處理報告
            self.generate_cache_report()

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise DocumentProcessingError(f"Batch processing failed: {str(e)}")

        finally:
            self.metrics.end_time = datetime.now().timestamp()
            logger.info(
                f"Document processing completed: "
                f"{len(results)}/{self.metrics.total_documents} "
                f"documents processed successfully"
            )

        return results

    def _extract_text_with_pdfplumber(self, file_path: str) -> Optional[str]:
        """使用 pdfplumber 提取文本"""
        try:
            texts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    try:
                        text = page.extract_text(x_tolerance=3, y_tolerance=3)
                        if text:
                            texts.append(text)
                        else:
                            tables = page.extract_tables()
                            if tables:
                                table_texts = []
                                for table in tables:
                                    table_text = "\n".join(
                                        " | ".join(filter(None, row))
                                        for row in table
                                        if any(cell for cell in row)
                                    )
                                    table_texts.append(table_text)
                                texts.extend(table_texts)
                    except Exception as e:
                        logger.warning(f"Error in pdfplumber extraction for page: {str(e)}")
                        continue
                    finally:
                        gc.collect()

                combined_text = "\n".join(filter(None, texts))
                return combined_text if combined_text.strip() else None

        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
            return None
        finally:
            gc.collect()

    def _extract_text_with_pymupdf(self, file_path: str) -> Optional[str]:
        """使用 PyMuPDF 提取文本"""
        try:
            texts = []
            images_batch = []

            with fitz.open(file_path) as doc:
                for page in doc:
                    text = page.get_text("text", sort=True)
                    if text.strip():
                        texts.append(text)

                    if not text.strip():
                        for img in page.get_images():
                            try:
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_data = base_image["image"]

                                image = Image.open(io.BytesIO(image_data))
                                if 'dpi' not in image.info:
                                    image.info['dpi'] = (300, 300)

                                images_batch.append(image)

                                if len(images_batch) >= self.ocr_batch_size:
                                    ocr_results = self._process_image_batch(images_batch)
                                    texts.extend(filter(None, ocr_results))
                                    images_batch = []
                            except Exception as e:
                                logger.warning(f"Error processing image: {str(e)}")
                                continue

                if images_batch:
                    ocr_results = self._process_image_batch(images_batch)
                    texts.extend(filter(None, ocr_results))

            combined_text = "\n".join(filter(None, texts))
            return combined_text if combined_text.strip() else None

        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}")
            return None
        finally:
            gc.collect()

    def _process_image_batch(self, images: List[Image.Image]) -> List[str]:
        """批次處理OCR"""
        try:
            results = []
            for image in images:
                try:
                    if not hasattr(image, 'info') or \
                            'dpi' not in image.info or \
                            (isinstance(image.info['dpi'], tuple) and image.info['dpi'][0] < 70):
                        scale = 300 / 72
                        new_size = tuple(int(dim * scale) for dim in image.size)
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                        image.info['dpi'] = (300, 300)

                    custom_config = r'--oem 1 --psm 3'
                    languages = 'chi_tra+eng' if not hasattr(pytesseract, 'get_languages') else '+'.join(
                        ['chi_tra', 'eng'])

                    text = pytesseract.image_to_string(
                        image,
                        lang=languages,
                        config=custom_config
                    )

                    text = text.strip()
                    results.append(text if text else "")
                except Exception as e:
                    logger.warning(f"OCR processing error: {str(e)}")
                    results.append("")
                finally:
                    try:
                        image.close()
                    except:
                        pass
                    gc.collect()
            return results
        finally:
            gc.collect()

    def _validate_manifest_file(self, file_path: Path) -> bool:
        """驗證manifest文件的完整性"""
        try:
            if not file_path.exists():
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            required_keys = {'created_at', 'last_updated', 'cache_format_version', 'files'}
            return all(key in data for key in required_keys)

        except Exception as e:
            logger.error(f"Manifest validation failed: {e}")
            return False

    def _safe_rename(self, src: Path, dst: Path) -> None:
        """安全地重命名文件"""
        try:
            if os.name == 'nt' and dst.exists():
                dst.unlink()
            src.replace(dst)
        except Exception as e:
            logger.error(f"Error during file rename: {e}")
            raise

    def generate_cache_report(self, output_dir: Optional[Path] = None) -> None:
        """生成快取報告"""
        if output_dir is None:
            output_dir = self.manifest_dir

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f"cache_report_{timestamp}.md"

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# PDF處理快取報告\n\n")

                # 總體統計
                f.write("## 1. 總體統計\n\n")
                total_files = len(self.cache_manifest.get("files", {}))
                total_size = sum(
                    info.get("file_size", 0)
                    for info in self.cache_manifest.get("files", {}).values()
                )
                f.write(f"- 快取文件總數: {total_files}\n")
                f.write(f"- 總大小: {total_size / 1024:.2f} KB\n")
                f.write(f"- 最後更新時間: {self.cache_manifest.get('last_updated', 'N/A')}\n")
                f.write(f"- 快取格式版本: {self.cache_manifest.get('cache_format_version', 'N/A')}\n\n")

                # 文件詳情
                f.write("## 2. 快取文件詳情\n\n")
                for original_path, info in self.cache_manifest.get("files", {}).items():
                    f.write(f"### {Path(original_path).name}\n\n")
                    f.write(f"- 快取文件名: {info.get('cache_filename', 'N/A')}\n")
                    f.write(f"- 創建時間: {info.get('created_at', 'N/A')}\n")
                    f.write(f"- 文件大小: {info.get('file_size', 0)} bytes\n")
                    if "processing_info" in info:
                        f.write("- 文本統計:\n")
                        f.write(f"  - 單詞數: {info['processing_info'].get('word_count', 0)}\n")
                        f.write(f"  - 字符數: {info['processing_info'].get('character_count', 0)}\n")
                        f.write(f"  - 行數: {info['processing_info'].get('line_count', 0)}\n")
                    f.write(f"- 內容雜湊: {info.get('content_hash', 'N/A')}\n\n")

        except Exception as e:
            logger.error(f"Error generating cache report: {e}")

    def _update_manifest(self, file_path: str, cache_info: Dict):
        """更新manifest"""
        with self.manifest_lock:
            try:
                # 更新快取資訊
                self.cache_manifest["last_updated"] = datetime.now().isoformat()
                self.cache_manifest["files"][str(file_path)] = cache_info

                # 生成備份
                self._create_manifest_backup()

                # 使用臨時文件進行寫入
                temp_file = self.manifest_file.with_suffix(f'.tmp.{os.getpid()}')

                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache_manifest, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                # 驗證臨時文件
                if not self._validate_manifest_file(temp_file):
                    raise ValueError("Failed to validate temporary manifest file")

                # 安全地替換原文件
                self._safe_rename(temp_file, self.manifest_file)
                logger.info("Successfully updated manifest file")

            except Exception as e:
                logger.error(f"Error updating manifest: {e}")
                if 'temp_file' in locals() and temp_file.exists():
                    temp_file.unlink()
                self._restore_from_backup()
                raise

    def _create_manifest_backup(self):
        """創建manifest備份"""
        try:
            backup_path = self.manifest_file.with_suffix(
                f'.backup.{datetime.now().strftime("%Y%m%d%H%M%S")}'
            )
            shutil.copy2(self.manifest_file, backup_path)

            # 清理舊備份，只保留最近的5個
            backups = sorted(
                self.manifest_dir.glob('cache_manifest.backup.*'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            for old_backup in backups[5:]:
                old_backup.unlink()

        except Exception as e:
            logger.error(f"Error creating manifest backup: {e}")

    def _restore_from_backup(self):
        """從最新的備份恢復"""
        try:
            backups = sorted(
                self.manifest_dir.glob('cache_manifest.backup.*'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            if backups:
                latest_backup = backups[0]
                shutil.copy2(latest_backup, self.manifest_file)
                logger.info(f"Restored manifest from backup: {latest_backup}")

                # 重新載入manifest
                self.cache_manifest = self._load_or_create_manifest()
            else:
                logger.warning("No backup found, creating new manifest")
                self.cache_manifest = self._create_new_manifest()

        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            raise

    def cleanup_invalid_cache(self) -> None:
        """清理無效的快取文件"""
        try:
            # 收集所有有效的快取文件名
            valid_cache_files = {
                info.get("cache_filename")
                for info in self.cache_manifest.get("files", {}).values()
                if info.get("cache_filename")
            }

            # 檢查快取目錄中的所有文件
            for cache_file in self.text_cache_dir.glob("*.txt"):
                if cache_file.name not in valid_cache_files:
                    logger.warning(f"Removing invalid cache file: {cache_file}")
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.error(f"Error removing invalid cache file {cache_file}: {e}")

            # 驗證快取完整性
            self._verify_cache_integrity()

        except Exception as e:
            logger.error(f"Error cleaning up invalid cache: {e}")

    def _verify_cache_integrity(self) -> bool:
        """驗證快取完整性"""
        try:
            for file_path, info in self.cache_manifest.get("files", {}).items():
                cache_filename = info.get("cache_filename")
                if not cache_filename:
                    logger.warning(f"Missing cache filename for {file_path}")
                    continue

                cache_path = self.text_cache_dir / cache_filename
                if not cache_path.exists():
                    logger.warning(f"Missing cache file: {cache_filename}")
                    continue

                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    current_hash = hashlib.md5(content.encode()).hexdigest()

                    if current_hash != info.get("content_hash"):
                        logger.warning(f"Hash mismatch for {cache_filename}")
                        continue
                except Exception as e:
                    logger.error(f"Error verifying cache file {cache_filename}: {e}")
                    continue

            return True
        except Exception as e:
            logger.error(f"Error verifying cache integrity: {e}")
            return False

    def _generate_cache_path(self, file_path: str) -> Tuple[Path, str]:
        """生成快取文件路徑和文件名"""
        original_name = Path(file_path).stem
        parent_dir = Path(file_path).parent.name
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]

        cache_filename = f"{parent_dir}_{original_name}_{file_hash}.txt"
        cache_path = self.text_cache_dir / cache_filename

        return cache_path, cache_filename

    def load_from_cache(self, file_path: str) -> Optional[str]:
        """從快取載入內容

        Args:
            file_path: 原始文件路徑

        Returns:
            Optional[str]: 快取的文本內容，如果不存在則返回None
        """
        try:
            if not hasattr(self, 'cache_manifest'):
                logger.error("Cache manifest not initialized")
                return None

            # 檢查manifest中是否有該文件的信息
            file_info = self.cache_manifest.get("files", {}).get(str(file_path))
            if not file_info:
                logger.debug(f"No cache info found for {file_path}")
                return None

            cache_path = self.text_cache_dir / file_info.get("cache_filename", "")
            if not cache_path.exists():
                logger.warning(f"Cache file not found: {cache_path}")
                return None

            # 讀取快取內容
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 驗證內容雜湊
                current_hash = hashlib.md5(content.encode()).hexdigest()
                if current_hash != file_info.get("content_hash"):
                    logger.warning(f"Cache content hash mismatch for {file_path}")
                    return None

                logger.info(f"Successfully loaded from cache: {file_path}")
                return content
            except Exception as read_error:
                logger.error(f"Error reading cache file {cache_path}: {read_error}")
                return None

        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None

    def save_to_cache(self, file_path: str, content: str) -> Optional[str]:
        """保存內容到快取

        Args:
            file_path: 原始文件路徑
            content: 要快取的文本內容

        Returns:
            Optional[str]: 快取文件的路徑，如果保存失敗則返回None
        """
        try:
            if not hasattr(self, 'cache_manifest'):
                logger.error("Cache manifest not initialized")
                return None

            # 生成快取路徑和文件名
            cache_path, cache_filename = self._generate_cache_path(file_path)

            # 準備快取信息
            cache_info = {
                "original_path": str(file_path),
                "cache_filename": cache_filename,
                "created_at": datetime.now().isoformat(),
                "file_size": len(content),
                "content_hash": hashlib.md5(content.encode()).hexdigest(),
                "processing_info": {
                    "word_count": len(content.split()),
                    "character_count": len(content),
                    "line_count": len(content.splitlines())
                }
            }

            # 使用臨時文件保存內容
            temp_path = cache_path.with_suffix(f'.tmp.{os.getpid()}')
            try:
                # 確保目錄存在
                cache_path.parent.mkdir(parents=True, exist_ok=True)

                # 寫入內容到臨時文件
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())

                # 驗證寫入是否成功
                if not temp_path.exists() or temp_path.stat().st_size == 0:
                    raise IOError("Failed to write temporary cache file")

                # 安全地移動文件
                self._safe_rename(temp_path, cache_path)

                # 更新manifest
                self._update_manifest(file_path, cache_info)

                logger.info(f"Successfully cached content to {cache_path}")
                return str(cache_path)

            except Exception as e:
                logger.error(f"Error writing cache file: {e}")
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up temporary file: {cleanup_error}")
                return None

        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return None

    def __del__(self):
        """確保在物件被刪除時保存manifest"""
        try:
            if hasattr(self, 'cache_manifest') and hasattr(self, 'manifest_file'):
                with open(self.manifest_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache_manifest, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving manifest during cleanup: {e}")


class CKIPProcessorWrapper:
    """CKIP處理器包裝類，處理GPU記憶體問題和快取"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.processor = None
        self.init_attempts = 3
        self.cache_manager = EnhancedCacheManager(config.cache_dir)
        # Add memory management parameters
        self.max_batch_size = 16  # Reduced from 32
        self.dynamic_batch = True

    def _initialize_processor(self):
        """Initialize processor with better GPU memory management"""
        for attempt in range(self.init_attempts):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Set GPU memory management
                    torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory

                if self.processor is None:
                    self.processor = CKIPProcessor(
                        cache_dir=self.config.cache_dir,
                        batch_size=self.max_batch_size,
                        model_name=self.config.ckip_model,
                        device=self.config.device,
                        use_delim=self.config.use_delim
                    )
                return True
            except Exception as e:
                logger.warning(f"CKIP initialization attempt {attempt + 1} failed: {str(e)}")
                if "CUDA out of memory" in str(e):
                    # Reduce batch size and try again
                    self.max_batch_size = max(1, self.max_batch_size // 2)
                    logger.info(f"Reducing batch size to {self.max_batch_size}")

                if self.processor:
                    del self.processor
                    self.processor = None
                torch.cuda.empty_cache()
                gc.collect()

                if attempt == self.init_attempts - 1 and self.config.device >= 0:
                    logger.warning("Switching to CPU due to persistent GPU issues")
                    self.config.device = -1

        return False

    def segment_parallel(self, texts: List[str]) -> List[str]:
        """Enhanced parallel segmentation with better error handling"""
        if not texts:
            return []

        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            if not text:  # Handle empty text
                results.append("")
                continue

            cached_result = self.cache_manager.load_ckip_cache(text)
            if cached_result is not None:
                results.append(cached_result)
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Process uncached texts
        if uncached_texts:
            if not self._initialize_processor():
                raise RuntimeError("Failed to initialize CKIP processor")

            try:
                # Dynamic batch processing
                current_batch_size = self.max_batch_size
                processed_results = []

                for i in range(0, len(uncached_texts), current_batch_size):
                    batch = uncached_texts[i:i + current_batch_size]
                    try:
                        batch_results = self.processor.segment_parallel(batch)
                        processed_results.extend(batch_results)
                    except Exception as e:
                        if "CUDA out of memory" in str(e) and current_batch_size > 1:
                            # Reduce batch size and retry
                            current_batch_size = max(1, current_batch_size // 2)
                            i -= len(batch)  # Retry the same batch
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        raise

                # Update results and cache
                for i, (idx, result) in enumerate(zip(uncached_indices, processed_results)):
                    if idx < len(results):  # Prevent index out of range
                        results[idx] = result
                        if i < len(uncached_texts):  # Prevent index out of range
                            self.cache_manager.save_ckip_cache(uncached_texts[i], result)

            except Exception as e:
                logger.error(f"Error in segmentation: {str(e)}")
                # Fill remaining None results with empty strings
                results = [r if r is not None else "" for r in results]

        return [r for r in results if r is not None]

    def __del__(self):
        """清理資源"""
        if self.processor:
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class VectorStoreManager:
    """向量存儲管理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings()
        self.vector_store_path = Path(config.vector_store_path)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.text_cache_dir = Path(config.cache_dir) / "text_cache"
        self.text_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manager = EnhancedCacheManager(config.cache_dir)

        # 初始化 CKIP 處理器
        self.ckip_processor = CKIPProcessor(
            cache_dir=config.cache_dir,
            batch_size=config.batch_size,
            model_name=config.ckip_model,
            device=config.device,
            use_delim=config.use_delim
        )

        # 使用新的增強型分割器
        self.text_splitter = CKIPEnhancedTextSplitter(
            ckip_processor=self.ckip_processor,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            keep_separator=True
        )
        logger.info(f"Initialized text splitter with chunk_size={config.chunk_size}, "
                    f"overlap={config.chunk_overlap}")

    def _generate_config_hash(self) -> str:
        """生成配置的hash值"""
        config_str = f"{self.config.chunk_size}_{self.config.chunk_overlap}"
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_store_path(self, category: str) -> Path:
        """獲取向量存儲路徑，包含chunk參數信息"""
        store_name = (f"{category}_"
                      f"s{self.config.chunk_size}_"
                      f"o{self.config.chunk_overlap}")
        return self.vector_store_path / store_name

    def _validate_store_config(self, store_path: Path) -> bool:
        """驗證存儲配置是否與當前設置匹配"""
        config_path = store_path / "config.json"
        if not config_path.exists():
            return False

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                stored_config = json.load(f)

            chunk_config = stored_config.get("chunk_config", {})
            model_config = stored_config.get("model_config", {})

            # 檢查 chunk 參數和 embedding model
            if (chunk_config.get("chunk_size") != self.config.chunk_size or
                    chunk_config.get("chunk_overlap") != self.config.chunk_overlap or
                    model_config.get("embedding_model") != self.embeddings.model):
                logger.info(f"Configuration mismatch. Stored: {stored_config}")
                logger.info(f"Current: chunk_size={self.config.chunk_size}, "
                            f"overlap={self.config.chunk_overlap}, "
                            f"model={self.embeddings.model}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating store config: {e}")
            return False

    def _save_config_metadata(self, store_path: Path) -> None:
        """保存配置元數據，包含 embedding model 信息"""
        metadata = {
            "creation_time": datetime.now().isoformat(),
            "chunk_config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap
            },
            "model_config": {
                "embedding_model": self.embeddings.model,
                "ckip_model": self.config.ckip_model,
                "use_delim": self.config.use_delim
            },
            "version": "1.0"
        }

        metadata_path = store_path / "config.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def create_vector_store(
            self,
            category: str,
            texts: List[str],
            file_ids: List[int]
    ) -> Optional[FAISS]:
        """
        Create vector store with fixed document handling.

        Args:
            category: Document category
            texts: List of text contents
            file_ids: List of file IDs

        Returns:
            Optional[FAISS]: Created vector store, or None if creation fails
        """
        logger.info(
            f"Creating vector store for category: {category} "
            f"(chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.chunk_overlap})"
        )
        logger.info(f"Input documents: {len(texts)}, File IDs: {len(file_ids)}")

        try:
            if not texts or not file_ids or len(texts) != len(file_ids):
                raise ValueError("Invalid input data")

            store_path = self._get_store_path(category)

            # Track chunking statistics
            chunk_stats = {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "max_chunk_size": 0,
                "min_chunk_size": float('inf'),
                "total_docs": len(texts)
            }

            # Create documents using enhanced splitter
            chunk_texts = []
            metadatas = []

            for text, file_id in zip(texts, file_ids):
                try:
                    metadata = {
                        "source": file_id,
                        "original_length": len(text),
                        "category": category
                    }

                    # 直接使用文本分割器處理文本
                    chunks = self.text_splitter.split_text(text)

                    # 為每個分塊創建對應的元數據
                    for chunk in chunks:
                        chunk_size = len(chunk)
                        chunk_stats["total_chunks"] += 1
                        chunk_stats["max_chunk_size"] = max(
                            chunk_stats["max_chunk_size"],
                            chunk_size
                        )
                        chunk_stats["min_chunk_size"] = min(
                            chunk_stats["min_chunk_size"],
                            chunk_size
                        )
                        chunk_stats["avg_chunk_size"] += chunk_size

                        # 添加分塊特定的元數據
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            "chunk_size": chunk_size,
                            "chunk_index": len(chunk_texts)
                        })

                        chunk_texts.append(chunk)
                        metadatas.append(chunk_metadata)

                except Exception as e:
                    logger.warning(f"Error processing document {file_id}: {str(e)}")
                    continue

            if not chunk_texts:
                raise ValueError("No valid chunks generated")

            # Finalize statistics
            if chunk_stats["total_chunks"] > 0:
                chunk_stats["avg_chunk_size"] /= chunk_stats["total_chunks"]

            # Log chunking statistics
            logger.info(
                f"Chunk statistics for {category}:\n"
                f"Total chunks: {chunk_stats['total_chunks']}\n"
                f"Average chunk size: {chunk_stats['avg_chunk_size']:.2f}\n"
                f"Max chunk size: {chunk_stats['max_chunk_size']}\n"
                f"Min chunk size: {chunk_stats['min_chunk_size']}\n"
                f"Documents processed: {chunk_stats['total_docs']}"
            )

            # Generate embeddings with caching
            logger.info("Generating embeddings...")
            cache_stats = {'hits': 0, 'misses': 0}

            try:
                # 使用 from_texts 創建向量存儲
                vector_store = FAISS.from_texts(
                    texts=chunk_texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
                logger.info("Successfully created FAISS index")
            except Exception as e:
                logger.error(f"Error creating FAISS index: {e}")
                raise

            # Save vector store and configuration
            if store_path.exists():
                logger.info(f"Removing existing store at {store_path}")
                shutil.rmtree(str(store_path))

            store_path.mkdir(parents=True)
            vector_store.save_local(str(store_path))

            # Save extended configuration
            metadata = {
                "creation_time": datetime.now().isoformat(),
                "chunk_config": {
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap  # 修正這裡
                },
                "model_config": {
                    "embedding_model": self.embeddings.model,
                    "model_version": getattr(self.embeddings, 'model_version', 'unknown'),
                    "ckip_model": self.config.ckip_model
                },
                "statistics": chunk_stats,
                "version": "1.0"
            }

            with open(store_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully created vector store for {category}")
            return vector_store

        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
            return None

    def load_vector_store(self, category: str) -> Optional[FAISS]:
        """載入向量存儲，確保chunk參數匹配"""
        store_path = self._get_store_path(category)
        logger.info(f"Attempting to load vector store for category: {category}")

        try:
            if not store_path.exists():
                logger.info(f"Vector store not found for category: {category}")
                return None

            # 驗證配置
            if not self._validate_store_config(store_path):
                logger.info(
                    f"Configuration mismatch for {category}, "
                    f"store needs rebuild with current chunk settings"
                )
                return None

            # 載入向量存儲
            vector_store = FAISS.load_local(
                str(store_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            if not self._validate_store(vector_store):
                logger.error(f"Vector store validation failed for {category}")
                return None

            # 載入並記錄統計信息
            try:
                with open(store_path / "config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                    stats = config.get("statistics", {})
                    if stats:
                        logger.info(
                            f"Loaded vector store statistics for {category}:\n"
                            f"Total chunks: {stats.get('total_chunks', 'N/A')}\n"
                            f"Average chunk size: {stats.get('avg_chunk_size', 'N/A'):.2f}\n"
                            f"Created at: {config.get('creation_time', 'N/A')}"
                        )
            except Exception as e:
                logger.warning(f"Could not load store statistics: {e}")

            logger.info(f"Successfully loaded vector store for {category}")
            return vector_store

        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None

    def clear_cache(self):
        """清理所有緩存和向量存儲"""
        try:
            if Path(self.config.cache_dir).exists():
                shutil.rmtree(self.config.cache_dir)
            if Path(self.config.vector_store_path).exists():
                shutil.rmtree(self.config.vector_store_path)
            logger.info("Successfully cleared all caches")
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")

    def _validate_store(self, store: FAISS) -> bool:
        """驗證向量存儲的有效性

        Args:
            store: FAISS 向量存儲實例

        Returns:
            bool: 存儲是否有效
        """
        try:
            # 檢查必要的屬性
            if not hasattr(store, 'index') or not hasattr(store, 'docstore'):
                logger.error("Vector store missing required attributes")
                return False

            # 檢查索引是否為空
            if not store.index or store.index.ntotal == 0:
                logger.error("Vector store index is empty")
                return False

            # 檢查文檔存儲是否為空
            if not store.docstore or len(store.docstore._dict) == 0:
                logger.error("Vector store docstore is empty")
                return False

            # 檢查維度是否匹配
            expected_dim = 1536
            if hasattr(store.index, 'ntotal') and store.index.ntotal > 0:
                actual_dim = store.index.d
                if actual_dim != expected_dim:
                    logger.error(
                        f"Dimension mismatch: expected {expected_dim}, "
                        f"got {actual_dim}"
                    )
                    return False

            # 記錄存儲統計信息
            logger.info(
                f"Vector store validation passed:\n"
                f"Total vectors: {store.index.ntotal}\n"
                f"Dimension: {store.index.d}\n"
                f"Documents: {len(store.docstore._dict)}"
            )

            return True

        except Exception as e:
            logger.error(f"Error validating vector store: {e}")
            return False

    def _validate_store_content(self, store: FAISS) -> bool:
        """驗證向量存儲的內容有效性

        Args:
            store: FAISS 向量存儲實例

        Returns:
            bool: 內容是否有效
        """
        try:
            # 進行簡單的查詢測試
            test_query = "test query"
            test_embedding = self.embeddings.embed_query(test_query)

            # 嘗試搜索
            store.similarity_search_with_score_by_vector(
                test_embedding,
                k=1
            )

            return True

        except Exception as e:
            logger.error(f"Error validating store content: {e}")
            return False

    def clear_store(self, category: Optional[str] = None) -> bool:
        """安全清理向量存儲

        Args:
            category: 可選，指定要清理的類別。如果為 None，清理所有類別

        Returns:
            bool: 清理是否成功
        """
        try:
            # 如果指定了類別，只清理該類別
            if category:
                store_path = self._get_store_path(category)
                if store_path.exists():
                    shutil.rmtree(store_path)
                    logger.info(f"Cleared vector store for category: {category}")
            # 如果沒有指定類別，清理所有類別
            else:
                if self.vector_store_path.exists():
                    for path in self.vector_store_path.glob("*"):
                        if path.is_dir():
                            try:
                                shutil.rmtree(path)
                                logger.info(f"Cleared vector store: {path.name}")
                            except Exception as e:
                                logger.error(f"Error clearing {path}: {e}")
                                return False
                    logger.info("Cleared all vector stores")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False


def normalize_unicode(text: str) -> str:
    """處理Unicode編碼問題"""
    try:
        # 嘗試解碼可能的轉義序列
        if '\\u' in text:
            text = text.encode('utf-8').decode('unicode-escape')

        # 標準化Unicode
        text = unicodedata.normalize('NFKC', text)
        return text
    except Exception as e:
        logger.warning(f"Unicode normalization failed: {str(e)}")
        return text


@dataclass
class CacheMetadata:
    """快取元數據"""
    created_at: str
    updated_at: str
    version: str = "1.0.0"
    entries: int = 0
    size_bytes: int = 0
    chunk_config: Optional[Dict[str, Any]] = None  # 用於存儲chunk相關配置

    @classmethod
    def create(cls, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> 'CacheMetadata':
        """創建新的元數據實例"""
        chunk_config = None
        if chunk_size is not None and chunk_overlap is not None:
            chunk_config = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }

        return cls(
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            chunk_config=chunk_config
        )

    def update(self, entries_delta: int = 0, size_delta: int = 0) -> None:
        """更新元數據"""
        self.updated_at = datetime.now().isoformat()
        self.entries += entries_delta
        self.size_bytes += size_delta

    def matches_config(self, chunk_size: Optional[int], chunk_overlap: Optional[int]) -> bool:
        """檢查配置是否匹配"""
        if self.chunk_config is None:
            return chunk_size is None and chunk_overlap is None

        return (self.chunk_config.get("chunk_size") == chunk_size and
                self.chunk_config.get("chunk_overlap") == chunk_overlap)


class EnhancedCacheManager:
    """增強型快取管理器"""

    def __init__(self, base_cache_dir: str):
        self.base_dir = Path(base_cache_dir)
        self.vector_cache_dir = self.base_dir / "vector_cache"
        self.text_cache_dir = self.base_dir / "text_cache"
        self.ckip_cache_dir = self.base_dir / "ckip_cache"
        self.embedding_cache_dir = self.base_dir / "embedding_cache"  # 新增

        # 建立所需目錄
        for dir_path in [self.vector_cache_dir, self.text_cache_dir, self.ckip_cache_dir,
                         self.embedding_cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.metadata = {}
        self._load_metadata()

        # 初始化 embedding 快取計數器
        self.embedding_stats = {
            'hits': 0,
            'misses': 0,
            'saved_api_calls': 0
        }

    def _generate_embedding_key(self, text: str) -> str:
        """生成 embedding 快取的鍵值"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def save_embedding(self, text: str, embedding: List[float]) -> bool:
        """保存 embedding 到快取"""
        try:
            cache_key = self._generate_embedding_key(text)
            cache_path = self.embedding_cache_dir / f"{cache_key}.pkl"

            # 使用臨時文件確保原子性寫入
            temp_path = cache_path.with_suffix('.tmp')

            cache_data = {
                'text': text,
                'embedding': embedding,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }

            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f)

            # 安全地移動臨時文件
            temp_path.replace(cache_path)

            # 更新元數據
            size_delta = os.path.getsize(cache_path)
            self._update_metadata(
                self.embedding_cache_dir,
                entries_delta=1,
                size_delta=size_delta
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
            return False

    def load_embedding(self, text: str) -> Optional[List[float]]:
        """從快取載入 embedding"""
        try:
            cache_key = self._generate_embedding_key(text)
            cache_path = self.embedding_cache_dir / f"{cache_key}.pkl"

            if not cache_path.exists():
                self.embedding_stats['misses'] += 1
                return None

            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # 驗證快取數據
            if (cache_data.get('text') != text or
                    'embedding' not in cache_data or
                    not isinstance(cache_data['embedding'], list)):
                return None

            self.embedding_stats['hits'] += 1
            self.embedding_stats['saved_api_calls'] += 1
            return cache_data['embedding']

        except Exception as e:
            logger.error(f"Failed to load embedding cache: {e}")
            return None

    def get_embedding_stats(self) -> Dict[str, int]:
        """獲取 embedding 快取統計信息"""
        total_requests = self.embedding_stats['hits'] + self.embedding_stats['misses']
        hit_rate = (self.embedding_stats['hits'] / total_requests * 100
                    if total_requests > 0 else 0)

        return {
            'cache_hits': self.embedding_stats['hits'],
            'cache_misses': self.embedding_stats['misses'],
            'hit_rate': f"{hit_rate:.2f}%",
            'saved_api_calls': self.embedding_stats['saved_api_calls']
        }

    def cleanup_old_embeddings(self, max_age_days: int = 30) -> None:
        """清理舊的 embedding 快取"""
        try:
            current_time = datetime.now()
            cleanup_count = 0

            for cache_file in self.embedding_cache_dir.glob("*.pkl"):
                try:
                    # 檢查文件年齡
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    age_days = (current_time - file_time).days

                    if age_days > max_age_days:
                        cache_file.unlink()
                        cleanup_count += 1

                except Exception as e:
                    logger.warning(f"Error processing cache file {cache_file}: {e}")
                    continue

            logger.info(f"Cleaned up {cleanup_count} old embedding cache files")

        except Exception as e:
            logger.error(f"Error during embedding cache cleanup: {e}")

    def clear_embedding_cache(self) -> None:
        """清理所有 embedding 快取"""
        try:
            for cache_file in self.embedding_cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")

            # 重置統計信息
            self.embedding_stats = {
                'hits': 0,
                'misses': 0,
                'saved_api_calls': 0
            }

            logger.info("Embedding cache cleared successfully")

        except Exception as e:
            logger.error(f"Error clearing embedding cache: {e}")

    def _generate_cache_key(self, content: Union[str, bytes], config: Optional[Dict] = None) -> str:
        """生成快取鍵值，可選擇性地包含配置信息"""
        if isinstance(content, str):
            content = content.encode('utf-8')

        if config:
            config_str = json.dumps(config, sort_keys=True)
            content = content + config_str.encode('utf-8')

        return hashlib.sha256(content).hexdigest()

    def _load_metadata(self) -> None:
        """載入每個快取目錄的元數據"""
        for cache_dir in [self.vector_cache_dir, self.text_cache_dir, self.ckip_cache_dir]:
            metadata_file = cache_dir / "metadata.json"
            try:
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        self.metadata[cache_dir] = json.load(f)
                else:
                    self._initialize_metadata(cache_dir)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {cache_dir}: {e}")
                self._initialize_metadata(cache_dir)

    def _initialize_metadata(self, cache_dir: Path, chunk_size: Optional[int] = None,
                             chunk_overlap: Optional[int] = None) -> None:
        """初始化快取元數據"""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "entries": 0,
            "size_bytes": 0
        }

        if chunk_size is not None and chunk_overlap is not None:
            metadata["chunk_config"] = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }

        self.metadata[cache_dir] = metadata
        self._save_metadata(cache_dir)

    def _save_metadata(self, cache_dir: Path) -> None:
        """保存快取元數據"""
        try:
            metadata_file = cache_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata[cache_dir], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata for {cache_dir}: {e}")

    def _update_metadata(self, cache_dir: Path, entries_delta: int = 0,
                         size_delta: int = 0, chunk_size: Optional[int] = None,
                         chunk_overlap: Optional[int] = None) -> None:
        """更新快取元數據"""
        if cache_dir not in self.metadata:
            self._initialize_metadata(cache_dir, chunk_size, chunk_overlap)

        metadata = self.metadata[cache_dir]
        metadata["updated_at"] = datetime.now().isoformat()
        metadata["entries"] += entries_delta
        metadata["size_bytes"] += size_delta

        if chunk_size is not None and chunk_overlap is not None:
            metadata["chunk_config"] = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }

        self._save_metadata(cache_dir)

    def check_chunk_config(self, cache_dir: Path, chunk_size: int,
                           chunk_overlap: int) -> bool:
        """檢查快取的chunk配置是否匹配"""
        if cache_dir not in self.metadata:
            return False

        metadata = self.metadata[cache_dir]
        chunk_config = metadata.get("chunk_config")

        if chunk_config is None:
            return False

        return (chunk_config.get("chunk_size") == chunk_size and
                chunk_config.get("chunk_overlap") == chunk_overlap)

    def save_text_cache(self, file_path: str, content: str) -> bool:
        """保存文本快取"""
        try:
            # 確保 text_cache 目錄存在
            self.text_cache_dir.mkdir(parents=True, exist_ok=True)

            cache_key = self._generate_cache_key(str(file_path))
            cache_file = self.text_cache_dir / f"{cache_key}.txt"

            # 使用臨時文件確保原子性寫入
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content)

            # 安全地移動臨時文件
            temp_file.replace(cache_file)

            # 更新元數據
            size_delta = os.path.getsize(cache_file)
            self._update_metadata(self.text_cache_dir, 1, size_delta)

            return True
        except Exception as e:
            logger.error(f"Failed to save text cache: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return False

    def load_text_cache(self, file_path: str) -> Optional[str]:
        """載入文本快取"""
        cache_key = self._generate_cache_key(str(file_path))
        cache_file = self.text_cache_dir / f"{cache_key}.txt"

        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to load text cache: {e}")
        return None

    def save_ckip_cache(self, text: str, segmented: str) -> bool:
        """保存CKIP分詞快取"""
        try:
            cache_key = self._generate_cache_key(text)
            cache_file = self.ckip_cache_dir / f"{cache_key}.pkl"

            # 使用臨時文件確保原子性寫入
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(segmented, f)

            # 安全地移動臨時文件
            temp_file.replace(cache_file)

            # 更新元數據
            size_delta = os.path.getsize(cache_file)
            self._update_metadata(self.ckip_cache_dir, 1, size_delta)

            return True
        except Exception as e:
            logger.error(f"Failed to save CKIP cache: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return False

    def load_ckip_cache(self, text: str) -> Optional[str]:
        """載入CKIP分詞快取"""
        cache_key = self._generate_cache_key(text)
        cache_file = self.ckip_cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load CKIP cache: {e}")
        return None

    def save_cache(self, cache_dir: Path, key: str, content: Any,
                   chunk_size: Optional[int] = None,
                   chunk_overlap: Optional[int] = None) -> bool:
        """保存內容到快取"""
        try:
            cache_path = cache_dir / f"{key}"

            # 使用臨時文件確保原子性寫入
            temp_path = cache_path.with_suffix('.tmp')

            # 根據內容類型選擇序列化方法
            if isinstance(content, (dict, list)):
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
            else:
                with open(temp_path, 'wb') as f:
                    pickle.dump(content, f)

            # 原子性替換文件
            temp_path.replace(cache_path)

            # 更新元數據
            size = cache_path.stat().st_size
            self._update_metadata(
                cache_dir,
                entries_delta=1,
                size_delta=size,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False

    def load_cache(self, cache_dir: Path, key: str,
                   chunk_size: Optional[int] = None,
                   chunk_overlap: Optional[int] = None) -> Optional[Any]:
        """從快取載入內容"""
        try:
            cache_path = cache_dir / f"{key}"

            if not cache_path.exists():
                return None

            # 如果指定了chunk配置，檢查是否匹配
            if chunk_size is not None and chunk_overlap is not None:
                if not self.check_chunk_config(cache_dir, chunk_size, chunk_overlap):
                    return None

            # 嘗試作為JSON載入
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                # 如果不是JSON，嘗試作為pickle載入
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None

    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """清理指定類型或所有快取"""
        if cache_type == "vector":
            self._clear_directory(self.vector_cache_dir)
        elif cache_type == "text":
            self._clear_directory(self.text_cache_dir)
        elif cache_type == "ckip":
            self._clear_directory(self.ckip_cache_dir)
        else:
            for dir_path in [self.vector_cache_dir, self.text_cache_dir, self.ckip_cache_dir]:
                self._clear_directory(dir_path)

    def _clear_directory(self, directory: Path) -> None:
        """清理指定目錄"""
        try:
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            self._initialize_metadata(directory)
        except Exception as e:
            logger.error(f"Failed to clear directory {directory}: {e}")

    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """獲取快取統計信息"""
        return {
            "vector_cache": self.metadata.get(self.vector_cache_dir, {}),
            "text_cache": self.metadata.get(self.text_cache_dir, {}),
            "ckip_cache": self.metadata.get(self.ckip_cache_dir, {})
        }


class QueryProcessor:
    """查詢處理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature
        )
        self.ckip_processor = CKIPProcessorWrapper(config)
        self.cache_manager = EnhancedCacheManager(config.cache_dir)

        # 添加查詢快取統計
        self.query_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'expansions': 0
        }

    def _get_cached_query_embedding(self, query: str) -> Optional[List[float]]:
        """獲取快取的查詢 embedding"""
        return self.cache_manager.load_embedding(f"query_{query}")

    def _cache_query_embedding(self, query: str, embedding: List[float]) -> None:
        """快取查詢 embedding"""
        self.cache_manager.save_embedding(f"query_{query}", embedding)

    def _normalize_unicode(self, text: str) -> str:
        """處理Unicode編碼問題"""
        try:
            # 嘗試解碼可能的轉義序列
            if '\\u' in text:
                text = text.encode('utf-8').decode('unicode-escape')

            # 標準化Unicode
            return unicodedata.normalize('NFKC', text)
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {str(e)}")
            return text

    def _parse_llm_output(self, output: Union[str, AIMessage, Any]) -> Dict:
        """解析 LLM 輸出，處理 AIMessage 和其他格式"""
        try:
            # 處理 AIMessage
            if isinstance(output, AIMessage):
                content = output.content
            else:
                content = str(output)

            # 移除可能的 Unicode 轉義
            content = self._normalize_unicode(content)

            # 嘗試直接解析 JSON
            try:
                result = json.loads(content)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

            # 嘗試找出並解析 JSON 部分
            json_matches = re.findall(r'\{.*?\}', content, re.DOTALL)

            for json_str in json_matches:
                try:
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        return result
                except:
                    continue

            # 嘗試解析結構化文本
            file_id_match = re.search(r'file_id["\s:]+(\d+)', content)
            confidence_match = re.search(r'confidence["\s:]+([0-9.]+)', content)

            if file_id_match and confidence_match:
                return {
                    "file_id": int(file_id_match.group(1)),
                    "confidence": float(confidence_match.group(1))
                }

            raise ValueError(f"Unable to parse output format: {content}")

        except Exception as e:
            logger.error(f"Error parsing LLM output: {str(e)}")
            return {
                "file_id": -1,
                "confidence": 0.0
            }

    def expand_query(self, query: str) -> QueryExpansion:
        """查詢擴展"""
        try:
            query = self._normalize_unicode(query)
            logger.info(f"Original query: {query}")

            # 定義提示模板
            template = """請協助擴展以下查詢並提供關鍵字。

原始查詢: {query}

請以以下JSON格式回應:
{{
    "expanded_query": "擴展後的查詢文字",
    "keywords": ["關鍵字1", "關鍵字2", "..."]
}}"""

            prompt = ChatPromptTemplate.from_template(template)

            # 執行查詢擴展
            logger.info("Sending query expansion request to OpenAI")
            response = self.llm.invoke(prompt.format(query=query))
            logger.info(f"OpenAI response for query expansion: {response}")

            # 解析回應
            parsed = self._parse_llm_output(response)
            logger.info(f"Parsed expansion result: {parsed}")

            # 確保回傳值符合預期格式
            expanded_query = parsed.get("expanded_query", query)
            keywords = parsed.get("keywords", [])

            # 確保 keywords 是列表
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",")]
            elif not isinstance(keywords, list):
                keywords = []

            logger.info(f"Final expanded query: {expanded_query}")
            logger.info(f"Final keywords: {keywords}")

            return QueryExpansion(
                expanded_query=expanded_query,
                keywords=keywords
            )

        except Exception as e:
            logger.error(f"Error in query expansion: {str(e)}", exc_info=True)
            return QueryExpansion(expanded_query=query, keywords=[])

    def rerank_results(
            self,
            query: str,
            expanded_query: str,
            keywords: List[str],
            contexts: List[Tuple[str, int]]
    ) -> RetrievalResult:
        """重新排序檢索結果"""
        try:
            # 定義提示模板
            template = """請分析查詢與檢索文件間的相關性。

查詢: {query}
擴展查詢: {expanded_query}
關鍵字: {keywords}

檢索到的文件:
{context}

請以以下JSON格式回應:
{{
    "file_id": <最相關文件ID，整數>,
    "confidence": <相關性分數，0到1之間的浮點數>
}}"""

            prompt = ChatPromptTemplate.from_template(template)

            # 格式化上下文
            context_str = "\n\n".join([
                f"文件 {file_id}:\n{text}"
                for text, file_id in contexts
            ])

            # 執行重排序
            response = self.llm.invoke(
                prompt.format(
                    query=query,
                    expanded_query=expanded_query,
                    keywords=", ".join(keywords),
                    context=context_str
                )
            )

            # 解析回應
            parsed = self._parse_llm_output(response)

            # 驗證和清理結果
            file_id = int(parsed.get("file_id", -1))
            confidence = float(parsed.get("confidence", 0.0))

            # 確保值在有效範圍內
            confidence = max(0.0, min(1.0, confidence))

            # 如果沒有有效的 file_id，使用第一個上下文的 ID
            if file_id == -1 and contexts:
                file_id = contexts[0][1]

            result = RetrievalResult(
                file_id=file_id,
                confidence=confidence
            )

            logger.info(
                f"Reranking result: file_id={result.file_id}, "
                f"confidence={result.confidence:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            default_file_id = contexts[0][1] if contexts else -1
            return RetrievalResult(
                file_id=default_file_id,
                confidence=0.5
            )

    def process_query(self, query: str, source: List[str], category: str) -> Tuple[str, List[str], List[float]]:
        """處理查詢並返回擴展後的查詢、關鍵字和 embedding"""
        try:
            self.query_stats['total_queries'] += 1

            # 1. 先嘗試從快取獲取 embedding
            cached_embedding = self._get_cached_query_embedding(query)
            if cached_embedding is not None:
                self.query_stats['cache_hits'] += 1
                logger.info("Using cached query embedding")
                # 即使使用快取的 embedding，我們仍然需要進行查詢擴展
                expanded = self.expand_query(query)
                return expanded.expanded_query, expanded.keywords, cached_embedding

            # 2. 如果快取未命中，進行完整的查詢處理
            # 查詢擴展
            expanded = self.expand_query(query)
            self.query_stats['expansions'] += 1

            # 生成新的 embedding
            embedder = OpenAIEmbeddings()
            embedding = embedder.embed_query(expanded.expanded_query)

            # 將新的 embedding 保存到快取
            self._cache_query_embedding(query, embedding)
            logger.info("Created and cached new query embedding")

            return expanded.expanded_query, expanded.keywords, embedding

        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            # 在錯誤情況下，返回原始查詢和空的關鍵字列表
            return query, [], []

    def get_query_stats(self) -> Dict[str, Any]:
        """獲取查詢處理統計信息"""
        stats = self.query_stats.copy()
        if stats['total_queries'] > 0:
            stats['cache_hit_rate'] = f"{(stats['cache_hits'] / stats['total_queries']) * 100:.2f}%"
        else:
            stats['cache_hit_rate'] = "0%"
        return stats

    def clear_query_cache(self) -> None:
        """清理查詢快取"""
        try:
            # 只清理查詢相關的快取
            for cache_file in self.cache_manager.embedding_cache_dir.glob("query_*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete query cache file {cache_file}: {e}")

            # 重置統計
            self.query_stats = {
                'total_queries': 0,
                'cache_hits': 0,
                'expansions': 0
            }
            logger.info("Query cache cleared successfully")

        except Exception as e:
            logger.error(f"Error clearing query cache: {e}")


from rank_bm25 import BM25Okapi
from typing import Dict, List, Optional
import numpy as np


# class EnhancedInformationExtractor:
#     """增強型信息提取器，實現分層提取和結構化整合"""
#
#     def __init__(self, llm):
#         self.llm = llm
#
#     def extract_document_level_info(self, doc_text: str, query: str) -> Dict[str, Any]:
#         """提取單個文檔級別的關鍵信息"""
#         prompt = f"""分析以下文檔，提取與查詢相關的關鍵信息：
#
# 文檔內容：{doc_text}
# 查詢問題：{query}
#
# 請提供以下分析：
# 1. 主要論點和事實
# 2. 關鍵數據和指標
# 3. 時間和條件信息
# 4. 因果關係
#
# 請以JSON格式回答，包含以上各點。"""
#
#         response = self.llm.invoke(prompt)
#         # 解析並返回結構化信息
#         return self._parse_llm_output(response.content)
#
#     def analyze_cross_doc_relations(self, doc_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """分析多個文檔之間的語義關聯"""
#         docs_summary = json.dumps(doc_infos, ensure_ascii=False, indent=2)
#         prompt = f"""分析以下多個文檔的信息之間的關聯：
#
# 文檔信息：{docs_summary}
#
# 請分析：
# 1. 信息的一致性和互補性
# 2. 可能的衝突點
# 3. 時序關係
# 4. 邏輯依賴關係
#
# 請以JSON格式提供分析結果。"""
#
#         response = self.llm.invoke(prompt)
#         return self._parse_llm_output(response.content)
#
#     def extract_key_sentences(self, text: str, query: str) -> List[str]:
#         """使用關鍵句抽取來識別最相關的句子"""
#         sentences = text.split('。')
#         prompt = f"""對於以下查詢，從給定句子中選出最相關的關鍵句：
#
# 查詢：{query}
# 句子列表：
# {json.dumps(sentences, ensure_ascii=False, indent=2)}
#
# 請選出最相關的3-5個句子，並說明選擇原因。
# 格式：[句子索引列表]"""
#
#         response = self.llm.invoke(prompt)
#         selected_indices = self._parse_sentence_indices(response.content)
#         return [sentences[i] for i in selected_indices if i < len(sentences)]
#
#     def create_structured_summary(self, doc_infos: Dict[str, Any],
#                                   cross_doc_analysis: Dict[str, Any],
#                                   key_sentences: List[str]) -> Dict[str, Any]:
#         """整合各層次信息創建結構化摘要"""
#         summary_data = {
#             "document_level_info": doc_infos,
#             "cross_document_analysis": cross_doc_analysis,
#             "key_evidence": key_sentences,
#             "timestamp": datetime.now().isoformat()
#         }
#
#         prompt = f"""基於以下信息創建完整的結構化摘要：
#
# 信息內容：{json.dumps(summary_data, ensure_ascii=False, indent=2)}
#
# 請提供：
# 1. 核心結論
# 2. 支持證據
# 3. 潛在不確定性
# 4. 相關上下文
#
# 以JSON格式回應。"""
#
#         response = self.llm.invoke(prompt)
#         return self._parse_llm_output(response.content)
#
#     def _parse_llm_output(self, output: str) -> Dict[str, Any]:
#         """解析LLM輸出為結構化數據"""
#         try:
#             # 清理輸出文本
#             cleaned_text = output.strip()
#             # 提取JSON部分
#             json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
#             if json_match:
#                 return json.loads(json_match.group())
#             return {"error": "無法解析輸出", "raw_output": cleaned_text}
#         except Exception as e:
#             logger.error(f"解析LLM輸出時出錯: {e}")
#             return {"error": str(e), "raw_output": output}
#
#     def _parse_sentence_indices(self, output: str) -> List[int]:
#         """從LLM輸出解析句子索引"""
#         try:
#             # 尋找所有數字
#             indices = re.findall(r'\d+', output)
#             return [int(idx) for idx in indices]
#         except Exception as e:
#             logger.error(f"解析句子索引時出錯: {e}")
#             return []

class EnhancedInformationExtractor:
    """增強型信息提取器，實現分層提取和結構化整合"""

    def __init__(self, llm):
        self.llm = llm

    def extract_document_level_info(self, doc_text: str, query: str) -> Dict[str, Any]:
        """提取單個文檔級別的關鍵信息"""
        prompt = f"""分析以下文檔，提取與查詢相關的關鍵信息：

文檔內容：{doc_text}
查詢問題：{query}

請提供以下分析：
1. 主要論點和事實
2. 關鍵數據和指標
3. 時間和條件信息
4. 因果關係

請以JSON格式回答，包含以上各點。"""

        response = self.llm.invoke(prompt)
        return self._parse_llm_output(response.content)

    def analyze_cross_doc_relations(self, doc_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析多個文檔之間的語義關聯"""
        docs_summary = json.dumps(doc_infos, ensure_ascii=False, indent=2)
        prompt = f"""分析以下多個文檔的信息之間的關聯：

文檔信息：{docs_summary}

請分析：
1. 信息的一致性和互補性
2. 可能的衝突點
3. 時序關係
4. 邏輯依賴關係

請以JSON格式提供分析結果。"""

        response = self.llm.invoke(prompt)
        return self._parse_llm_output(response.content)

    def extract_key_sentences(self, text: str, query: str) -> List[str]:
        """使用關鍵句抽取來識別最相關的句子"""
        sentences = text.split('。')
        prompt = f"""對於以下查詢，從給定句子中選出最相關的關鍵句：

查詢：{query}
句子列表：
{json.dumps(sentences, ensure_ascii=False, indent=2)}

請選出最相關的3-5個句子，並說明選擇原因。
格式：[句子索引列表]"""

        response = self.llm.invoke(prompt)
        selected_indices = self._parse_sentence_indices(response.content)
        return [sentences[i] for i in selected_indices if i < len(sentences)]

    def create_structured_summary(self, doc_infos: Dict[str, Any],
                                  cross_doc_analysis: Dict[str, Any],
                                  key_sentences: List[str]) -> Dict[str, Any]:
        """整合各層次信息創建結構化摘要"""
        summary_data = {
            "document_level_info": doc_infos,
            "cross_document_analysis": cross_doc_analysis,
            "key_evidence": key_sentences,
            "timestamp": datetime.now().isoformat()
        }

        prompt = f"""基於以下信息創建完整的結構化摘要：

信息內容：{json.dumps(summary_data, ensure_ascii=False, indent=2)}

請提供：
1. 核心結論
2. 支持證據
3. 潛在不確定性
4. 相關上下文

以JSON格式回應。"""

        response = self.llm.invoke(prompt)
        return self._parse_llm_output(response.content)

    def _parse_llm_output(self, output: str) -> Dict[str, Any]:
        """解析LLM輸出為結構化數據"""
        try:
            cleaned_text = output.strip()
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"error": "無法解析輸出", "raw_output": cleaned_text}
        except Exception as e:
            logger.error(f"解析LLM輸出時出錯: {e}")
            return {"error": str(e), "raw_output": output}

    def _parse_sentence_indices(self, output: str) -> List[int]:
        """從LLM輸出解析句子索引"""
        try:
            indices = re.findall(r'\d+', output)
            return [int(idx) for idx in indices]
        except Exception as e:
            logger.error(f"解析句子索引時出錯: {e}")
            return []


class EnhancedCoTFilter:
    """增強型CoT引導過濾器，實現細粒度推理和證據評分"""

    def __init__(self, llm):
        self.llm = llm
        self.reasoning_steps = [
            "識別關鍵實體和概念",
            "分析時間和條件約束",
            "確定因果關係",
            "評估證據完整性",
            "考慮反事實情況"
        ]

    def generate_detailed_cot(self, query: str, context: str) -> Dict[str, Any]:
        """生成詳細的思維鏈分析"""
        prompt = f"""對以下查詢進行詳細的推理分析：

查詢：{query}
上下文：{context}

請按照以下步驟進行分析：
{self._format_reasoning_steps()}

對每個步驟：
1. 提供具體推理過程
2. 列出關鍵證據
3. 標註不確定性
4. 提供可能的替代解釋

請以JSON格式回應，包含每個步驟的分析結果。"""

        response = self.llm.invoke(prompt)
        return self._parse_reasoning_output(response.content)

    def evaluate_evidence(self, chunk: str, reasoning: Dict[str, Any]) -> Dict[str, float]:
        """評估證據的相關性和可靠性"""
        evidence_criteria = {
            "relevance": "與查詢的直接相關性",
            "completeness": "信息的完整性",
            "consistency": "與其他證據的一致性",
            "specificity": "細節的具體程度"
        }

        prompt = f"""評估以下文本段落作為證據的質量：

文本：{chunk}
推理分析：{json.dumps(reasoning, ensure_ascii=False, indent=2)}

請對以下標準進行評分（0-1分）：
{json.dumps(evidence_criteria, ensure_ascii=False, indent=2)}

解釋您的評分理由。
請以JSON格式回應，包含分數和理由。"""

        response = self.llm.invoke(prompt)
        return self._parse_evidence_scores(response.content)

    def counterfactual_analysis(self, query: str, chunk: str,
                                reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """進行反事實分析"""
        prompt = f"""對以下情況進行反事實分析：

查詢：{query}
文本：{chunk}
當前推理：{json.dumps(reasoning, ensure_ascii=False, indent=2)}

請考慮：
1. 如果關鍵信息有誤，結論是否改變？
2. 是否存在其他可能的解釋？
3. 缺少什麼關鍵信息？
4. 當前結論的穩健性如何？

請以JSON格式提供分析結果。"""

        response = self.llm.invoke(prompt)
        return self._parse_llm_output(response.content)

    def filter_chunks(self, query: str, chunks: List[str],
                      min_score: float = 0.6) -> List[Dict[str, Any]]:
        """基於詳細分析過濾文檔塊"""
        filtered_results = []

        for chunk in chunks:
            # 生成詳細推理
            reasoning = self.generate_detailed_cot(query, chunk)

            # 評估證據
            evidence_scores = self.evaluate_evidence(chunk, reasoning)

            # 進行反事實分析
            counterfactual = self.counterfactual_analysis(query, chunk, reasoning)

            # 計算綜合分數
            overall_score = self._calculate_overall_score(evidence_scores, counterfactual)

            if overall_score >= min_score:
                filtered_results.append({
                    "chunk": chunk,
                    "reasoning": reasoning,
                    "evidence_scores": evidence_scores,
                    "counterfactual": counterfactual,
                    "overall_score": overall_score
                })

        return sorted(filtered_results, key=lambda x: x["overall_score"], reverse=True)

    def _format_reasoning_steps(self) -> str:
        """格式化推理步驟"""
        return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(self.reasoning_steps))

    def _parse_reasoning_output(self, output: str) -> Dict[str, Any]:
        """解析推理輸出"""
        try:
            cleaned_text = output.strip()
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"error": "無法解析推理輸出"}
        except Exception as e:
            logger.error(f"解析推理輸出時出錯: {e}")
            return {"error": str(e)}

    def _parse_evidence_scores(self, output: str) -> Dict[str, float]:
        """解析證據評分"""
        try:
            data = self._parse_llm_output(output)
            return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
        except Exception as e:
            logger.error(f"解析證據評分時出錯: {e}")
            return {}

    def _calculate_overall_score(self, evidence_scores: Dict[str, float],
                                 counterfactual: Dict[str, Any]) -> float:
        """計算綜合評分"""
        weights = {
            "relevance": 0.4,
            "completeness": 0.3,
            "consistency": 0.2,
            "specificity": 0.1
        }

        base_score = sum(score * weights.get(criterion, 0)
                         for criterion, score in evidence_scores.items())

        confidence = counterfactual.get("conclusion_robustness", 0.5)
        return base_score * confidence
# class EnhancedCoTFilter:
#     """增強型CoT引導過濾器，實現細粒度推理和證據評分"""
#
#     def __init__(self, llm):
#         self.llm = llm
#         self.reasoning_steps = [
#             "識別關鍵實體和概念",
#             "分析時間和條件約束",
#             "確定因果關係",
#             "評估證據完整性",
#             "考慮反事實情況"
#         ]
#
#     def generate_detailed_cot(self, query: str, context: str) -> Dict[str, Any]:
#         """生成詳細的思維鏈分析"""
#         prompt = f"""對以下查詢進行詳細的推理分析：
#
# 查詢：{query}
# 上下文：{context}
#
# 請按照以下步驟進行分析：
# {self._format_reasoning_steps()}
#
# 對每個步驟：
# 1. 提供具體推理過程
# 2. 列出關鍵證據
# 3. 標註不確定性
# 4. 提供可能的替代解釋
#
# 請以JSON格式回應，包含每個步驟的分析結果。"""
#
#         response = self.llm.invoke(prompt)
#         return self._parse_reasoning_output(response.content)
#
#     def evaluate_evidence(self, chunk: str, reasoning: Dict[str, Any]) -> Dict[str, float]:
#         """評估證據的相關性和可靠性"""
#         evidence_criteria = {
#             "relevance": "與查詢的直接相關性",
#             "completeness": "信息的完整性",
#             "consistency": "與其他證據的一致性",
#             "specificity": "細節的具體程度"
#         }
#
#         prompt = f"""評估以下文本段落作為證據的質量：
#
# 文本：{chunk}
# 推理分析：{json.dumps(reasoning, ensure_ascii=False, indent=2)}
#
# 請對以下標準進行評分（0-1分）：
# {json.dumps(evidence_criteria, ensure_ascii=False, indent=2)}
#
# 解釋您的評分理由。
# 請以JSON格式回應，包含分數和理由。"""
#
#         response = self.llm.invoke(prompt)
#         return self._parse_evidence_scores(response.content)
#
#     def counterfactual_analysis(self, query: str, chunk: str,
#                                 reasoning: Dict[str, Any]) -> Dict[str, Any]:
#         """進行反事實分析"""
#         prompt = f"""對以下情況進行反事實分析：
#
# 查詢：{query}
# 文本：{chunk}
# 當前推理：{json.dumps(reasoning, ensure_ascii=False, indent=2)}
#
# 請考慮：
# 1. 如果關鍵信息有誤，結論是否改變？
# 2. 是否存在其他可能的解釋？
# 3. 缺少什麼關鍵信息？
# 4. 當前結論的穩健性如何？
#
# 請以JSON格式提供分析結果。"""
#
#         response = self.llm.invoke(prompt)
#         return self._parse_llm_output(response.content)
#
#     def filter_chunks(self, query: str, chunks: List[str],
#                       min_score: float = 0.6) -> List[Dict[str, Any]]:
#         """基於詳細分析過濾文檔塊"""
#         filtered_results = []
#
#         for chunk in chunks:
#             # 生成詳細推理
#             reasoning = self.generate_detailed_cot(query, chunk)
#
#             # 評估證據
#             evidence_scores = self.evaluate_evidence(chunk, reasoning)
#
#             # 進行反事實分析
#             counterfactual = self.counterfactual_analysis(query, chunk, reasoning)
#
#             # 計算綜合分數
#             overall_score = self._calculate_overall_score(evidence_scores, counterfactual)
#
#             if overall_score >= min_score:
#                 filtered_results.append({
#                     "chunk": chunk,
#                     "reasoning": reasoning,
#                     "evidence_scores": evidence_scores,
#                     "counterfactual": counterfactual,
#                     "overall_score": overall_score
#                 })
#
#         return sorted(filtered_results, key=lambda x: x["overall_score"], reverse=True)
#
#     def _format_reasoning_steps(self) -> str:
#         """格式化推理步驟"""
#         return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(self.reasoning_steps))
#
#     def _parse_reasoning_output(self, output: str) -> Dict[str, Any]:
#         """解析推理輸出"""
#         try:
#             cleaned_text = output.strip()
#             json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
#             if json_match:
#                 return json.loads(json_match.group())
#             return {"error": "無法解析推理輸出"}
#         except Exception as e:
#             logger.error(f"解析推理輸出時出錯: {e}")
#             return {"error": str(e)}
#
#     def _parse_evidence_scores(self, output: str) -> Dict[str, float]:
#         """解析證據評分"""
#         try:
#             data = self._parse_llm_output(output)
#             return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
#         except Exception as e:
#             logger.error(f"解析證據評分時出錯: {e}")
#             return {}
#
#     def _calculate_overall_score(self, evidence_scores: Dict[str, float],
#                                  counterfactual: Dict[str, Any]) -> float:
#         """計算綜合評分"""
#         # 基礎分數是證據評分的加權平均
#         weights = {
#             "relevance": 0.4,
#             "completeness": 0.3,
#             "consistency": 0.2,
#             "specificity": 0.1
#         }
#
#         base_score = sum(score * weights.get(criterion, 0)
#                          for criterion, score in evidence_scores.items())
#
#         # 根據反事實分析調整分數
#         confidence = counterfactual.get("conclusion_robustness", 0.5)
#
#         return base_score * confidence



class BM25Retriever:
    """BM25檢索器"""

    def __init__(self, ckip_processor, use_jieba: bool = False):
        self.ckip_processor = ckip_processor
        self.use_jieba = use_jieba
        if use_jieba:
            import jieba
            self.jieba = jieba
            logger.info("Using Jieba for tokenization")
        else:
            logger.info("Using CKIP for tokenization")

        self.corpus_dict = {
            'finance': {},
            'insurance': {},
            'faq': {}
        }

    def _tokenize(self, text: str) -> List[str]:
        """分詞方法"""
        if self.use_jieba:
            return list(self.jieba.cut_for_search(text))
        else:
            return self.ckip_processor.segment_parallel([text])[0].split()

    def create_store(self, category: str, texts: List[str], file_ids: List[int]) -> None:
        """保存文檔集合"""
        try:
            self.corpus_dict[category] = {
                file_id: text
                for file_id, text in zip(file_ids, texts)
            }
            logger.info(f"Saved {len(texts)} documents for category {category}")
        except Exception as e:
            logger.error(f"Error saving documents for {category}: {e}")
            raise

    def retrieve(self, query: str, source: List[str], category: str) -> int:
        """執行BM25檢索"""
        try:
            if category not in self.corpus_dict:
                raise ValueError(f"No documents found for category: {category}")

            # 只取source中的文檔
            filtered_corpus = []
            valid_source_ids = []

            for file_id in source:
                doc_id = int(file_id)
                if doc_id in self.corpus_dict[category]:
                    filtered_corpus.append(self.corpus_dict[category][doc_id])
                    valid_source_ids.append(doc_id)

            source_ids = [int(s) for s in source]

            if not filtered_corpus:
                logger.warning("No valid documents found in source")
                return source_ids[0]

            # 對篩選後的文檔進行分詞
            tokenized_corpus = [self._tokenize(doc) for doc in filtered_corpus]

            # 建立BM25模型
            bm25 = BM25Okapi(tokenized_corpus)

            # 對查詢進行分詞
            tokenized_query = self._tokenize(query)

            # 使用 get_top_n 取得最相關文檔
            top_docs = bm25.get_top_n(
                tokenized_query,
                filtered_corpus,
                n=1
            )

            if not top_docs:
                logger.warning("No results from BM25")
                return source_ids[0]

            best_doc = top_docs[0]

            # 找回文檔ID
            for i, doc_text in enumerate(filtered_corpus):
                if doc_text == best_doc:
                    retrieved_id = valid_source_ids[i]
                    logger.info(f"BM25 selected document {retrieved_id}")
                    return retrieved_id

            logger.warning("Could not map result back to document ID")
            return source_ids[0]

        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {e}")
            return source_ids[0] if source_ids else -1

    def create_stores_from_documents(self, documents: Dict[str, Dict[int, str]]) -> None:
        """從文檔集合創建所有類別的存儲"""
        for category, docs in documents.items():
            try:
                texts = list(docs.values())
                file_ids = [int(id_) for id_ in docs.keys()]
                self.create_store(category, texts, file_ids)
            except Exception as e:
                logger.error(f"Failed to create store for {category}: {e}")


# class BM25Retriever:
#     """BM25檢索器 - 依類別處理"""
#
#     def __init__(self, ckip_processor, use_jieba: bool = True):
#         import jieba
#         self.jieba = jieba
#         # 分別存儲不同類型的語料庫
#         self.corpus_dict_insurance = {}
#         self.corpus_dict_finance = {}
#         self.corpus_dict_faq = {}
#
#     def create_store(self, category: str, texts: List[str], file_ids: List[int]) -> None:
#         """根據類別保存文檔"""
#         corpus_dict = {
#             file_id: text
#             for file_id, text in zip(file_ids, texts)
#         }
#
#         if category == 'insurance':
#             self.corpus_dict_insurance = corpus_dict
#         elif category == 'finance':
#             self.corpus_dict_finance = corpus_dict
#         elif category == 'faq':
#             self.corpus_dict_faq = corpus_dict
#
#     def retrieve(self, query: str, source: List[str], category: str) -> int:
#         """
#         按照原始程式碼的邏輯處理不同類別
#         """
#         try:
#             # 根據類別選擇對應的語料庫
#             if category == 'insurance':
#                 corpus_dict = self.corpus_dict_insurance
#             elif category == 'finance':
#                 corpus_dict = self.corpus_dict_finance
#             elif category == 'faq':
#                 corpus_dict = self.corpus_dict_faq
#             else:
#                 raise ValueError(f"Unknown category: {category}")
#
#             # 只使用source中的文檔
#             filtered_corpus = [corpus_dict[int(file_id)] for file_id in source]
#
#             # BM25檢索
#             tokenized_corpus = [list(self.jieba.cut_for_search(doc)) for doc in filtered_corpus]
#             bm25 = BM25Okapi(tokenized_corpus)
#             tokenized_query = list(self.jieba.cut_for_search(query))
#             ans = bm25.get_top_n(tokenized_query, filtered_corpus, n=1)
#             best_doc = ans[0]
#
#             # 找回文檔ID
#             for file_id in source:
#                 if corpus_dict[int(file_id)] == best_doc:
#                     return int(file_id)
#
#             return int(source[0])
#
#         except Exception as e:
#             logger.error(f"Error in BM25 retrieval: {e}")
#             return int(source[0]) if source else -1
#
#     def create_stores_from_documents(self, documents: Dict[str, Dict[int, str]]) -> None:
#         """從文檔集合創建所有類別的存儲"""
#         for category, docs in documents.items():
#             texts = list(docs.values())
#             file_ids = [int(id_) for id_ in docs.keys()]
#             self.create_store(category, texts, file_ids)

class RAGSystem:
    """RAG系統主類"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.cache_manager = EnhancedCacheManager(config.cache_dir)
        self.query_processor = QueryProcessor(config)
        self.vector_store_manager = VectorStoreManager(config)
        self.document_processor = EnhancedPDFProcessor(config)
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature
        )

        # 確保必要的目錄存在
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(config.vector_store_path).mkdir(parents=True, exist_ok=True)

        self.bm25_retriever = BM25Retriever(self.query_processor.ckip_processor)
        # 添加檢索模式設置
        self.retrieval_mode = "vector"  # 默認使用向量檢索
        self.enhanced_extractor = EnhancedInformationExtractor(self.llm)
        self.enhanced_filter = EnhancedCoTFilter(self.llm)

    def create_retriever_stores(self, documents: Dict[str, Dict[int, str]], retriever_type: str = "vector") -> None:
        """創建檢索存儲"""
        try:
            # 如果是 BM25 或混合模式，創建所有類別的 BM25 存儲
            if retriever_type in ["bm25", "hybrid"]:
                logger.info("Creating BM25 stores for all categories")
                self.bm25_retriever.create_stores_from_documents(documents)

            # 如果是向量或混合模式，創建向量存儲
            if retriever_type in ["vector", "hybrid"]:
                for category, docs in documents.items():
                    logger.info(f"Creating vector store for category: {category}")
                    texts = list(docs.values())
                    file_ids = [int(id_) for id_ in docs.keys()]
                    self.vector_store_manager.create_vector_store(category, texts, file_ids)

        except Exception as e:
            logger.error(f"Error creating retriever stores: {e}")
            if retriever_type in ["bm25", "hybrid"]:
                self.bm25_retriever.clear_cache()
            raise

    def _hybrid_rerank(self, query: str, candidates: List[Tuple[int, str]], source: List[str]) -> int:
        """對混合檢索結果進行重排序"""
        try:
            # 使用現有的查詢處理器進行重排序
            expanded = self.query_processor.expand_query(query)
            contexts = [(str(doc_id), doc_id) for doc_id, _ in candidates]
            result = self.query_processor.rerank_results(
                query,
                expanded.expanded_query,
                expanded.keywords,
                contexts
                # expanded.query_type
            )
            return result.file_id

        except Exception as e:
            logger.error(f"Error in hybrid reranking: {e}")
            return candidates[0][0]  # 返回第一個候選項

    def process_documents(self, source_path: str) -> Dict[str, Dict[int, str]]:
        """處理文檔集合"""
        results = {}

        # 處理各類別文檔
        for category in ['finance', 'insurance']:
            category_path = Path(source_path) / category
            if category_path.exists():
                logger.info(f"Processing {category} documents...")
                file_paths = [str(f) for f in category_path.glob("*.pdf")]

                if file_paths:
                    try:
                        # 處理文檔並生成該類別的快取報告
                        results[category] = self.document_processor.process_documents(file_paths)

                        # 生成該類別的快取報告
                        report_dir = Path(self.config.cache_dir) / "reports" / category
                        report_dir.mkdir(parents=True, exist_ok=True)
                        self.document_processor.generate_cache_report(report_dir)
                    except Exception as e:
                        logger.error(f"Error processing {category} documents: {str(e)}", exc_info=True)
                        continue

        # 處理FAQ文檔 (保持不變)
        faq_path = Path(source_path) / 'faq' / 'pid_map_content.json'
        if faq_path.exists():
            logger.info("Processing FAQ documents...")
            try:
                with open(faq_path, 'r', encoding='utf-8') as f:
                    faq_data = json.load(f)

                logger.info(f"Loaded FAQ data with {len(faq_data)} entries")

                # 驗證 FAQ 數據格式
                if not isinstance(faq_data, dict):
                    raise ValueError(f"FAQ data should be a dictionary, got {type(faq_data)}")

                # 驗證並轉換數據
                processed_faq = {}
                for k, v in faq_data.items():
                    try:
                        int_key = int(k)
                        str_value = str(v)
                        if str_value.strip():
                            processed_faq[int_key] = str_value
                    except ValueError as e:
                        logger.warning(f"Skipping invalid FAQ entry {k}: {e}")
                        continue

                if not processed_faq:
                    logger.error("No valid FAQ entries found after processing")
                    raise ValueError("No valid FAQ entries")

                logger.info(f"Successfully processed {len(processed_faq)} FAQ entries")
                results['faq'] = processed_faq

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding FAQ JSON file: {str(e)}", exc_info=True)
            except Exception as e:
                logger.error(f"Error processing FAQ documents: {str(e)}", exc_info=True)

        # 處理完成後的清理和報告生成
        try:
            # 清理無效快取
            self.document_processor.cleanup_invalid_cache()

            # 生成最終的快取報告
            final_report_dir = Path(self.config.cache_dir) / "reports"
            self.document_processor.generate_cache_report(final_report_dir)
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")

        return results

    # def create_vector_store(
    #         self,
    #         category: str,
    #         texts: List[str],
    #         file_ids: List[int]
    # ) -> Optional[FAISS]:
    #     """
    #     創建向量存儲，包含完整的文檔處理、分塊、embedding和驗證流程
    #
    #     Args:
    #         category: 文檔類別
    #         texts: 要處理的文本列表
    #         file_ids: 文件ID列表
    #
    #     Returns:
    #         Optional[FAISS]: 創建的向量存儲，失敗時返回None
    #     """
    #     logger.info(f"Creating vector store for category: {category}")
    #     logger.info(f"Input documents: {len(texts)}, File IDs: {len(file_ids)}")
    #
    #     try:
    #         # 1. 輸入驗證
    #         if not texts or not file_ids or len(texts) != len(file_ids):
    #             raise ValueError(
    #                 f"Invalid input: texts({len(texts) if texts else 0}), "
    #                 f"file_ids({len(file_ids) if file_ids else 0})"
    #             )
    #
    #         store_path = self._get_store_path(category)
    #         logger.info(f"Vector store will be saved to: {store_path}")
    #
    #         # 2. 初始化處理統計
    #         processing_stats = {
    #             "start_time": time.time(),
    #             "total_docs": len(texts),
    #             "processed_docs": 0,
    #             "total_chunks": 0,
    #             "chunk_sizes": [],
    #             "embedding_stats": {
    #                 "total_embeddings": 0,
    #                 "cache_hits": 0,
    #                 "errors": 0
    #             }
    #         }
    #
    #         # 3. 文檔分塊處理
    #         documents = []
    #         for idx, (text, file_id) in enumerate(zip(texts, file_ids), 1):
    #             try:
    #                 # 創建基礎元數據
    #                 metadata = {
    #                     "source": file_id,
    #                     "category": category,
    #                     "original_length": len(text),
    #                     "timestamp": datetime.now().isoformat(),
    #                     "processing_id": f"{category}_{file_id}"
    #                 }
    #
    #                 # 使用分割器處理文本
    #                 doc_chunks = self.text_splitter.create_documents(
    #                     texts=[text],
    #                     metadatas=[metadata]
    #                 )
    #
    #                 # 更新統計信息
    #                 for doc in doc_chunks:
    #                     chunk_size = len(doc["page_content"])
    #                     processing_stats["chunk_sizes"].append(chunk_size)
    #                     processing_stats["total_chunks"] += 1
    #
    #                 documents.extend(doc_chunks)
    #                 processing_stats["processed_docs"] += 1
    #
    #                 # 定期記錄進度
    #                 if idx % 100 == 0:
    #                     logger.info(
    #                         f"Processed {idx}/{len(texts)} documents, "
    #                         f"generated {len(documents)} chunks"
    #                     )
    #
    #             except Exception as e:
    #                 logger.error(f"Error processing document {file_id}: {str(e)}")
    #                 continue
    #
    #         # 4. 驗證處理結果
    #         if not documents:
    #             raise ValueError("No valid documents generated after processing")
    #
    #         # 5. 計算處理統計
    #         if processing_stats["chunk_sizes"]:
    #             chunk_stats = {
    #                 "avg_size": sum(processing_stats["chunk_sizes"]) / len(processing_stats["chunk_sizes"]),
    #                 "max_size": max(processing_stats["chunk_sizes"]),
    #                 "min_size": min(processing_stats["chunk_sizes"]),
    #                 "total_chunks": len(processing_stats["chunk_sizes"])
    #             }
    #         else:
    #             chunk_stats = {
    #                 "avg_size": 0,
    #                 "max_size": 0,
    #                 "min_size": 0,
    #                 "total_chunks": 0
    #             }
    #
    #         # 6. 記錄處理統計
    #         logger.info(
    #             f"\nDocument processing statistics:\n"
    #             f"Processed documents: {processing_stats['processed_docs']}/{processing_stats['total_docs']}\n"
    #             f"Total chunks: {chunk_stats['total_chunks']}\n"
    #             f"Average chunk size: {chunk_stats['avg_size']:.2f}\n"
    #             f"Max chunk size: {chunk_stats['max_size']}\n"
    #             f"Min chunk size: {chunk_stats['min_size']}"
    #         )
    #
    #         # 7. 準備向量存儲數據
    #         chunks = [doc["page_content"] for doc in documents]
    #         metadatas = [doc["metadata"] for doc in documents]
    #
    #         # 8. 添加分塊配置到元數據
    #         for metadata in metadatas:
    #             metadata.update({
    #                 "chunk_config": {
    #                     "size": self.config.chunk_size,
    #                     "overlap": self.config.chunk_overlap
    #                 },
    #                 "embedding_model": self.embeddings.embeddings.get_model_info()
    #             })
    #
    #         # 9. 創建向量存儲
    #         try:
    #             logger.info("Creating FAISS index...")
    #             vector_store = FAISS.from_texts(
    #                 texts=chunks,
    #                 embedding=self.embeddings,
    #                 metadatas=metadatas
    #             )
    #             logger.info("FAISS index created successfully")
    #         except Exception as e:
    #             logger.error(f"Error creating FAISS index: {e}")
    #             raise
    #
    #         # 10. 驗證向量存儲
    #         if not hasattr(vector_store, 'index') or not hasattr(vector_store, 'docstore'):
    #             raise ValueError("Created vector store missing required attributes")
    #
    #         if not vector_store.index or vector_store.index.ntotal == 0:
    #             raise ValueError("Created vector store index is empty")
    #
    #         # 11. 記錄向量存儲統計
    #         store_stats = {
    #             "total_vectors": vector_store.index.ntotal,
    #             "dimension": vector_store.index.d,
    #             "total_documents": len(vector_store.docstore._dict)
    #         }
    #
    #         logger.info(
    #             f"\nVector store statistics:\n"
    #             f"Total vectors: {store_stats['total_vectors']}\n"
    #             f"Vector dimension: {store_stats['dimension']}\n"
    #             f"Total documents: {store_stats['total_documents']}"
    #         )
    #
    #         # 12. 保存向量存儲
    #         try:
    #             if store_path.exists():
    #                 logger.info(f"Removing existing store at {store_path}")
    #                 shutil.rmtree(str(store_path))
    #
    #             store_path.mkdir(parents=True)
    #             vector_store.save_local(str(store_path))
    #             logger.info(f"Vector store saved to {store_path}")
    #
    #             # 13. 保存完整的配置和統計信息
    #             metadata = {
    #                 "creation_time": datetime.now().isoformat(),
    #                 "processing_time": time.time() - processing_stats["start_time"],
    #                 "chunk_config": {
    #                     "chunk_size": self.config.chunk_size,
    #                     "chunk_overlap": self.config.chunk_overlap
    #                 },
    #                 "model_config": {
    #                     "ckip_model": self.config.ckip_model,
    #                     "use_delim": self.config.use_delim
    #                 },
    #                 "embedding_config": self.embeddings.embeddings.get_model_info(),
    #                 "processing_stats": processing_stats,
    #                 "chunk_stats": chunk_stats,
    #                 "store_stats": store_stats,
    #                 "version": "1.0"
    #             }
    #
    #             with open(store_path / "config.json", "w", encoding="utf-8") as f:
    #                 json.dump(metadata, f, ensure_ascii=False, indent=2)
    #
    #             logger.info("Configuration and statistics saved")
    #
    #             return vector_store
    #
    #         except Exception as e:
    #             logger.error(f"Error saving vector store: {e}")
    #             if store_path.exists():
    #                 try:
    #                     shutil.rmtree(str(store_path))
    #                 except Exception as cleanup_error:
    #                     logger.error(f"Error cleaning up failed store: {cleanup_error}")
    #             raise
    #
    #     except Exception as e:
    #         logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
    #         return None

    # def _vector_retrieve(
    #         self,
    #         query: str,
    #         source: List[str],
    #         vector_store: FAISS,
    #         category: str = None,
    #         top_k: int = 8
    # ) -> int:
    #     """優化的向量檢索方法"""
    #     logger.info(f"\nStarting vector retrieval for category: {category}")
    #     logger.info(f"Query: {query}")
    #
    #     try:
    #         source_ids = [int(s) for s in source]
    #
    #         # 根據類別設置檢索參數
    #         if category == 'insurance':
    #             # 保險文檔的特定模式
    #             query_patterns = {
    #                 'time_req': r'[多少|幾](?:天|日|年|個月)|何時|時間|期限',
    #                 'obligation': r'[應該|必須|可以|不得|是否].*?(?:通知|同意|申請|檢具|給付)',
    #                 'payment': r'(?:保險金|費用).*?(?:給付|計算|金額)',
    #                 'procedure': r'(?:如何|應該怎麼).*?(?:申請|檢具|通知|變更)',
    #                 'document': r'(?:文件|證明|資料).*?(?:檢具|提供|通知)'
    #             }
    #             boost_factor = 0.15
    #             min_score_threshold = 0.8
    #
    #         elif category == 'finance':
    #             # 財務文檔的特定模式
    #             query_patterns = {
    #                 # 財務數據查詢模式
    #                 'amount': r'(?:金額|總額|金[額计]為?)[是為有]?[多少幾]|[是為有]多少',
    #                 'percentage': r'[百分之|佔]+.*?[比率|比例]',
    #                 'change': r'[增加|減少|變動].*?[多少|幾]',
    #                 # 財務時期模式
    #                 'period': r'(?:第[一二三四]季|[0-9]{3}年第[一二三四]季|年度)',
    #                 'year_month': r'(?:民國)?(?:[0-9]{2,3}|一一[0-9一二三四五六七八九十])?年(?:[0-9]{1,2}|[一二三四五六七八九十]+)?月',
    #                 # 財務科目模式
    #                 'pl_items': r'(?:營業)?(?:收入|利益|[毛營淨]利|損益|虧損|支出|費用)',
    #                 'bs_items': r'(?:資產|負債|權益|存貨|應收|應付|現金|約當現金)',
    #                 'cf_items': r'(?:營業|投資|籌資).*?(?:活動|現金流[入出])'
    #             }
    #             boost_factor = 0.12
    #             min_score_threshold = 0.65
    #         else:
    #             query_patterns = {}
    #             boost_factor = 0
    #             min_score_threshold = 0.9
    #
    #         # 分析查詢中的模式
    #         pattern_matches = {
    #             name: bool(re.search(pattern, query))
    #             for name, pattern in query_patterns.items()
    #         }
    #         matched_patterns = sum(pattern_matches.values())
    #
    #         # 財務查詢的特殊處理
    #         if category == 'finance':
    #             # 檢測是否是精確數字查詢
    #             has_exact_number = bool(re.search(r'[0-9]+(?:\.[0-9]+)?', query))
    #             has_financial_term = any(
    #                 re.search(pattern, query)
    #                 for pattern in [r'營業收入', r'淨利', r'資產總額', r'權益總額', r'現金流[入出]']
    #             )
    #
    #             # 對於精確數字查詢，增加相關模式的權重
    #             if has_exact_number and has_financial_term:
    #                 boost_factor *= 1.2
    #                 min_score_threshold = 0.6
    #
    #         # 使用 QueryProcessor 處理查詢
    #         expanded_query, keywords, query_embedding = self.query_processor.process_query(
    #             query, source, category
    #         )
    #
    #         # 檢索結果
    #         candidates = []
    #         for source_id in source_ids:
    #             try:
    #                 results = vector_store.similarity_search_with_score_by_vector(
    #                     query_embedding,
    #                     k=top_k,
    #                     filter={"source": source_id}
    #                 )
    #
    #                 if not results:
    #                     continue
    #
    #                 for doc, base_score in results:
    #                     doc_text = doc.page_content
    #
    #                     # 計算模式匹配分數
    #                     doc_pattern_matches = sum(
    #                         1 for name, pattern in query_patterns.items()
    #                         if pattern_matches[name] and re.search(pattern, doc_text)
    #                     )
    #
    #                     # 財務文檔的特殊分數調整
    #                     if category == 'finance':
    #                         # 檢查是否包含相關數字
    #                         if has_exact_number and re.search(r'[0-9]+(?:\.[0-9]+)?', doc_text):
    #                             doc_pattern_matches += 1
    #
    #                         # 檢查是否包含相關期間
    #                         if re.search(r'第[一二三四]季', query) and re.search(r'第[一二三四]季', doc_text):
    #                             doc_pattern_matches += 1
    #
    #                     # 根據模式匹配調整分數
    #                     if doc_pattern_matches > 0:
    #                         pattern_boost = boost_factor * doc_pattern_matches
    #                         adjusted_score = base_score * (1 - pattern_boost)
    #                     else:
    #                         adjusted_score = base_score
    #
    #                     # 只保留高於閾值的結果
    #                     if adjusted_score < min_score_threshold:
    #                         candidates.append({
    #                             'doc_id': source_id,
    #                             'content': doc_text,
    #                             'score': adjusted_score,
    #                             'metadata': doc.metadata,
    #                             'pattern_matches': doc_pattern_matches
    #                         })
    #
    #             except Exception as e:
    #                 logger.error(f"Search failed for document {source_id}: {e}")
    #                 continue
    #
    #         if not candidates:
    #             return source_ids[0]
    #
    #         # 選擇最佳候選文件
    #         candidates.sort(key=lambda x: (x['score'], -x['pattern_matches']))
    #         best_candidate = candidates[0]
    #
    #         logger.info(f"Selected document {best_candidate['doc_id']}")
    #         logger.info(f"Score: {best_candidate['score']:.4f}")
    #         logger.info(f"Pattern matches: {best_candidate['pattern_matches']}")
    #
    #         return best_candidate['doc_id']
    #
    #     except Exception as e:
    #         logger.error(f"Error in vector retrieve: {e}")
    #         return source_ids[0] if source_ids else -1

    # def _vector_retrieve(self, query: str, source: List[str], vector_store: FAISS, category: str = None,
    #                      top_k: int = 8) -> int:
    #     """優化的向量檢索方法"""
    #     logger.info(f"\nStarting vector retrieval for category: {category}")
    #     logger.info(f"Query: {query}")
    #
    #     try:
    #         source_ids = [int(s) for s in source]
    #
    #         # 設置默認參數
    #         query_patterns = {}
    #         boost_factor = 0
    #         min_score_threshold = 0.9
    #
    #         # 根據類別設置檢索參數
    #         if category == 'insurance':
    #             # 保險文檔的特定模式
    #             query_patterns = {
    #                 'time_req': r'[多少|幾](?:天|日|年|個月)|何時|時間|期限',
    #                 'obligation': r'[應該|必須|可以|不得|是否].*?(?:通知|同意|申請|檢具|給付)',
    #                 'payment': r'(?:保險金|費用).*?(?:給付|計算|金額)',
    #                 'procedure': r'(?:如何|應該怎麼).*?(?:申請|檢具|通知|變更)',
    #                 'document': r'(?:文件|證明|資料).*?(?:檢具|提供|通知)'
    #             }
    #             boost_factor = 0.15
    #             min_score_threshold = 0.8
    #
    #         elif category == 'finance':
    #             # 檢測精確數字
    #             numbers = re.findall(r'[-+]?\d*\.?\d+', query)
    #             exact_match_boost = 0.3
    #
    #             # 財務時期模式
    #             period_patterns = {
    #                 'quarter': r'第[一二三四]季|Q[1-4]|[1-4]Q',
    #                 'year': r'\d{3}年|\d{4}年',
    #                 'month': r'\d{1,2}月'
    #             }
    #
    #             # 財務科目模式
    #             account_patterns = {
    #                 'balance_sheet': r'資產|負債|權益|存貨|應收|應付|現金|約當現金',
    #                 'income': r'營業收入|營業費用|營業利益|稅前淨利|本期淨利',
    #                 'cash_flow': r'營業活動|投資活動|籌資活動|現金流[入出]'
    #             }
    #
    #             # 財務查詢的特定模式
    #             query_patterns = {
    #                 'amount': r'(?:金額|總額|金[額计]為?)[是為有]?[多少幾]|[是為有]多少',
    #                 'percentage': r'[百分之|佔]+.*?[比率|比例]',
    #                 'change': r'[增加|減少|變動].*?[多少|幾]',
    #                 'period': r'(?:第[一二三四]季|[0-9]{3}年第[一二三四]季|年度)',
    #                 'year_month': r'(?:民國)?(?:[0-9]{2,3}|一一[0-9一二三四五六七八九十])?年(?:[0-9]{1,2}|[一二三四五六七八九十]+)?月',
    #                 'pl_items': r'(?:營業)?(?:收入|利益|[毛營淨]利|損益|虧損|支出|費用)',
    #                 'bs_items': r'(?:資產|負債|權益|存貨|應收|應付|現金|約當現金)',
    #                 'cf_items': r'(?:營業|投資|籌資).*?(?:活動|現金流[入出])'
    #             }
    #             boost_factor = 0.12
    #             min_score_threshold = 0.65
    #
    #         # 使用 QueryProcessor 處理查詢
    #         expanded_query, keywords, query_embedding = self.query_processor.process_query(
    #             query, source, category
    #         )
    #
    #         # 檢索結果
    #         candidates = []
    #
    #         # 分析查詢中的模式
    #         pattern_matches = {
    #             name: bool(re.search(pattern, query))
    #             for name, pattern in query_patterns.items()
    #         }
    #         matched_patterns = sum(pattern_matches.values())
    #
    #         for source_id in source_ids:
    #             try:
    #                 results = vector_store.similarity_search_with_score_by_vector(
    #                     query_embedding,
    #                     k=top_k,
    #                     filter={"source": source_id}
    #                 )
    #
    #                 if not results:
    #                     continue
    #
    #                 for doc, base_score in results:
    #                     doc_text = doc.page_content
    #                     adjusted_score = base_score
    #                     doc_pattern_matches = 0
    #
    #                     if category == 'finance':
    #                         # 財務類別的特殊處理
    #                         adjustment = 1.0
    #
    #                         # 檢查精確數字匹配
    #                         if numbers:
    #                             for number in numbers:
    #                                 if number in doc_text:
    #                                     adjustment -= exact_match_boost
    #
    #                         # 檢查時期匹配
    #                         for period_type, pattern in period_patterns.items():
    #                             if re.search(pattern, query) and re.search(pattern, doc_text):
    #                                 adjustment -= 0.1
    #
    #                         # 檢查財務科目匹配
    #                         for account_type, pattern in account_patterns.items():
    #                             if re.search(pattern, query) and re.search(pattern, doc_text):
    #                                 adjustment -= 0.05
    #
    #                         adjusted_score = base_score * adjustment
    #
    #                     else:
    #                         # 其他類別的模式匹配處理
    #                         doc_pattern_matches = sum(
    #                             1 for name, pattern in query_patterns.items()
    #                             if pattern_matches[name] and re.search(pattern, doc_text)
    #                         )
    #
    #                         if doc_pattern_matches > 0:
    #                             pattern_boost = boost_factor * doc_pattern_matches
    #                             adjusted_score = base_score * (1 - pattern_boost)
    #
    #                     # 只保留高於閾值的結果
    #                     if adjusted_score < min_score_threshold:
    #                         candidates.append({
    #                             'doc_id': source_id,
    #                             'content': doc_text,
    #                             'score': adjusted_score,
    #                             'metadata': doc.metadata,
    #                             'pattern_matches': doc_pattern_matches
    #                         })
    #
    #             except Exception as e:
    #                 logger.error(f"Search failed for document {source_id}: {e}")
    #                 continue
    #
    #         if not candidates:
    #             return source_ids[0]
    #
    #         # 選擇最佳候選文件
    #         candidates.sort(key=lambda x: (x['score'], -x['pattern_matches']))
    #         best_candidate = candidates[0]
    #
    #         logger.info(f"Selected document {best_candidate['doc_id']}")
    #         logger.info(f"Score: {best_candidate['score']:.4f}")
    #         logger.info(f"Pattern matches: {best_candidate['pattern_matches']}")
    #
    #         return best_candidate['doc_id']
    #
    #     except Exception as e:
    #         logger.error(f"Error in vector retrieve: {e}")
    #         return source_ids[0] if source_ids else -1

    def _analyze_query_params(self, query: str, category: str) -> Dict[str, float]:
        """分析查詢並返回最佳參數配置"""
        params = {
            'exact_match_boost': 0.3,
            'min_score_threshold': 0.7,
            'boost_factor': 0.15
        }

        if category == 'finance':
            # 財報數字查詢的特殊處理
            if re.search(r'(合併)?資產負債表|損益表|現金流量表', query):
                if re.search(r'[金額|總額|為].*?多少', query):
                    params.update({
                        'exact_match_boost': 0.6,
                        'min_score_threshold': 0.5,  # 降低閾值以包含更多相關結果
                        'boost_factor': 0.3
                    })

                    # 如果包含具體財務項目，進一步調整
                    if re.search(r'現金及約當現金|資產總額|負債總額', query):
                        params['exact_match_boost'] = 0.7

                    # 如果包含明確的時期，增加權重
                    if re.search(r'\d+年第[一二三四]季', query):
                        params['boost_factor'] = 0.35

        return params

    def _get_special_patterns(self, category: str) -> Dict[str, str]:
        """獲取特定類別的核心模式"""
        if category == 'insurance':
            return {
                'time_notice': r'[多少|幾].*?[天日].*?通知|通知.*?[多少|幾].*?[天日]',
                'policy_change': r'改為|變更為|轉換.*?(展期|定期|繳清).*?保險',
                'benefit_loss': r'是否.*?適用|能否.*?給付|喪失.*?權利'
            }
        elif category == 'finance':
            return {
                'time_specific': r'\d{3}年第[一二三四]季|\d{4}年第[一二三四]季',
                'litigation': r'訴訟.*?專利號|專利.*?侵權',
                'department': r'部門|集團|體系'
            }
        return {}

    # def _adjust_hybrid_weights(self, query: str, category: str) -> Dict[str, float]:
    #     """根據不同類別和查詢類型動態調整混合檢索權重"""
    #     if category == 'finance':
    #         if re.search(r'(合併)?資產負債表|損益表|現金流量表', query):
    #             if re.search(r'[金額|總額|為].*?多少', query):
    #                 return {
    #                     'vector': 0.5,  # 降低向量搜索權重
    #                     'bm25': 0.5  # 增加精確匹配權重
    #                 }
    #             return {
    #                 'vector': 0.6,
    #                 'bm25': 0.4
    #             }
    #         elif re.search(r'\d+年第[一二三四]季|\d+年度', query):
    #             return {
    #                 'vector': 0.7,
    #                 'bm25': 0.3
    #             }
    #         # 其他財務查詢
    #         return {
    #             'vector': 0.8,
    #             'bm25': 0.2
    #         }
    #
    #     elif category == 'insurance':
    #         # 保單條款相關查詢
    #         if re.search(r'(改為|變更為|轉換).*(定期|展期|繳清)', query):
    #             return {
    #                 'vector': 0.6,
    #                 'bm25': 0.4  # 增加BM25權重以確保精確匹配
    #             }
    #         # 金額計算相關
    #         elif re.search(r'(金額|價值).*(計算|辦理|給付)', query):
    #             return {
    #                 'vector': 0.65,
    #                 'bm25': 0.35
    #             }
    #         # 權限相關
    #         elif re.search(r'(變更|指定|同意|通知)', query):
    #             weights = {'vector': 0.7, 'bm25': 0.3}
    #             if '受益人' in query:  # 特別處理受益人相關查詢
    #                 weights['bm25'] = 0.4
    #                 weights['vector'] = 0.6
    #             return weights
    #         # 期限相關
    #         elif re.search(r'[多少|幾].*?(天|日|年|月)|期限', query):
    #             return {
    #                 'vector': 0.75,
    #                 'bm25': 0.25
    #             }
    #         # 預設保險權重
    #         return {
    #             'vector': 0.9,
    #             'bm25': 0.1
    #         }
    #
    #     # FAQ保持原有權重
    #     elif category == 'faq':
    #         return {'vector': 1.0, 'bm25': 0.0}
    #
    #     # 默認配置
    #     return {'vector': 0.8, 'bm25': 0.2}

    def _calculate_context_weight(self, query: str) -> float:
        """計算上下文的重要性權重"""
        # 時序比較相關
        time_patterns = [
            r'之前|以前',  # 過去
            r'之後|以後',  # 未來
            r'期間|過程中',  # 期間
            r'前後|前期|後期',  # 相對時間
            r'當時|同時',  # 時點
        ]

        # 因果關係相關
        causal_patterns = [
            r'原因|因為|由於',  # 原因
            r'所以|因此|導致',  # 結果
            r'影響|造成|引起',  # 影響
            r'如果|假如|若|倘若',  # 條件
            r'才能|必須|需要'  # 必要條件
        ]

        # 邏輯關係相關
        logic_patterns = [
            r'而且|並且|同時',  # 並列
            r'或者|或是|還是',  # 選擇
            r'但是|然而|不過',  # 轉折
            r'除了|另外|此外'  # 補充
        ]

        weight = 0.0

        # 檢查時序模式
        for pattern in time_patterns:
            if re.search(pattern, query):
                weight += 0.15
                break

        # 檢查因果模式
        for pattern in causal_patterns:
            if re.search(pattern, query):
                weight += 0.2
                break

        # 檢查邏輯關係
        for pattern in logic_patterns:
            if re.search(pattern, query):
                weight += 0.1
                break

        # 確保最終權重不會過大
        return min(weight, 0.3)

    def _adjust_hybrid_weights(self, query: str, category: str) -> Dict[str, float]:
        """根據不同類別和查詢類型動態調整混合檢索權重"""
        # 獲取基礎權重
        base_weights = {}

        if category == 'finance':
            if re.search(r'(合併)?資產負債表|損益表|現金流量表', query):
                if re.search(r'[金額|總額|為].*?多少', query):
                    base_weights = {
                        'vector': 0.5,
                        'bm25': 0.5
                    }
                else:
                    base_weights = {
                        'vector': 0.6,
                        'bm25': 0.4
                    }
            elif re.search(r'\d+年第[一二三四]季|\d+年度', query):
                base_weights = {
                    'vector': 0.7,
                    'bm25': 0.3
                }
            else:
                base_weights = {
                    'vector': 0.8,
                    'bm25': 0.2
                }

        elif category == 'insurance':
            if re.search(r'(改為|變更為|轉換).*(定期|展期|繳清)', query):
                base_weights = {
                    'vector': 0.6,
                    'bm25': 0.4
                }
            elif re.search(r'(金額|價值).*(計算|辦理|給付)', query):
                base_weights = {
                    'vector': 0.65,
                    'bm25': 0.35
                }
            elif re.search(r'(變更|指定|同意|通知)', query):
                base_weights = {'vector': 0.7, 'bm25': 0.3}
                if '受益人' in query:
                    base_weights = {
                        'vector': 0.6,
                        'bm25': 0.4
                    }
            elif re.search(r'[多少|幾].*?(天|日|年|月)|期限', query):
                base_weights = {
                    'vector': 0.75,
                    'bm25': 0.25
                }
            else:
                base_weights = {
                    'vector': 0.9,
                    'bm25': 0.1
                }

        elif category == 'faq':
            base_weights = {'vector': 1.0, 'bm25': 0.0}
        else:
            base_weights = {'vector': 0.8, 'bm25': 0.2}

        # 獲取上下文權重調整
        context_weight = self._calculate_context_weight(query)

        if context_weight > 0:
            # 增加向量搜索的權重來捕捉上下文關係
            vector_weight = min(base_weights['vector'] + context_weight, 0.95)
            bm25_weight = 1 - vector_weight

            return {
                'vector': vector_weight,
                'bm25': bm25_weight
            }

        return base_weights

    def _get_special_patterns(self, category: str) -> Dict[str, str]:
        """獲取特定類別的模式"""
        if category == 'finance':
            return {
                'financial_item': r'(營業活動|投資活動|籌資活動|現金流[入出]|股利|淨利)',
                'number_query': r'[金額|總額|為].*?[多少|幾]',
                'period': r'第[一二三四]季|[1-4]Q|\d{2,3}年'
            }
        elif category == 'insurance':
            return {
                'policy_change': r'(改為|變更為|轉換)',
                'insurance_type': r'(定期|展期|繳清)保險',
                'amount_calc': r'(金額|價值).*(計算|辦理|給付)',
                'specific_terms': r'(保單價值準備金|欠繳保險費|借款本息|墊繳保險費)'
            }
        return {}

    def _extract_global_info(self, query: str, source_ids: List[int], vector_store: FAISS,
                             query_embedding: List[float]) -> str:
        """改進的全局信息提取"""
        try:
            # 1. 首先從每個源文檔獲取完整內容
            source_docs = []
            for doc_id in source_ids:
                # 獲取完整原始文檔而不是chunks
                results = vector_store.similarity_search_with_score_by_vector(
                    query_embedding,
                    k=1,
                    filter={"source": doc_id}
                )
                if results:
                    source_docs.append(results[0][0].page_content)
            if re.search(r'(合併)?資產負債表|損益表|現金流量表', query):
                global_prompt = f"""基於以下財務報表,分析:
                        1. 報表類型與時期
                        2. 相關科目與項目
                        3. 金額的計算邏輯
                        4. 各項目間的關係

                        文檔內容: {" ".join(source_docs)}
                        問題: {query}

                        請提供完整且結構化的財報資訊摘要。"""
            else:
                # 2. 構建提示來提取全局信息
                global_prompt = f"""基於以下完整文檔,提取回答問題必要的:
            1. 關鍵背景信息
            2. 文檔間的邏輯關係
            3. 重要的上下文聯繫

            文檔內容: {" ".join(source_docs)}
            問題: {query}

            請提供簡潔但完整的全局信息總結。"""

            # 3. 使用LLM提取全局信息
            global_info = self.llm.invoke(global_prompt).content
            return global_info
        except Exception as e:
            logger.error(f"Error extracting global info: {e}")
            return ""

    #
    def _filter_with_cot(self, query: str, chunks: List[Tuple], global_info: str) -> List[dict]:
        """改進的CoT引導過濾"""
        try:
            # 1. 生成思維鏈
            cot_prompt = f"""基於問題和背景信息,分析解答需要什麼關鍵事實:

            問題: {query}
            背景信息: {global_info}

            請按照以下步驟分析:
            1. 確定問題類型和關鍵要素
            2. 列出找到答案所需的事實細節
            3. 描述這些事實之間的關聯

            以上述邏輯提供思考步驟。"""

            cot = self.llm.invoke(cot_prompt).content

            # 2. 使用思維鏈評估每個chunk
            filtered_chunks = []
            for doc, score in chunks:
                eval_prompt = f"""基於思考步驟,評估此文本段落:

                思考步驟: {cot}
                文本段落: {doc.page_content}

                請判斷:
                1. 此段落是否包含必要的事實細節
                2. 這些細節如何對應到思考步驟
                3. 與其他已知信息的關聯度

                如果確實包含關鍵信息,回答"True",否則回答"False"。"""

                is_relevant = "true" in self.llm.invoke(eval_prompt).content.lower()

                if is_relevant:
                    filtered_chunks.append({
                        'content': doc.page_content,
                        'score': score,
                        'reasoning': cot  # 保存推理過程
                    })

            return filtered_chunks
        except Exception as e:
            logger.error(f"Error in CoT filtering: {e}")
            return []

    def _generate_answer(self, query: str, global_info: str, filtered_chunks: List[dict]) -> int:
        """整合全局和局部信息生成答案"""
        try:
            # 組織局部詳細信息
            local_details = "\n".join(chunk['content'] for chunk in filtered_chunks)

            # 構建生成提示
            gen_prompt = f"""基於全局背景和局部細節回答問題:

            問題: {query}

            全局背景信息:
            {global_info}

            相關事實細節:
            {local_details}

            請整合以上信息,重點關注:
            1. 全局背景下的邏輯一致性
            2. 細節信息的準確性
            3. 答案的完整性

            請給出最相關文檔的ID。"""

            response = self.llm.invoke(gen_prompt)

            # 解析回應獲取最相關文檔ID
            result = self._parse_llm_output(response.content)
            return result['selected_id']

        except Exception as e:
            logger.error(f"Error in answer generation: {e}")
            return filtered_chunks[0]['doc_id'] if filtered_chunks else -1

    # def _vector_retrieve(self, query: str, source: List[str], vector_store: FAISS, category: str = None) -> int:
    #     """優化的向量檢索方法，整合 LongRAG 的雙視角機制和增強分析"""
    #     logger.info(f"\nStarting vector retrieval for category: {category}")
    #     logger.info(f"Query: {query}")
    #
    #     try:
    #         if category not in ['finance', 'insurance']:
    #             # FAQ等短文本使用原始檢索方式
    #             return self._original_vector_retrieve(query, source, vector_store)
    #
    #         # 1. 基礎查詢處理
    #         expanded_query, keywords, query_embedding = self.query_processor.process_query(
    #             query, source, category
    #         )
    #         source_ids = [int(s) for s in source]
    #
    #         # 2. 初始檢索 - 獲取候選文檔
    #         initial_candidates = []
    #         for doc_id in source_ids:
    #             try:
    #                 results = vector_store.similarity_search_with_score_by_vector(
    #                     query_embedding,
    #                     k=3,  # 每個文檔取前3個最相關片段
    #                     filter={"source": doc_id}
    #                 )
    #                 if results:
    #                     initial_candidates.extend(results)
    #             except Exception as e:
    #                 logger.error(f"Error retrieving initial chunks for doc {doc_id}: {e}")
    #                 continue
    #
    #         if not initial_candidates:
    #             return source_ids[0]
    #
    #         # 3. Information Extractor (E): 分層信息提取
    #         try:
    #             doc_level_info = {}  # 存儲文檔級別信息
    #             for doc_id in source_ids:
    #                 results = vector_store.similarity_search_with_score_by_vector(
    #                     query_embedding,
    #                     k=1,
    #                     filter={"source": doc_id}
    #                 )
    #                 if results:
    #                     doc_text = results[0][0].page_content
    #                     # 調用增強型提取器
    #                     doc_info = self.enhanced_extractor.extract_document_level_info(
    #                         doc_text=doc_text,
    #                         query=query
    #                     )
    #                     doc_level_info[doc_id] = doc_info
    #
    #             # 分析文檔間關係
    #             cross_doc_analysis = self.enhanced_extractor.analyze_cross_doc_relations(
    #                 list(doc_level_info.values())
    #             )
    #
    #             # 提取關鍵句子
    #             key_sentences = []
    #             for doc_id, info in doc_level_info.items():
    #                 extracted_sentences = self.enhanced_extractor.extract_key_sentences(
    #                     text=info.get('raw_text', ''),
    #                     query=query
    #                 )
    #                 key_sentences.extend(extracted_sentences)
    #
    #             # 生成結構化全局信息
    #             global_info = self.enhanced_extractor.create_structured_summary(
    #                 doc_level_info,
    #                 cross_doc_analysis,
    #                 key_sentences
    #             )
    #
    #         except Exception as e:
    #             logger.error(f"Error in information extraction: {e}")
    #             global_info = {"error": str(e)}
    #
    #         # 4. CoT-guided Filter (F): 細粒度過濾
    #         try:
    #             candidate_chunks = [
    #                 {
    #                     'content': doc.page_content,
    #                     'score': score,
    #                     'metadata': doc.metadata
    #                 }
    #                 for doc, score in initial_candidates
    #             ]
    #
    #             # 使用增強型過濾器進行分析
    #             filtered_results = self.enhanced_filter.filter_chunks(
    #                 query=query,
    #                 chunks=[chunk['content'] for chunk in candidate_chunks],
    #                 min_score=0.6  # 可調整的閾值
    #             )
    #
    #             if not filtered_results:
    #                 logger.warning("No chunks passed enhanced filtering")
    #                 return source_ids[0]
    #
    #         except Exception as e:
    #             logger.error(f"Error in CoT filtering: {e}")
    #             filtered_results = []
    #
    #         # 5. 最終決策邏輯
    #         try:
    #             if filtered_results:
    #                 # 獲取最高評分的結果
    #                 best_result = filtered_results[0]
    #
    #                 # 根據全局信息和局部分析做出決策
    #                 decision_prompt = f"""根據以下信息選擇最相關的文檔：
    #
    # 查詢問題：{query}
    #
    # 全局信息摘要：
    # {json.dumps(global_info, ensure_ascii=False, indent=2)}
    #
    # 最佳候選文檔分析：
    # {json.dumps(best_result, ensure_ascii=False, indent=2)}
    #
    # 請選擇最相關的文檔ID，格式如下：
    # {{"selected_id": <document_id>, "confidence": <float between 0 and 1>}}"""
    #
    #                 response = self.llm.invoke(decision_prompt)
    #                 decision = self._parse_llm_output(response.content)
    #
    #                 # 驗證決策結果
    #                 selected_id = int(decision.get('selected_id', source_ids[0]))
    #                 if selected_id in source_ids:
    #                     logger.info(f"Selected document {selected_id} with enhanced analysis")
    #                     return selected_id
    #
    #             # 如果沒有有效的過濾結果或決策結果，回退到基於分數的選擇
    #             best_candidate = sorted(candidate_chunks, key=lambda x: x['score'])[0]
    #             source_id = best_candidate['metadata'].get('source', source_ids[0])
    #             logger.info(f"Falling back to score-based selection: {source_id}")
    #             return source_id
    #
    #         except Exception as e:
    #             logger.error(f"Error in final decision making: {e}")
    #             return source_ids[0]
    #
    #     except Exception as e:
    #         logger.error(f"Error in vector retrieve: {e}")
    #         return source_ids[0] if source_ids else -1

    def _vector_retrieve(self, query: str, source: List[str], vector_store: FAISS, category: str = None) -> int:
        """優化的向量檢索方法，整合 LongRAG 的雙視角機制和增強分析"""
        logger.info(f"\nStarting vector retrieval for category: {category}")
        logger.info(f"Query: {query}")

        try:
            if category not in ['finance', 'insurance']:
                # FAQ等短文本使用原始檢索方式
                return self._original_vector_retrieve(query, source, vector_store)

            # 1. 基礎查詢處理
            expanded_query, keywords, query_embedding = self.query_processor.process_query(
                query, source, category
            )
            source_ids = [int(s) for s in source]

            # 2. 初始檢索 - 獲取候選文檔
            initial_candidates = []
            for doc_id in source_ids:
                try:
                    results = vector_store.similarity_search_with_score_by_vector(
                        query_embedding,
                        k=3,  # 每個文檔取前3個最相關片段
                        filter={"source": doc_id}
                    )
                    if results:
                        initial_candidates.extend(results)
                except Exception as e:
                    logger.error(f"Error retrieving initial chunks for doc {doc_id}: {e}")
                    continue

            if not initial_candidates:
                return source_ids[0]

            # 3. Information Extractor (E): 分層信息提取
            try:
                doc_level_info = {}  # 存儲文檔級別信息
                for doc_id in source_ids:
                    results = vector_store.similarity_search_with_score_by_vector(
                        query_embedding,
                        k=1,
                        filter={"source": doc_id}
                    )
                    if results:
                        doc_text = results[0][0].page_content
                        # 調用增強型提取器
                        doc_info = self.enhanced_extractor.extract_document_level_info(
                            doc_text=doc_text,
                            query=query
                        )
                        doc_level_info[doc_id] = doc_info

                # 分析文檔間關係
                cross_doc_analysis = self.enhanced_extractor.analyze_cross_doc_relations(
                    list(doc_level_info.values())
                )

                # 提取關鍵句子
                key_sentences = []
                for doc_id, info in doc_level_info.items():
                    extracted_sentences = self.enhanced_extractor.extract_key_sentences(
                        text=info.get('raw_text', ''),
                        query=query
                    )
                    key_sentences.extend(extracted_sentences)

                # 生成結構化全局信息
                global_info = self.enhanced_extractor.create_structured_summary(
                    doc_level_info,
                    cross_doc_analysis,
                    key_sentences
                )

            except Exception as e:
                logger.error(f"Error in information extraction: {e}")
                global_info = {"error": str(e)}

            # 4. CoT-guided Filter (F): 細粒度過濾
            try:
                candidate_chunks = [
                    {
                        'content': doc.page_content,
                        'score': score,
                        'metadata': doc.metadata
                    }
                    for doc, score in initial_candidates
                ]

                # 使用增強型過濾器進行分析
                filtered_results = self.enhanced_filter.filter_chunks(
                    query=query,
                    chunks=[chunk['content'] for chunk in candidate_chunks],
                    min_score=0.6  # 可調整的閾值
                )

                if not filtered_results:
                    logger.warning("No chunks passed enhanced filtering")
                    return source_ids[0]

            except Exception as e:
                logger.error(f"Error in CoT filtering: {e}")
                filtered_results = []

            # 5. 最終決策邏輯
            try:
                if filtered_results:
                    # 獲取最高評分的結果
                    best_result = filtered_results[0]

                    # 根據全局信息和局部分析做出決策
                    decision_prompt = f"""根據以下信息選擇最相關的文檔：

    查詢問題：{query}

    全局信息摘要：
    {json.dumps(global_info, ensure_ascii=False, indent=2)}

    最佳候選文檔分析：
    {json.dumps(best_result, ensure_ascii=False, indent=2)}

    請選擇最相關的文檔ID，格式如下：
    {{"selected_id": <document_id>, "confidence": <float between 0 and 1>}}"""

                    response = self.llm.invoke(decision_prompt)
                    decision = self._parse_llm_output(response.content)

                    # 驗證決策結果
                    selected_id = int(decision.get('selected_id', source_ids[0]))
                    if selected_id in source_ids:
                        logger.info(f"Selected document {selected_id} with enhanced analysis")
                        return selected_id

                # 如果沒有有效的過濾結果或決策結果，回退到基於分數的選擇
                best_candidate = sorted(candidate_chunks, key=lambda x: x['score'])[0]
                source_id = best_candidate['metadata'].get('source', source_ids[0])
                logger.info(f"Falling back to score-based selection: {source_id}")
                return source_id

            except Exception as e:
                logger.error(f"Error in final decision making: {e}")
                return source_ids[0]

        except Exception as e:
            logger.error(f"Error in vector retrieve: {e}")
            return source_ids[0] if source_ids else -1

    def _original_vector_retrieve(self, query: str, source: List[str], vector_store: FAISS, category: str = None,
                                  top_k: int = 8) -> int:
        """優化的向量檢索方法"""
        logger.info(f"\nStarting vector retrieval for category: {category}")
        logger.info(f"Query: {query}")

        try:
            patterns = self._get_special_patterns(category)
            pattern_matches = {
                name: bool(re.search(pattern, query))
                for name, pattern in patterns.items()
            }

            # 根據匹配情況調整參數
            boost_factor = 0.15
            min_score_threshold = 0.8

            if category == 'insurance':
                if pattern_matches.get('time_notice'):
                    boost_factor = 0.3
                    min_score_threshold = 0.7
                elif pattern_matches.get('policy_change') and pattern_matches.get('benefit_loss'):
                    boost_factor = 0.25
                    min_score_threshold = 0.75

            elif category == 'finance':
                if pattern_matches.get('time_specific'):
                    if pattern_matches.get('litigation') or pattern_matches.get('department'):
                        boost_factor = 0.25
                        min_score_threshold = 0.6

            source_ids = [int(s) for s in source]

            # 獲取查詢參數
            params = self._analyze_query_params(query, category)
            exact_match_boost = params['exact_match_boost']
            min_score_threshold = params['min_score_threshold']
            boost_factor = params['boost_factor']

            # 使用 QueryProcessor 處理查詢
            expanded_query, keywords, query_embedding = self.query_processor.process_query(
                query, source, category
            )

            # 檢索結果
            candidates = []
            for source_id in source_ids:
                try:
                    results = vector_store.similarity_search_with_score_by_vector(
                        query_embedding,
                        k=top_k,
                        filter={"source": source_id}
                    )

                    if not results:
                        continue

                    for doc, base_score in results:
                        doc_text = doc.page_content
                        adjusted_score = base_score

                        # 財務類別特殊處理
                        if category == 'finance':
                            # 1. 檢查是否為財報金額查詢
                            is_financial_report = bool(re.search(r'(合併)?資產負債表|損益表|現金流量表', query))
                            is_amount_query = bool(re.search(r'[金額|總額|為].*?多少', query))

                            if is_financial_report and is_amount_query:
                                # 2. 優先選擇包含完整報表的文檔
                                doc_scores = []
                                for doc_id in source_ids:
                                    try:
                                        results = vector_store.similarity_search_with_score_by_vector(
                                            query_embedding,
                                            k=1,
                                            filter={"source": doc_id}
                                        )
                                        if not results:
                                            continue

                                        doc, base_score = results[0]
                                        doc_text = doc.page_content

                                        # 3. 增加關鍵特徵檢查
                                        score_adjustments = 0.0

                                        # 檢查是否包含完整報表結構
                                        if re.search(r'合併.*?資產負債表.*?單位：新台幣', doc_text, re.DOTALL):
                                            score_adjustments -= 0.3

                                        # 檢查是否包含目標財務項目
                                        target_item = re.search(r'現金及約當現金.*?\$\s*[\d,]+', doc_text)
                                        if target_item:
                                            score_adjustments -= 0.3

                                        # 檢查報表時期（民國年）
                                        period_match = re.search(r'民國\s*(\d+)\s*年', doc_text)
                                        if period_match:
                                            score_adjustments -= 0.2

                                        # 4. 調整最終分數
                                        final_score = base_score + score_adjustments
                                        doc_scores.append((doc_id, final_score, doc_text))

                                    except Exception as e:
                                        logger.error(f"Error processing document {doc_id}: {e}")
                                        continue

                            # 5. 選擇最佳文檔
                            if doc_scores:
                                # 按調整後的分數排序
                                doc_scores.sort(key=lambda x: x[1])
                                return doc_scores[0][0]
                        # 保險類別特殊處理
                        elif category == 'insurance':
                            # 檢查期限匹配
                            if re.search(r'[多少|幾](?:天|日|年|個月)|期限', query):
                                time_patterns = re.findall(r'\d+\s*(?:天|日|年|個月)', doc_text)
                                if time_patterns:
                                    adjusted_score *= (1 - exact_match_boost)

                        # 只保留高於閾值的結果
                        if adjusted_score < min_score_threshold:
                            candidates.append({
                                'doc_id': source_id,
                                'content': doc_text,
                                'score': adjusted_score,
                                'metadata': doc.metadata
                            })

                except Exception as e:
                    logger.error(f"Search failed for document {source_id}: {e}")
                    continue

            if not candidates:
                return source_ids[0]

            # 選擇最佳候選文件
            candidates.sort(key=lambda x: x['score'])
            best_candidate = candidates[0]

            logger.info(f"Selected document {best_candidate['doc_id']}")
            logger.info(f"Score: {best_candidate['score']:.4f}")

            return best_candidate['doc_id']

        except Exception as e:
            logger.error(f"Error in vector retrieve: {e}")
            return source_ids[0] if source_ids else -1

    def _hybrid_retrieve(self, query: str, source: List[str], vector_store: FAISS, category: str,
                         query_embedding: List[float]) -> int:
        """改進的混合檢索方法"""
        try:
            weights = self._adjust_hybrid_weights(query, category)
            source_ids = [int(s) for s in source]

            # 向量檢索結果
            vector_scores = {}
            for doc_id in source_ids:
                try:
                    results = vector_store.similarity_search_with_score_by_vector(
                        query_embedding,
                        k=1,
                        filter={"source": doc_id}
                    )
                    if results:
                        vector_scores[doc_id] = 1 - results[0][1]
                except Exception as e:
                    logger.warning(f"Error in vector scoring for doc {doc_id}: {e}")
                    vector_scores[doc_id] = 0.0

            # BM25檢索結果
            bm25_id = self.bm25_retriever.retrieve(query, source, category)
            bm25_scores = {int(doc_id): 1.0 if doc_id == bm25_id else 0.0 for doc_id in source_ids}

            # 計算加權分數
            final_scores = {}
            for doc_id in source_ids:
                doc_id = int(doc_id)
                final_scores[doc_id] = (
                        weights['vector'] * vector_scores.get(doc_id, 0) +
                        weights['bm25'] * bm25_scores.get(doc_id, 0)
                )

            if not final_scores:
                return int(source[0])

            return max(final_scores.items(), key=lambda x: x[1])[0]

        except Exception as e:
            logger.error(f"Error in hybrid retrieve: {e}")
            return int(source[0]) if source else -1

    def retrieve(
            self,
            query: str,
            source: List[str],
            vector_store: Optional[FAISS] = None,
            retrieval_mode: str = "vector",
            category: str = None
    ) -> int:
        """優化後的檢索方法，整合向量檢索和BM25檢索"""
        logger.info("=" * 50)
        logger.info(f"Starting retrieval process with mode: {retrieval_mode}")
        logger.info(f"Query: {query}")
        logger.info(f"Source documents: {source}")
        logger.info(f"Category: {category}")

        try:
            if retrieval_mode == "vector":
                if vector_store is None:
                    raise ValueError("Vector store is required for vector retrieval mode")
                return self._vector_retrieve(query, source, vector_store)

            elif retrieval_mode == "bm25":
                if category is None:
                    raise ValueError("Category is required for BM25 retrieval")
                return self.bm25_retriever.retrieve(query, source, category)

            elif retrieval_mode == "hybrid":
                # 調用 process_query 來獲取 embedding
                expanded_query, keywords, query_embedding = self.query_processor.process_query(query, source, category)
                if vector_store is None or category is None:
                    raise ValueError("Both vector store and category are required for hybrid retrieval")
                return self._hybrid_retrieve(query, source, vector_store, category, query_embedding)  # 傳入已計算的 embedding

            else:
                logger.warning(f"Unknown retrieval mode: {retrieval_mode}, falling back to vector retrieval")
                return self._vector_retrieve(query, source, vector_store)

        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            return int(source[0]) if source else -1

    # def _hybrid_retrieve(self, query: str, source: List[str], vector_store: FAISS, category: str,
    #                      query_embedding: List[float]) -> int:
    #     """
    #     根據不同分類的權重配置混合vector和bm25的結果
    #
    #     Args:
    #         query: 使用者問題
    #         source: 候選文檔列表
    #         category: 分類 ('insurance', 'finance', 'faq')
    #         query_embedding: 已計算好的 query embedding
    #     """
    #     try:
    #         source_ids = [int(s) for s in source]
    #         # 使用傳入的 embedding 進行向量檢索
    #         vector_scores = {}
    #         for doc_id in source_ids:
    #             try:
    #                 results = vector_store.similarity_search_with_score_by_vector(
    #                     query_embedding,  # 使用已計算的 embedding
    #                     k=1,
    #                     filter={"source": doc_id}
    #                 )
    #                 if results:
    #                     vector_scores[doc_id] = 1 - results[0][1]  # 轉換距離為相似度分數
    #             except Exception as e:
    #                 logger.warning(f"Error in vector scoring for doc {doc_id}: {e}")
    #                 vector_scores[doc_id] = 0.0
    #
    #         # 獲取BM25檢索結果
    #         bm25_id = self.bm25_retriever.retrieve(query, source, category)
    #         bm25_scores = {int(doc_id): 1.0 if doc_id == bm25_id else 0.0 for doc_id in source_ids}
    #
    #         # 根據分類獲取對應的權重
    #         weights = {
    #             'insurance': {'vector': 0.9, 'bm25': 0.1},
    #             'finance': {'vector': 0.8, 'bm25': 0.2},
    #             'faq': {'vector': 1.0, 'bm25': 0.0}
    #         }
    #
    #         category_weights = weights.get(category, {'vector': 0.5, 'bm25': 0.5})
    #
    #         # 計算最終分數
    #         final_scores = {}
    #         for doc_id in source_ids:
    #             doc_id = int(doc_id)
    #             final_scores[doc_id] = (
    #                     category_weights['vector'] * vector_scores.get(doc_id, 0) +
    #                     category_weights['bm25'] * bm25_scores.get(doc_id, 0)
    #             )
    #
    #         # 返回得分最高的文檔
    #         if not final_scores:
    #             return int(source[0])
    #
    #         return max(final_scores.items(), key=lambda x: x[1])[0]
    #
    #     except Exception as e:
    #         logger.error(f"Error in hybrid retrieve: {e}")
    #         return int(source[0]) if source else -1

    def _normalize_scores(self, scores):
        """標準化分數到[0,1]範圍"""
        if not scores:
            return {}
        max_score = max(scores.values())
        if max_score == 0:
            return scores
        return {k: v / max_score for k, v in scores.items()}

    def _get_category_from_source(self, source: List[str], question_category: str = None) -> str:
        """
        根據問題的類別返回正確的類別
        """
        return question_category  # 直接使用問題中的類別

    def _log_vector_store_status(self, vector_store: FAISS) -> None:
        """記錄向量存儲的狀態"""
        try:
            total_vectors = vector_store.index.ntotal if hasattr(vector_store, 'index') else 'Unknown'
            docstore_size = len(vector_store.docstore._dict) if hasattr(vector_store, 'docstore') else 'Unknown'

            logger.info("\nVector store status:")
            logger.info(f"Total vectors: {total_vectors}")
            logger.info(f"Docstore size: {docstore_size}")
        except Exception as e:
            logger.error(f"Error checking vector store status: {e}")

    def _parse_llm_output(self, output: str) -> Dict[str, Any]:
        """解析 LLM 輸出結果"""
        try:
            # 清理輸出文本
            output = output.strip()
            logger.debug(f"Raw LLM output: {output}")

            # 解析結果
            result = {}

            # 尋找 ID 和原因
            id_match = re.search(r'selected_id:?\s*(\d+)', output, re.IGNORECASE)
            reason_match = re.search(r'reason:?\s*(.+?)(?=selected_id|$)', output, re.IGNORECASE | re.DOTALL)

            if id_match:
                result['selected_id'] = int(id_match.group(1))
                logger.debug(f"Found ID: {result['selected_id']}")

            if reason_match:
                result['reason'] = reason_match.group(1).strip()
                logger.debug(f"Found reason: {result['reason']}")

            # 備用解析：尋找任何數字作為 ID
            if 'selected_id' not in result:
                numbers = re.findall(r'\d+', output)
                if numbers:
                    result['selected_id'] = int(numbers[0])
                    logger.debug(f"Using backup method, found ID: {result['selected_id']}")

            # 驗證結果
            if 'selected_id' not in result:
                raise ValueError("No valid document ID found in output")

            return result

        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}")
            logger.error(f"Original output: {output}")
            return {"selected_id": -1, "reason": "parsing error"}


class ResultAnalyzer:
    def __init__(self, ground_truth_path: str, output_dir: str = "analysis_results", ckip_processor=None):
        self.ground_truth_path = Path(ground_truth_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 設置 CKIP 處理器
        self.ckip_processor = ckip_processor
        if self.ckip_processor is None:
            logger.warning("CKIP processor not provided, analysis may be less accurate")

        # 設置日誌
        self.setup_logger()

        # 載入正確答案
        self.ground_truths = self.load_ground_truths()

        # 分析結果暫存
        self.segmentation_cache = {}
        self.error_patterns = {}
        self.analysis = None

    def _get_segmented_text(self, text: str) -> List[str]:
        """使用 CKIP 進行分詞"""
        if not text.strip():
            return []

        cache_key = hash(text)
        if cache_key in self.segmentation_cache:
            return self.segmentation_cache[cache_key]

        try:
            if self.ckip_processor:
                segmented = self.ckip_processor.segment_parallel([text])[0].split()
                self.segmentation_cache[cache_key] = segmented
                return segmented
            else:
                # 如果沒有 CKIP 處理器，使用簡單的字符分割
                return list(text.strip())
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            return list(text.strip())

    def _analyze_term_overlap(self, query: str, doc: str) -> Dict[str, Any]:
        """分析詞彙重疊情況"""
        query_terms = set(self._get_segmented_text(query))
        doc_terms = set(self._get_segmented_text(doc))

        overlap_terms = query_terms & doc_terms
        missing_terms = query_terms - doc_terms

        overlap_ratio = len(overlap_terms) / len(query_terms) if query_terms else 0

        return {
            'overlap_terms': list(overlap_terms),
            'missing_terms': list(missing_terms),
            'overlap_ratio': overlap_ratio,
            'segmented_query': list(query_terms),
            'segmented_doc': list(doc_terms)
        }

    def analyze_retrieval_error(self, case: Dict, documents: Dict[str, Dict[str, str]],
                                vector_store: FAISS) -> Dict[str, Any]:
        """分析檢索錯誤的具體原因"""
        try:
            query = case['query']
            predicted_doc = documents.get(str(case['predicted']), '')
            correct_doc = documents.get(str(case['correct']), '')

            # 分析詞彙重疊
            pred_overlap = self._analyze_term_overlap(query, predicted_doc)
            correct_overlap = self._analyze_term_overlap(query, correct_doc)

            # 計算重要指標
            pred_ratio = pred_overlap['overlap_ratio']
            correct_ratio = correct_overlap['overlap_ratio']
            ratio_diff = abs(pred_ratio - correct_ratio)

            error_analysis = {
                'error_type': '',
                'root_cause': '',
                'term_analysis': {
                    'predicted': {
                        'overlap_ratio': f"{pred_ratio:.2%}",
                        'overlap_terms': pred_overlap['overlap_terms'],
                        'missing_terms': pred_overlap['missing_terms']
                    },
                    'correct': {
                        'overlap_ratio': f"{correct_ratio:.2%}",
                        'overlap_terms': correct_overlap['overlap_terms'],
                        'missing_terms': correct_overlap['missing_terms']
                    }
                },
                'recommendations': []
            }

            # 判斷錯誤類型
            if ratio_diff > 0.5:
                error_analysis['error_type'] = '關鍵詞匹配差異過大'
                error_analysis['root_cause'] = (
                    f"預測文檔詞彙重疊率({pred_ratio:.2%})與"
                    f"正確文檔重疊率({correct_ratio:.2%})差異顯著"
                )
                error_analysis['recommendations'].extend([
                    "檢查文檔分詞結果",
                    "優化向量表示方法",
                    "考慮增加同義詞擴展"
                ])
            elif len(pred_overlap['missing_terms']) > len(correct_overlap['missing_terms']):
                error_analysis['error_type'] = '關鍵詞缺失'
                error_analysis['root_cause'] = (
                    f"預測文檔缺失重要關鍵詞：{', '.join(pred_overlap['missing_terms'])}"
                )
                error_analysis['recommendations'].extend([
                    "改進檢索策略以確保關鍵詞覆蓋",
                    "調整文本分塊大小",
                    "考慮使用關鍵詞加權"
                ])
            else:
                error_analysis['error_type'] = '語意理解偏差'
                error_analysis['root_cause'] = "詞彙重疊相近但語意理解可能存在偏差"
                error_analysis['recommendations'].extend([
                    "優化向量表示以更好捕捉語意",
                    "考慮使用更高級的語意理解模型",
                    "增加領域特定的訓練資料"
                ])

            return error_analysis

        except Exception as e:
            logger.error(f"Error in retrieval error analysis: {e}")
            return {
                'error_type': 'analysis_failed',
                'root_cause': str(e),
                'recommendations': ['檢查錯誤日誌並修復分析流程']
            }

    def generate_error_report(self, questions: List[Dict[str, Any]],
                              documents: Dict[str, Dict[str, str]],
                              vector_store: FAISS):
        """生成詳細的錯誤分析報告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"error_analysis_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 檢索錯誤深度分析報告\n\n")

            # 整體統計
            f.write("## 1. 整體統計\n")
            total_cases = len(self.analysis['incorrect_cases'])
            f.write(f"- 總問題數: {self.analysis['total']}\n")
            f.write(f"- 錯誤檢索數: {total_cases}\n")
            f.write(f"- 準確率: {self.analysis['accuracy']:.2%}\n\n")

            # 按類型分析錯誤
            f.write("## 2. 錯誤類型分析\n\n")
            error_types = {}
            detail_cases = []

            for case in self.analysis['incorrect_cases']:
                error_analysis = self.analyze_retrieval_error(case, documents, vector_store)
                error_type = error_analysis['error_type']
                error_types[error_type] = error_types.get(error_type, 0) + 1

                # 收集詳細案例信息
                detail_case = {
                    'qid': case['qid'],
                    'query': case['query'],
                    'analysis': error_analysis
                }
                detail_cases.append(detail_case)

            # 輸出錯誤類型統計
            for error_type, count in error_types.items():
                percentage = count / total_cases * 100
                f.write(f"### {error_type}\n")
                f.write(f"- 出現次數: {count} ({percentage:.1f}%)\n")
                f.write("- 典型案例:\n")

                # 找到該類型的典型案例
                typical_case = next(
                    (case for case in detail_cases
                     if case['analysis']['error_type'] == error_type),
                    None
                )

                if typical_case:
                    f.write(f"  - 查詢: {typical_case['query']}\n")
                    f.write(f"  - 原因: {typical_case['analysis']['root_cause']}\n")
                    f.write("  - 詞彙分析:\n")
                    term_analysis = typical_case['analysis']['term_analysis']
                    f.write(f"    - 預測文檔重疊率: {term_analysis['predicted']['overlap_ratio']}\n")
                    f.write(f"    - 正確文檔重疊率: {term_analysis['correct']['overlap_ratio']}\n")
                f.write("\n")

            # 詳細案例分析
            f.write("\n## 3. 詳細案例分析\n\n")
            for case in detail_cases:
                f.write(f"### 案例 {case['qid']}\n")
                f.write(f"- 查詢: {case['query']}\n")
                f.write(f"- 錯誤類型: {case['analysis']['error_type']}\n")
                f.write(f"- 根本原因: {case['analysis']['root_cause']}\n")

                term_analysis = case['analysis']['term_analysis']
                f.write("- 詞彙分析:\n")
                f.write("  - 預測文檔:\n")
                f.write(f"    - 重疊率: {term_analysis['predicted']['overlap_ratio']}\n")
                f.write(f"    - 重疊詞: {', '.join(term_analysis['predicted']['overlap_terms'])}\n")
                f.write(f"    - 缺失詞: {', '.join(term_analysis['predicted']['missing_terms'])}\n")
                f.write("  - 正確文檔:\n")
                f.write(f"    - 重疊率: {term_analysis['correct']['overlap_ratio']}\n")
                f.write(f"    - 重疊詞: {', '.join(term_analysis['correct']['overlap_terms'])}\n")
                f.write(f"    - 缺失詞: {', '.join(term_analysis['correct']['missing_terms'])}\n")

                f.write("- 改進建議:\n")
                for rec in case['analysis']['recommendations']:
                    f.write(f"  - {rec}\n")
                f.write("\n---\n\n")

            # 系統性建議
            f.write("\n## 4. 系統性改進建議\n\n")
            self._write_system_recommendations(f, error_types, total_cases)

    def _write_system_recommendations(self, f, error_types: Dict[str, int], total_cases: int):
        """寫入系統層面的改進建議"""
        # 根據錯誤類型分布給出建議
        major_issues = [
            (type_, count)
            for type_, count in error_types.items()
            if count / total_cases > 0.2
        ]

        if major_issues:
            f.write("### 主要問題\n")
            for issue_type, count in major_issues:
                percentage = count / total_cases * 100
                f.write(f"- {issue_type} ({percentage:.1f}%)\n")

            f.write("\n### 優先改進方向\n")
            if any('關鍵詞' in type_ for type_, _ in major_issues):
                f.write("1. 分詞和關鍵詞提取\n")
                f.write("   - 優化 CKIP 分詞參數\n")
                f.write("   - 建立領域特定的詞典\n")
                f.write("   - 實現關鍵詞權重計算\n\n")

            if any('語意' in type_ for type_, _ in major_issues):
                f.write("2. 語意理解\n")
                f.write("   - 使用更適合金融領域的預訓練模型\n")
                f.write("   - 增加領域特定的訓練資料\n")
                f.write("   - 優化向量表示方法\n\n")

            f.write("3. 系統配置優化\n")
            f.write("   - 調整文本分塊大小和重疊度\n")
            f.write("   - 優化向量檢索參數\n")
            f.write("   - 實現檢索結果重排序\n\n")

    def setup_logger(self):
        """設置日誌系統"""
        log_file = self.output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        self.logger = logging.getLogger("ResultAnalyzer")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 檔案處理器
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # 控制台處理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def load_ground_truths(self) -> Dict[int, Dict[str, Any]]:
        """載入正確答案"""
        try:
            with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 將正確答案轉換為以 qid 為鍵的字典
            ground_truths = {}
            for item in data.get('ground_truths', []):
                if 'qid' in item:
                    ground_truths[item['qid']] = item

            logger.info(f"Loaded {len(ground_truths)} ground truth items")
            return ground_truths

        except Exception as e:
            logger.error(f"Error loading ground truths: {e}")
            return {}

    def analyze_results(self, predictions: List[Dict], questions: List[Dict]) -> Dict[str, Any]:
        """分析檢索結果

        Args:
            predictions: 預測結果列表
            questions: 問題列表

        Returns:
            Dict[str, Any]: 分析結果
        """
        try:
            analysis = {
                'total': len(predictions),
                'correct': 0,
                'incorrect': 0,
                'errors_by_category': {'insurance': 0, 'finance': 0, 'faq': 0},
                'error_patterns': {},
                'incorrect_cases': [],
                'category_accuracy': {},
            }

            # 初始化類別統計
            category_stats = {
                'insurance': {'total': 0, 'correct': 0},
                'finance': {'total': 0, 'correct': 0},
                'faq': {'total': 0, 'correct': 0}
            }

            # 獲取問題查詢映射
            query_map = {q['qid']: q for q in questions}

            for pred in predictions:
                qid = pred['qid']
                retrieved_id = pred['retrieve']

                if qid not in self.ground_truths:
                    logger.warning(f"Question ID {qid} not found in ground truths")
                    continue

                ground_truth = self.ground_truths[qid]
                category = ground_truth['category']
                correct_id = ground_truth['retrieve']

                # 更新類別統計
                category_stats[category]['total'] += 1

                if retrieved_id == correct_id:
                    analysis['correct'] += 1
                    category_stats[category]['correct'] += 1
                else:
                    analysis['incorrect'] += 1
                    analysis['errors_by_category'][category] += 1

                    # 收集錯誤案例的詳細信息
                    question_info = query_map.get(qid, {})
                    error_case = {
                        'qid': qid,
                        'category': category,
                        'query': question_info.get('query', ''),
                        'predicted': retrieved_id,
                        'correct': correct_id,
                        'source_list': question_info.get('source', [])
                    }
                    analysis['incorrect_cases'].append(error_case)

            # 計算各類別準確率
            for category in category_stats:
                total = category_stats[category]['total']
                correct = category_stats[category]['correct']
                if total > 0:
                    analysis['category_accuracy'][category] = {
                        'accuracy': correct / total,
                        'correct': correct,
                        'total': total
                    }

            # 計算整體準確率
            analysis['accuracy'] = analysis['correct'] / analysis['total'] if analysis['total'] > 0 else 0

            # 保存分析結果供後續使用
            self.analysis = analysis
            return analysis

        except Exception as e:
            logger.error(f"Error in result analysis: {e}")
            return {
                'total': 0,
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0,
                'errors_by_category': {},
                'error_patterns': {},
                'incorrect_cases': [],
                'category_accuracy': {}
            }


def setup_argparser():
    """
    設置命令列參數解析器。

    Question Example:
    ```
    {
        "questions": [
            {
                "qid": 1,
                "source": [
                    442,
                    115,
                    440,
                    196,
                    431,
                    392,
                    14,
                    51
                ],
                "query": "匯款銀行及中間行所收取之相關費用由誰負擔?",
                "category": "insurance"
            },
            {
                "qid": 2,
                "source": [
                    475,
                    325,
                    578,
                    428,
                    606,
                    258,
                    275,
                    565
                ],
                "query": "本公司應在效力停止日前多少天以書面通知要保人？",
                "category": "insurance"
            }
        ]
    }
    ```
    當 category 為 insurance or finance，source 和 answer 在對應 reference 中 數字對應的檔案，
    如果是 faq，則在 faq/pid_map_content.json 的 key 中

    Returns:
        Tuple[argparse.Namespace, RAGConfig]: 解析後的參數和RAG配置對象
    """
    parser = argparse.ArgumentParser(
        description='Financial QA RAG System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必要參數（沒有默認值）
    parser.add_argument(
        '--question_path',
        type=str,
        required=True,
        help='Path to questions JSON file'
    )
    parser.add_argument(
        '--source_path',
        type=str,
        required=True,
        help='Path to source documents directory'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to output JSON file'
    )

    # 系統配置參數（有默認值）
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='ckip_cache',
        help='Directory for CKIP cache'
    )
    parser.add_argument(
        '--vector_store_path',
        type=str,
        default='vector_stores',
        help='Directory for vector stores'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=None,
        help='Number of processes for parallel processing. Defaults to (CPU count - 1)'
    )

    # 模型相關參數
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU device ID (-1 for CPU)'
    )
    parser.add_argument(
        '--ckip_model',
        type=str,
        default='bert-base',
        choices=['bert-base', 'bert-tiny', 'albert-base', 'albert-tiny'],
        help='CKIP model to use'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt-4o',
        help='OpenAI model to use'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for OpenAI model'
    )

    # 文本處理參數
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=2000,
        help='Size of text chunks for processing'
    )
    parser.add_argument(
        '--chunk_overlap',
        type=int,
        default=1000,
        help='Overlap size between text chunks'
    )

    # 添加新的參數
    parser.add_argument(
        '--retrieval_mode',
        type=str,
        default='vector',
        choices=['vector', 'bm25', 'hybrid'],
        help='Retrieval mode to use (vector, bm25, or hybrid)'
    )

    parser.add_argument(
        '--use_delim',
        type=bool,
        default=True,
        help='Whether to use delimiters in CKIP processing'
    )
    parser.add_argument(
        '--max_retries',
        type=int,
        default=3,
        help='Maximum number of retries for failed operations'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout in seconds for operations'
    )

    parser.add_argument(
        '--rebuild-stores',
        action='store_true',
        help='Force rebuild of vector stores'
    )

    # 添加jieba選項
    parser.add_argument(
        '--use-jieba',
        action='store_true',
        help='Use Jieba instead of CKIP for BM25 tokenization'
    )

    args = parser.parse_args()

    # 創建並返回 RAGConfig 實例
    config = RAGConfig(
        cache_dir=args.cache_dir,
        vector_store_path=args.vector_store_path,
        batch_size=args.batch_size,
        num_processes=args.num_processes,
        device=args.device,
        ckip_model=args.ckip_model,
        model_name=args.model_name,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_delim=args.use_delim,
        max_retries=args.max_retries,
        timeout=args.timeout,
        rebuild_stores=args.rebuild_stores,
        use_jieba=args.use_jieba

    )

    return args, config


def process_questions(
        rag_system: RAGSystem,
        questions: List[Dict],
        vector_stores: Dict[str, FAISS],
        retrieval_mode: str
) -> List[Dict]:
    """修正的問題處理函數"""
    answers = []

    for question in tqdm(questions, desc="Processing questions"):
        try:
            category = question.get('category')  # 使用 get() 避免 KeyError
            if not category:
                logger.warning(f"No category found for question {question.get('qid')}")
                continue
            # processor = CKIPProcessor(
            #     cache_dir=rag_system.config.cache_dir,
            #     batch_size=32,
            #     model_name=rag_system.config.ckip_model,
            #     device=rag_system.config.device,
            #     use_delim=rag_system.config.use_delim
            # )
            # query = ' '.join(processor.ws_driver([question['query']])[0])

            retrieved = rag_system.retrieve(
                query=question['query'],
                source=question['source'],
                vector_store=vector_stores.get(category) if retrieval_mode != 'bm25' else None,
                retrieval_mode=retrieval_mode,
                category=category  # 確保傳遞 category
            )

            answers.append({
                "qid": question['qid'],
                "retrieve": retrieved
            })

        except Exception as e:
            logger.error(f"Error processing question {question.get('qid', 'unknown')}: {str(e)}")
            answers.append({
                "qid": question['qid'],
                "retrieve": int(question['source'][0]) if question.get('source') else -1
            })

    return answers


def main():
    """主程序"""
    try:
        # 解析命令行參數
        args, config = setup_argparser()

        # 設置分析結果目錄
        analysis_dir = Path("analysis_results")
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # 初始化RAG系統
        logger.info("Initializing RAG system...")
        rag_system = RAGSystem(config)
        vector_store_manager = rag_system.vector_store_manager
        vector_stores = {}

        # 載入正確答案用於分析，並初始化分析器
        ground_truth_path = Path("dataset/preliminary/ground_truths_example.json")
        analyzer = ResultAnalyzer(
            ground_truth_path=str(ground_truth_path),
            output_dir=str(analysis_dir),
            ckip_processor=rag_system.query_processor.ckip_processor
        )
        logger.info(f"Loaded ground truth data from {ground_truth_path}")

        # 確保快取目錄結構
        cache_structure = {
            'text_cache': Path(config.cache_dir) / 'text_cache',
            'vector_cache': Path(config.cache_dir) / 'vector_cache',
            'reports': Path(config.cache_dir) / 'reports',
            'manifests': Path(config.cache_dir) / 'manifests'
        }

        for dir_path in cache_structure.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # 根據檢索模式決定是否需要向量存儲
        documents = None
        if args.retrieval_mode in ['vector', 'hybrid']:
            # 如果指定重建，則清理所有向量存儲
            if config.rebuild_stores:
                logger.info("Rebuild flag set, clearing all vector stores...")
                vector_store_manager.clear_store()

            # 處理各類別文檔
            for category in ['finance', 'insurance', 'faq']:
                logger.info(f"Processing category: {category}")
                store = None
                rebuild_required = config.rebuild_stores

                if not rebuild_required:
                    # 嘗試載入現有存儲
                    store = vector_store_manager.load_vector_store(category)
                    rebuild_required = store is None

                if rebuild_required:
                    if documents is None:
                        logger.info("Processing documents...")
                        documents = rag_system.process_documents(args.source_path)
                        if not documents:
                            raise ValueError("No documents processed successfully")

                    if category in documents:
                        logger.info(f"Creating new vector store for {category}")
                        texts = [doc for doc in documents[category].values()]
                        file_ids = [int(id_) for id_ in documents[category].keys()]
                        store = vector_store_manager.create_vector_store(
                            category=category,
                            texts=texts,
                            file_ids=file_ids
                        )

                        if not store:
                            raise ValueError(f"Failed to create vector store for {category}")

                vector_stores[category] = store
                logger.info(f"Successfully processed vector store for {category}")

        # 初始化 BM25 檢索器（如果需要）
        if args.retrieval_mode in ['bm25', 'hybrid']:
            logger.info("Initializing BM25 retriever...")
            if documents is None:
                documents = rag_system.process_documents(args.source_path)
            rag_system.create_retriever_stores(documents, args.retrieval_mode)

        # 載入問題
        logger.info(f"Loading questions from {args.question_path}")
        with open(args.question_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
            questions = questions_data['questions']

        # 處理問題
        logger.info(f"Processing questions using {args.retrieval_mode} mode...")
        answers = process_questions(
            rag_system,
            questions,
            vector_stores,
            args.retrieval_mode
        )

        # 保存結果
        logger.info(f"Saving results to {args.output_path}")
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump({"answers": answers}, f, ensure_ascii=False, indent=2)

        logger.info("\nProcessing completed successfully")
        logger.info(f"Results saved to {args.output_path}")
        logger.info(f"Analysis results saved to {analysis_dir}")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Critical error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
