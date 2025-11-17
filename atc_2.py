import json
import traceback
import gc
import sys
import datetime
import logging
import os
import re
import string
import asyncio
import shutil
import uuid
import signal
import time
from collections import Counter
from typing import List, Tuple, Any, Optional

# PyTorch必须在其他库之前导入，以避免CUDA初始化问题
import torch

# LightRAG 相关导入
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.hf import hf_model_complete, hf_embed
from transformers import AutoModel, AutoTokenizer
from lightrag.kg.shared_storage import initialize_pipeline_status, get_namespace_data

# 导入本地LLM模型
from localllm import LlamaQAModel

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ========== 超参数配置 ==========
# Embedding 模型配置
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
EMBEDDING_DIM = 1024

# LightRAG 查询模式
DEFAULT_QUERY_MODE = "hybrid"  # 可选: "naive", "local", "global", "hybrid"

# 磁盘清理配置
MAX_WORKING_DIRS = 5  # 保留的最大工作目录数量
DISK_CLEANUP_INTERVAL = 3  # 每处理N个样本后执行一次磁盘清理

# 内存管理配置
MEMORY_CLEANUP_INTERVAL = 10  # 每处理N个样本后执行一次完整内存清理
FORCE_CPU_FOR_EMBEDDING = False  # 如果CUDA问题持续，可以设置为True

# 评估配置
TEST_FILE_PATH = "/root/raptor/test.hard.json"  # 测试文件路径
SHOW_FULL_CONTEXT_SAMPLES = 10  # 展示完整上下文的样本数量
# ==============================

# 全局变量
_lightrag_instance = None
_lightrag_tasks = set()  # 跟踪所有异步任务
BASE_WORKING_DIR = "./lightrag_test_hard_temp"  # 基础工作目录
LOG_DIR = "./logs"  # 定义日志目录
LAST_LOG_PATH = None  # 保存最新的日志路径
INTERRUPTED = False  # 全局中断标志
SHUTDOWN_EVENT = asyncio.Event()  # 用于优雅关闭的事件
_current_working_dir = None  # 当前使用的工作目录

# 确保目录存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BASE_WORKING_DIR, exist_ok=True)

# 全局评估状态
evaluation_state = {
    "em_total": 0.0,
    "f1_total": 0.0,
    "corrected_em_total": 0.0,
    "corrected_f1_total": 0.0,
    "total": 0,
    "last_processed_line": 0
}

# 初始化 LlamaQAModel - 添加错误恢复机制
try:
    llama_model = LlamaQAModel(model_name="meta-llama/Llama-3.1-8B-Instruct")
    logging.info("LlamaQAModel initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize LlamaQAModel: {str(e)}")
    # 在严重错误时退出
    sys.exit(1)

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text, flags=re.IGNORECASE)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split()

def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return f1

def check_containment_relationship(gold: str, pred: str) -> bool:
    """检查是否满足包含关系：gold完全包含pred或pred完全包含gold"""
    # 特殊处理：如果gold是"N/A"，那么pred也必须是"N/A"才算正确
    if gold == "N/A":
        return pred == "N/A"
    
    # 如果pred是"N/A"，但gold不是"N/A"，不算正确
    if pred == "N/A" and gold != "N/A":
        return False
    
    norm_gold = normalize_answer(gold)
    norm_pred = normalize_answer(pred)
    
    # 避免空字符串的情况
    if not norm_gold or not norm_pred:
        return False
        
    return norm_gold in norm_pred or norm_pred in norm_gold

def clear_directory_sync(directory: str):
    """同步清理目录，用于启动和关闭时"""
    try:
        if os.path.exists(directory):
            # 尝试安全删除
            shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)
        logging.debug(f"Sync cleared directory: {directory}")
    except Exception as e:
        logging.error(f"Error during synchronous directory cleanup {directory}: {str(e)}")

async def cleanup_torch_resources():
    """清理PyTorch相关资源，特别是CUDA缓存 - 增强版本"""
    try:
        # 1. 强制Python垃圾回收
        gc.collect()
        
        # 2. 如果CUDA可用，清理CUDA资源
        if torch.cuda.is_available():
            try:
                # 先尝试正常的缓存清理
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                # 重置内存统计
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                
                # 记录当前内存使用
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                logging.debug(f"PyTorch cleanup - Allocated: {allocated/1024**3:.2f}GB, Reserved: {reserved/1024**3:.2f}GB")
                
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logging.warning(f"CUDA error during cleanup, attempting recovery: {str(e)}")
                    # 对于CUDA错误，尝试更激进的清理
                    try:
                        # 强制垃圾回收
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as recovery_e:
                        logging.error(f"Failed to recover from CUDA error: {str(recovery_e)}")
                else:
                    raise e
                    
    except Exception as e:
        logging.warning(f"Error during PyTorch resource cleanup: {str(e)}")

async def cleanup_working_directory(directory: str):
    """异步清理单个工作目录"""
    if not os.path.exists(directory):
        return
    
    try:
        # 先尝试优雅删除
        await asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)
        logging.debug(f"Cleaned working directory: {directory}")
    except Exception as e:
        logging.warning(f"Error cleaning directory {directory}: {str(e)}")
        # 作为备选方案，尝试逐个文件删除
        try:
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except Exception as f:
                        logging.debug(f"Error removing file {name}: {f}")
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception as f:
                        logging.debug(f"Error removing directory {name}: {f}")
            os.rmdir(directory)
        except Exception as e:
            logging.error(f"Final error removing directory {directory}: {e}")

async def cleanup_lightrag_resources(full_cleanup: bool = False):
    """彻底清理LightRAG相关资源"""
    global _lightrag_instance, _lightrag_tasks, _current_working_dir
    
    # 1. 清理LightRAG实例
    if _lightrag_instance is not None:
        try:
            await asyncio.sleep(0.1)  # 给异步操作一些时间完成
            await _lightrag_instance.finalize_storages()
        except Exception as e:
            logging.warning(f"Error finalizing LightRAG storages during cleanup: {e}")
        finally:
            # 显式删除实例引用
            del _lightrag_instance
            _lightrag_instance = None
    
    # 2. 清理当前工作目录
    if _current_working_dir is not None:
        await cleanup_working_directory(_current_working_dir)
        _current_working_dir = None
    
    # 3. 取消所有pending的异步任务
    if _lightrag_tasks:
        logging.debug(f"Cleaning up {len(_lightrag_tasks)} pending tasks")
        for task in _lightrag_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logging.debug(f"Error cancelling task: {e}")
        _lightrag_tasks.clear()
    
    # 4. 清理PyTorch资源
    await cleanup_torch_resources()
    
    # 5. 完整清理时额外清理
    if full_cleanup:
        # 清理旧的工作目录，只保留最新的几个
        await cleanup_old_working_dirs()

async def cleanup_old_working_dirs():
    """清理旧的工作目录，只保留最新的MAX_WORKING_DIRS个"""
    try:
        if not os.path.exists(BASE_WORKING_DIR):
            return
        
        # 获取所有子目录
        dirs = [d for d in os.listdir(BASE_WORKING_DIR) 
                if os.path.isdir(os.path.join(BASE_WORKING_DIR, d))]
        
        # 按修改时间排序（最新的在后）
        dirs.sort(key=lambda x: os.path.getmtime(os.path.join(BASE_WORKING_DIR, x)), reverse=True)
        
        # 保留最新的MAX_WORKING_DIRS个目录
        to_remove = dirs[MAX_WORKING_DIRS:] if len(dirs) > MAX_WORKING_DIRS else []
        
        for dir_name in to_remove:
            dir_path = os.path.join(BASE_WORKING_DIR, dir_name)
            await cleanup_working_directory(dir_path)
            logging.info(f"Removed old working directory: {dir_path}")
            
    except Exception as e:
        logging.error(f"Error during old working directories cleanup: {str(e)}")

async def lightrag_init() -> LightRAG:
    """Initialize LightRAG model with complete isolation and error recovery."""
    global _lightrag_instance, _current_working_dir
    
    # 每次初始化前彻底清理资源
    await cleanup_lightrag_resources()
    
    # 生成唯一工作目录
    unique_id = f"run_{int(time.time())}_{str(uuid.uuid4())[:6]}"
    working_dir = os.path.join(BASE_WORKING_DIR, unique_id)
    _current_working_dir = working_dir  # 记录当前工作目录
    
    # 确保工作目录是干净的
    os.makedirs(working_dir, exist_ok=True)

    try:
        # 加载 embedding 模型和 tokenizer
        if FORCE_CPU_FOR_EMBEDDING:
            # 强制使用CPU来避免CUDA问题
            embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, device_map="cpu")
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        else:
            # 正常加载，让transformers自动选择设备
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
            embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=hf_model_complete,
            llm_model_name="meta-llama/Llama-3.1-8B-Instruct",
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=5000,
                func=lambda texts: hf_embed(
                    texts,
                    tokenizer=tokenizer,
                    embed_model=embed_model,
                ),
            ),
        )

        # 等待存储完全初始化
        await rag.initialize_storages()
        await initialize_pipeline_status()
        
        _lightrag_instance = rag
        return rag
        
    except Exception as e:
        # 如果初始化失败，清理资源并重新抛出异常
        await cleanup_lightrag_resources()
        logging.error(f"Failed to initialize LightRAG: {str(e)}")
        raise

async def extract_answer_from_response(s: str) -> str:
    """Extract the final answer from the assistant's response."""
    last_assistant_pos = s.rfind('assistant\n')
    return s[last_assistant_pos+10:].strip() if last_assistant_pos != -1 else ""

async def call_lightrag_with_retry(question: str, context: str, mode: str = DEFAULT_QUERY_MODE, max_retries: int = 2) -> Tuple[str, str]:
    """Get predicted answer from LightRAG model with retry mechanism for CUDA errors."""
    global _lightrag_tasks
    
    for attempt in range(max_retries + 1):
        # 检查是否需要中断
        if SHUTDOWN_EVENT.is_set():
            raise asyncio.CancelledError("Operation cancelled due to shutdown request")
        
        try:
            rag = await lightrag_init()
            
            # 生成唯一文档ID
            doc_id = f"doc_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            await rag.ainsert(context, ids=[doc_id])
            
            # 创建查询参数，设置 only_need_context=True
            param = QueryParam(mode=mode, only_need_context=True)
            
            # 直接获取上下文字符串
            context_str = await rag.aquery(question, param=param)
            
            # 使用 LlamaQAModel 生成答案
            loop = asyncio.get_event_loop()
            try:
                answer = await loop.run_in_executor(
                    None, 
                    lambda: llama_model.answer_question(context_str, question)
                )
                
                # 尝试从完整响应中提取答案
                final_answer = await extract_answer_from_response(answer)
                
                # 基本清理：移除引号、多余空格等
                final_answer = final_answer.strip().strip('"').strip("'")
                
                return final_answer, context_str
                
            except Exception as e:
                logging.error(f"Error generating answer with LlamaQAModel (attempt {attempt+1}/{max_retries+1}): {str(e)}")
                if attempt == max_retries:
                    # 最后一次尝试仍然失败
                    return f"Error: {str(e)}", context_str
                else:
                    # 重试前进行更彻底的清理
                    await cleanup_torch_resources()
                    await asyncio.sleep(1)  # 等待1秒再重试
                    
        except RuntimeError as e:
            if "CUDA" in str(e) and attempt < max_retries:
                logging.warning(f"CUDA error detected (attempt {attempt+1}/{max_retries+1}), performing deep cleanup and retrying: {str(e)}")
                # 对于CUDA错误，进行深度清理
                await cleanup_lightrag_resources(full_cleanup=True)
                await asyncio.sleep(2)  # 等待更长时间让CUDA恢复
            else:
                # 其他RuntimeError或达到最大重试次数
                logging.error(f"Runtime error in call_lightrag (attempt {attempt+1}/{max_retries+1}): {str(e)}")
                if attempt == max_retries:
                    return f"Runtime Error: {str(e)}", ""
                else:
                    await cleanup_lightrag_resources()
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logging.error(f"Unexpected error in call_lightrag (attempt {attempt+1}/{max_retries+1}): {str(e)}")
            if attempt == max_retries:
                return f"Error: {str(e)}", ""
            else:
                await cleanup_lightrag_resources()
                await asyncio.sleep(1)
    
    # 所有重试都失败
    return "Error: All retries failed", ""

# 保持向后兼容性
call_lightrag = call_lightrag_with_retry

async def evaluate_dataset_async(file_path: str) -> bool:
    """Evaluate dataset using EM and F1 scores with enhanced error handling."""
    global LAST_LOG_PATH, evaluation_state, INTERRUPTED
    
    # 重置评估状态
    evaluation_state = {
        "em_total": 0.0,
        "f1_total": 0.0,
        "corrected_em_total": 0.0,
        "corrected_f1_total": 0.0,
        "total": 0,
        "last_processed_line": 0
    }
    
    interrupted = False

    # 获取基础文件名（不含路径和扩展名）
    base_filename = os.path.basename(file_path)
    base_name = os.path.splitext(base_filename)[0]
    
    # 创建带时间戳的日志文件名
    iso_str = datetime.datetime.now().isoformat().replace(':', '-').replace('.', '_')
    log_filename = f"{base_name}_{iso_str}_lightrag_log.txt"
    full_log_path = os.path.join(LOG_DIR, log_filename)
    LAST_LOG_PATH = full_log_path  # 保存日志路径以便中断时访问

    # 设置日志
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logging.basicConfig(
        filename=full_log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w',
        force=True
    )
    logger.addHandler(logging.StreamHandler())  # 仍然在控制台输出

    # 设置不同模块的日志级别
    logging.getLogger('lightrag').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)  # 减少asyncio日志噪音
    
    # 明确记录日志文件位置
    logging.info(f"Log file will be saved to: {full_log_path}")
    print(f"Log file will be saved to: {full_log_path}")
    
    # 记录超参数配置
    logging.info(f"Embedding Model: {EMBEDDING_MODEL_NAME}, Dimension: {EMBEDDING_DIM}")
    logging.info(f"Query Mode: {DEFAULT_QUERY_MODE}")
    logging.info(f"Disk Cleanup: Keeping only {MAX_WORKING_DIRS} latest working directories")
    logging.info(f"Memory Cleanup Interval: Every {MEMORY_CLEANUP_INTERVAL} samples")
    logging.info(f"Force CPU for Embedding: {FORCE_CPU_FOR_EMBEDDING}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            logging.info(f"Opened file: {file_path}")
            logging.info("Using model: lightrag with custom LlamaQAModel")
            
            for line_num, line in enumerate(f, 1):
                # 检查是否收到中断信号
                if SHUTDOWN_EVENT.is_set():
                    logging.warning(f"Shutdown requested, stopping after line {line_num-1}")
                    interrupted = True
                    break
                
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing line {line_num}: {line[:50]}... | Error: {str(e)}")
                    continue
                
                idx = entry.get("idx", "N/A")
                question = entry.get("question", "")
                context = entry.get("context", "")
                targets = entry.get("targets", [])

                try:
                    # Get prediction from LightRAG with retry mechanism
                    prediction, retrieved_context = await call_lightrag_with_retry(
                        question, 
                        context, 
                        mode=DEFAULT_QUERY_MODE
                    )
                except asyncio.CancelledError:
                    logging.info(f"Task cancelled at line {line_num}, preparing to exit")
                    interrupted = True
                    break
                except Exception as e:
                    logging.error(f"Error processing line {line_num}: {str(e)}")
                    traceback.print_exc()
                    continue
                
                processed_targets = ["N/A" if t.strip() == "" else t for t in targets]

                # Calculate original scores
                em_scores = [compute_exact(t, prediction) for t in processed_targets]
                f1_scores = [compute_f1(t, prediction) for t in processed_targets]
                em_score = max(em_scores) if em_scores else 0.0
                f1_score = max(f1_scores) if f1_scores else 0.0

                # Calculate corrected scores
                corrected_em_score = em_score
                corrected_f1_score = f1_score
                
                # 如果原始EM不为1，检查是否满足包含关系
                if em_score < 1.0:
                    for target in processed_targets:
                        if check_containment_relationship(target, prediction):
                            corrected_em_score = 1.0
                            corrected_f1_score = 1.0
                            break

                # 更新全局状态
                evaluation_state["em_total"] += em_score
                evaluation_state["f1_total"] += f1_score
                evaluation_state["corrected_em_total"] += corrected_em_score
                evaluation_state["corrected_f1_total"] += corrected_f1_score
                evaluation_state["total"] += 1
                evaluation_state["last_processed_line"] = line_num

                # 决定是否显示完整上下文 - 使用超参数
                show_full_context = line_num <= SHOW_FULL_CONTEXT_SAMPLES
                
                # Log entry
                containment_status = " (CONTAINMENT CORRECTED)" if corrected_em_score > em_score else ""
                log_entry = (
                    f"\n===== ENTRY {line_num} ====={containment_status}"
                    f"\n{'✅' if em_score == 1 else '❌'}{'✅' if corrected_em_score == 1 and em_score != 1 else ''} "
                    f"[{'CORRECT' if em_score == 1 else 'INCORRECT'}{' -> CORRECT' if corrected_em_score == 1 and em_score != 1 else ''}] ID: {idx}"
                    f"\nQuestion: {question}"
                    f"\nPrediction: {prediction}"
                )
                
                # 添加完整上下文（仅前N个示例，使用超参数）
                if show_full_context:
                    log_entry += f"\n\n--- RETRIEVED CONTEXT (debug) ---\n{retrieved_context}\n--- END CONTEXT ---\n"
                
                log_entry += (
                    f"\nProcessed Targets: {processed_targets}"
                    f"\nOriginal EM: {em_score}, F1: {f1_score:.4f}"
                    f"\nCorrected EM: {corrected_em_score}, F1: {corrected_f1_score:.4f}"
                    f"\n=================\n"
                )
                logging.info(log_entry)
                print(log_entry)

                # 定期磁盘清理和内存清理
                if line_num % DISK_CLEANUP_INTERVAL == 0:
                    logging.info(f"Performing disk cleanup after {line_num} samples")
                    await cleanup_old_working_dirs()
                
                # 定期内存清理（比磁盘清理频率低）
                if line_num % MEMORY_CLEANUP_INTERVAL == 0:
                    logging.info(f"Performing memory cleanup after {line_num} samples")
                    await cleanup_torch_resources()

    except KeyboardInterrupt:
        logging.warning("Evaluation interrupted by user with Ctrl-C.")
        print("\n" + "="*60)
        print("[INTERRUPTED] Evaluation stopped by user (Ctrl-C)")
        print("="*60)
        interrupted = True
        INTERRUPTED = True

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        traceback.print_exc()
    finally:
        # 最终清理所有资源
        await cleanup_lightrag_resources(full_cleanup=True)
        
        # 计算分数
        total = evaluation_state["total"]
        if total > 0:
            em_avg = (evaluation_state["em_total"] / total) * 100
            f1_avg = (evaluation_state["f1_total"] / total) * 100
            corrected_em_avg = (evaluation_state["corrected_em_total"] / total) * 100
            corrected_f1_avg = (evaluation_state["corrected_f1_total"] / total) * 100
        else:
            em_avg = f1_avg = corrected_em_avg = corrected_f1_avg = 0.0

        # 确定结果前缀
        status_prefix = "[INTERRUPTED] Partial results: " if interrupted else "[COMPLETED] Final results: "
        
        # 创建结果报告
        final_log = (
            f"\n{'='*60}"
            f"\n{status_prefix}Total={total}"
            f"\nModel: LightRAG with custom LlamaQAModel"
            f"\nEmbedding Model: {EMBEDDING_MODEL_NAME}, Dimension: {EMBEDDING_DIM}"
            f"\nQuery Mode: {DEFAULT_QUERY_MODE}"
            f"\nOriginal Scores: EM={em_avg:.2f}%, F1={f1_avg:.2f}%"
            f"\nCorrected Scores: EM={corrected_em_avg:.2f}%, F1={corrected_f1_avg:.2f}%"
            f"\nImprovement: EM +{corrected_em_avg - em_avg:.2f}%, F1 +{corrected_f1_avg - f1_avg:.2f}%"
            f"\nLog file saved to: {full_log_path}"
            f"\n{'='*60}"
        )
        
        # 记录到日志
        logging.info(final_log)
        
        # 强制刷新所有日志
        for handler in logging.getLogger().handlers:
            handler.flush()
        
        # 同时打印到控制台
        print(final_log)
            
        return interrupted

def evaluate_dataset(file_path: str) -> bool:
    """同步包装函数，用于运行异步评估"""
    global LAST_LOG_PATH, SHUTDOWN_EVENT, INTERRUPTED
    
    try:
        # 重置事件
        SHUTDOWN_EVENT.clear()
        INTERRUPTED = False
        
        return asyncio.run(evaluate_dataset_async(file_path))
    except KeyboardInterrupt:
        logging.warning("[INTERRUPTED] Evaluation stopped by user (Ctrl-C)")
        return True
    except Exception as e:
        logging.error(f"Unexpected error in evaluate_dataset: {str(e)}")
        traceback.print_exc()
        return True

async def safe_shutdown(exit_code: int = 1):
    """安全关闭所有资源，确保日志完整写入"""
    global LAST_LOG_PATH, evaluation_state
    
    logging.info("Initiating safe shutdown sequence...")
    print("\n" + "="*60)
    print("[SHUTDOWN] Performing safe shutdown...")
    print("="*60)
    
    try:
        # 1. 设置关闭标志
        SHUTDOWN_EVENT.set()
        
        # 2. 计算当前结果
        total = evaluation_state["total"]
        if total > 0:
            em_avg = (evaluation_state["em_total"] / total) * 100
            f1_avg = (evaluation_state["f1_total"] / total) * 100
            corrected_em_avg = (evaluation_state["corrected_em_total"] / total) * 100
            corrected_f1_avg = (evaluation_state["corrected_f1_total"] / total) * 100
            
            shutdown_log = (
                f"\n{'='*60}"
                f"\n[SAFESHUTDOWN] Partial results after {total} samples:"
                f"\nOriginal Scores: EM={em_avg:.2f}%, F1={f1_avg:.2f}%"
                f"\nCorrected Scores: EM={corrected_em_avg:.2f}%, F1={corrected_f1_avg:.2f}%"
                f"\nLog file: {LAST_LOG_PATH}"
                f"\n{'='*60}"
            )
            logging.info(shutdown_log)
            print(shutdown_log)
        
        # 3. 强制刷新日志
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except:
                pass
        
        # 4. 彻底清理所有资源
        await cleanup_lightrag_resources(full_cleanup=True)
        
        # 5. 再次刷新日志
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except:
                pass
        
        logging.info("Safe shutdown completed successfully")
        
    except Exception as e:
        logging.error(f"Error during safe shutdown: {str(e)}")
        traceback.print_exc()
    finally:
        # 6. 退出程序
        sys.exit(exit_code)

def signal_handler(sig, frame):
    """处理信号，确保正常退出"""
    global LAST_LOG_PATH
    
    print("\n" + "="*60)
    print(f"[INTERRUPTED] Received signal {sig}, initiating safe shutdown...")
    print("="*60)
    
    # 禁用SIGINT处理，防止嵌套中断
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # 尝试打印最后的日志路径
    if LAST_LOG_PATH and os.path.exists(LAST_LOG_PATH):
        print(f"\n[INFO] Latest log file is available at: {LAST_LOG_PATH}")
    
    # 启动安全关闭流程
    try:
        asyncio.run(safe_shutdown(1))
    except Exception as e:
        print(f"Error during shutdown: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 程序启动时同步清理基础工作目录
    clear_directory_sync(BASE_WORKING_DIR)
    
    # 设置PyTorch内存限制，防止OOM
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)  # 降低到85%以留出更多余量
        logging.info(f"PyTorch memory limit set to 85% of GPU capacity")
    
    # 注册信号处理程序
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl-C
    signal.signal(signal.SIGTERM, signal_handler)  # kill 命令
    
    # 设置asyncio策略以处理Windows和Linux差异
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        # 使用超参数配置的测试文件路径
        test_file = TEST_FILE_PATH
        
        # 检查文件是否存在
        if not os.path.exists(test_file):
            print(f"错误: 测试文件 '{test_file}' 不存在!")
            print("请确保测试文件路径正确，或者修改脚本中的 TEST_FILE_PATH 超参数")
            sys.exit(1)
            
        print(f"开始 LightRAG 测试，使用文件: {test_file}")
        print(f"日志文件将保存在: ./{LOG_DIR}/ 目录下")
        print(f"Embedding Model: {EMBEDDING_MODEL_NAME}, Dimension: {EMBEDDING_DIM}")
        print(f"Query Mode: {DEFAULT_QUERY_MODE}")
        print(f"Disk Cleanup: Keeping only {MAX_WORKING_DIRS} latest working directories")
        print(f"Memory Cleanup: Every {MEMORY_CLEANUP_INTERVAL} samples")
        print(f"Force CPU for Embedding: {FORCE_CPU_FOR_EMBEDDING}")
        print("按 Ctrl-C 可随时中断测试并查看当前结果...")
        interrupted = evaluate_dataset(test_file)
        exit_code = 1 if interrupted else 0
    except KeyboardInterrupt:
        # 主函数级别的 KeyboardInterrupt 处理
        print("\n" + "="*60)
        print("[INTERRUPTED] Program terminated by user (Ctrl-C)")
        print("="*60)
        exit_code = 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        exit_code = 2
    finally:
        # 确保最后进行彻底清理
        try:
            asyncio.run(cleanup_lightrag_resources(full_cleanup=True))
            # 最终同步清理基础目录
            clear_directory_sync(BASE_WORKING_DIR)
        except Exception as e:
            logging.error(f"Final cleanup error: {str(e)}")
        
        sys.exit(exit_code)