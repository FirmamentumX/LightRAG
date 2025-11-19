#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特定行评估脚本 - 仅评估指定行号的测试用例
"""

# PyTorch必须在其他库之前导入，以避免CUDA初始化问题
import torch
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
import time
import argparse
from collections import Counter
from typing import List, Tuple, Any, Optional, Dict

# LightRAG 相关导入
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.hf import hf_model_complete, hf_embed
from transformers import AutoModel, AutoTokenizer
from lightrag.kg.shared_storage import initialize_pipeline_status

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
DISK_CLEANUP_INTERVAL = 10  # 每处理N个样本后执行一次磁盘清理

# 内存管理配置
MEMORY_CLEANUP_INTERVAL = 20  # 每处理N个样本后执行一次完整内存清理
FORCE_CPU_FOR_EMBEDDING = False  # 如果CUDA问题持续，可以设置为True

# 评估配置
TEST_FILE_PATH = "/root/raptor/test.hard.json"  # 测试文件路径
SHOW_FULL_CONTEXT_SAMPLES = 5  # 展示完整上下文的样本数量

# 工作目录配置
BASE_WORKING_DIR = "./lightrag_selective_test_temp"  # 基础工作目录
LOG_DIR = "./logs_selective"  # 专用日志目录
# ==============================

# 全局变量
_lightrag_instance = None
_lightrag_tasks = set()  # 跟踪所有异步任务
_current_working_dir = None  # 当前使用的工作目录
LAST_LOG_PATH = None  # 保存最新的日志路径

# 确保目录存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BASE_WORKING_DIR, exist_ok=True)

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
            shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)
        logging.debug(f"Sync cleared directory: {directory}")
    except Exception as e:
        logging.error(f"Error during synchronous directory cleanup {directory}: {str(e)}")

async def cleanup_torch_resources():
    """清理PyTorch相关资源，特别是CUDA缓存"""
    try:
        # 1. 强制Python垃圾回收
        gc.collect()
        # 2. 如果CUDA可用，清理CUDA资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
    except Exception as e:
        logging.warning(f"Error during PyTorch resource cleanup: {str(e)}")

async def cleanup_working_directory(directory: str):
    """异步清理单个工作目录"""
    if not os.path.exists(directory):
        return
    try:
        await asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)
        logging.debug(f"Cleaned working directory: {directory}")
    except Exception as e:
        logging.warning(f"Error cleaning directory {directory}: {str(e)}")

async def cleanup_lightrag_resources(full_cleanup: bool = False):
    """彻底清理LightRAG相关资源"""
    global _lightrag_instance, _lightrag_tasks, _current_working_dir
    
    # 1. 清理LightRAG实例
    if _lightrag_instance is not None:
        try:
            await _lightrag_instance.finalize_storages()
        except Exception as e:
            logging.warning(f"Error finalizing LightRAG storages during cleanup: {e}")
        finally:
            del _lightrag_instance
            _lightrag_instance = None
    
    # 2. 清理当前工作目录
    if _current_working_dir is not None:
        await cleanup_working_directory(_current_working_dir)
        _current_working_dir = None
    
    # 3. 取消所有pending的异步任务
    if _lightrag_tasks:
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
    """Initialize LightRAG model with complete isolation."""
    global _lightrag_instance, _current_working_dir
    
    # 每次初始化前彻底清理资源
    await cleanup_lightrag_resources()
    
    # 生成唯一工作目录
    unique_id = f"run_{int(time.time())}_{str(uuid.uuid4())[:6]}"
    working_dir = os.path.join(BASE_WORKING_DIR, unique_id)
    _current_working_dir = working_dir
    
    # 确保工作目录是干净的
    os.makedirs(working_dir, exist_ok=True)
    
    try:
        # 加载 embedding 模型和 tokenizer
        if FORCE_CPU_FOR_EMBEDDING:
            embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, device_map="cpu")
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        else:
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

async def call_lightrag(question: str, context: str, mode: str = DEFAULT_QUERY_MODE) -> Tuple[str, str]:
    """Get predicted answer from LightRAG model."""
    global _lightrag_tasks
    
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
        answer = await loop.run_in_executor(
            None, 
            lambda: LlamaQAModel(model_name="meta-llama/Llama-3.1-8B-Instruct").answer_question(context_str, question)
        )
        
        # 尝试从完整响应中提取答案
        final_answer = await extract_answer_from_response(answer)
        # 基本清理：移除引号、多余空格等
        final_answer = final_answer.strip().strip('"').strip("'")
        return final_answer, context_str
        
    except Exception as e:
        logging.error(f"Error in call_lightrag: {str(e)}")
        return f"Error: {str(e)}", ""

async def evaluate_specific_lines(file_path: str, line_numbers: List[int]) -> Dict[str, Any]:
    """Evaluate only the specified line numbers in the dataset."""
    global LAST_LOG_PATH
    
    # 初始化评估状态
    evaluation_state = {
        "em_total": 0.0,
        "f1_total": 0.0,
        "corrected_em_total": 0.0,
        "corrected_f1_total": 0.0,
        "total": 0,
        "start_time": time.time(),
        "processed_lines": []
    }
    
    # 创建带时间戳的日志文件名
    iso_str = datetime.datetime.now().isoformat().replace(':', '-').replace('.', '_')
    log_filename = f"selective_{iso_str}_lightrag_log.txt"
    full_log_path = os.path.join(LOG_DIR, log_filename)
    LAST_LOG_PATH = full_log_path  # 保存日志路径
    
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
    
    # 明确记录日志文件位置
    logging.info(f"Log file will be saved to: {full_log_path}")
    print(f"Log file will be saved to: {full_log_path}")
    
    # 记录超参数配置
    logging.info(f"Embedding Model: {EMBEDDING_MODEL_NAME}, Dimension: {EMBEDDING_DIM}")
    logging.info(f"Query Mode: {DEFAULT_QUERY_MODE}")
    logging.info(f"Disk Cleanup: Keeping only {MAX_WORKING_DIRS} latest working directories")
    logging.info(f"Evaluating specific lines: {sorted(line_numbers)}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # 读取所有行
            all_lines = f.readlines()
            
            # 按行号排序，确保按顺序处理
            for target_line_num in sorted(line_numbers):
                # 转换为0-based索引
                idx = target_line_num - 1
                
                # 检查行号是否有效
                if idx < 0 or idx >= len(all_lines):
                    logging.warning(f"Line number {target_line_num} is out of range (file has {len(all_lines)} lines), skipping")
                    continue
                
                raw_line = all_lines[idx]
                line_num = idx + 1  # 转回1-based用于显示
                
                try:
                    entry = json.loads(raw_line.strip())
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing line {line_num}: {raw_line[:50]}... | Error: {str(e)}")
                    continue
                
                idx_val = entry.get("idx", "N/A")
                question = entry.get("question", "")
                context = entry.get("context", "")
                targets = entry.get("targets", [])
                
                try:
                    # Get prediction from LightRAG
                    prediction, retrieved_context = await call_lightrag(
                        question, 
                        context, 
                        mode=DEFAULT_QUERY_MODE
                    )
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
                evaluation_state["processed_lines"].append(line_num)
                
                # 决定是否显示完整上下文
                show_full_context = evaluation_state["total"] <= SHOW_FULL_CONTEXT_SAMPLES
                
                # 计算当前平均分数
                current_total = evaluation_state["total"]
                current_em_avg = (evaluation_state["em_total"] / current_total) * 100
                current_f1_avg = (evaluation_state["f1_total"] / current_total) * 100
                current_corrected_em_avg = (evaluation_state["corrected_em_total"] / current_total) * 100
                current_corrected_f1_avg = (evaluation_state["corrected_f1_total"] / current_total) * 100
                
                # Log entry
                containment_status = " (CONTAINMENT CORRECTED)" if corrected_em_score > em_score else ""
                log_entry = (
                    f"\n===== ENTRY {line_num} ====={containment_status}\n"
                    f"{'✅' if em_score == 1 else '❌'}{'✅' if corrected_em_score == 1 and em_score != 1 else ''} "
                    f"[{'CORRECT' if em_score == 1 else 'INCORRECT'}{' -> CORRECT' if corrected_em_score == 1 and em_score != 1 else ''}] ID: {idx_val}\n"
                    f"Question: {question}\n"
                    f"Prediction: {prediction}\n"
                )
                
                # 添加完整上下文（仅前N个示例）
                if show_full_context:
                    log_entry += f"--- RETRIEVED CONTEXT (debug) ---\n{retrieved_context}\n--- END CONTEXT ---\n"
                
                log_entry += (
                    f"Processed Targets: {processed_targets}\n"
                    f"Original EM: {em_score}, F1: {f1_score:.4f}\n"
                    f"Corrected EM: {corrected_em_score}, F1: {corrected_f1_score:.4f}\n"
                    f"Current Averages: EM={current_em_avg:.2f}%, F1={current_f1_avg:.2f}%, "
                    f"C_EM={current_corrected_em_avg:.2f}%, C_F1={current_corrected_f1_avg:.2f}%\n"
                    f"=================\n"
                )
                
                logging.info(log_entry)
                print(log_entry)
                
                # 定期磁盘清理
                if current_total % DISK_CLEANUP_INTERVAL == 0:
                    logging.info(f"Performing disk cleanup after {current_total} samples")
                    await cleanup_old_working_dirs()
                
                # 定期内存清理
                if current_total % MEMORY_CLEANUP_INTERVAL == 0:
                    logging.info(f"Performing memory cleanup after {current_total} samples")
                    await cleanup_torch_resources()
    
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
            
            # 计算总运行时间
            end_time = time.time()
            elapsed_time = end_time - evaluation_state["start_time"]
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours):02d}h:{int(minutes):02d}m:{int(seconds):02d}s"
        else:
            em_avg = f1_avg = corrected_em_avg = corrected_f1_avg = 0.0
            time_str = "00h:00m:00s"
        
        # 创建结果报告
        final_log = (
            f"\n{'='*60}\n"
            f"[COMPLETED] Selective evaluation of {total} specified lines\n"
            f"Model: LightRAG with custom LlamaQAModel\n"
            f"Embedding Model: {EMBEDDING_MODEL_NAME}, Dimension: {EMBEDDING_DIM}\n"
            f"Query Mode: {DEFAULT_QUERY_MODE}\n"
            f"Processed Lines: {sorted(evaluation_state['processed_lines'])}\n"
            f"Original Scores: EM={em_avg:.2f}%, F1={f1_avg:.2f}%\n"
            f"Corrected Scores: EM={corrected_em_avg:.2f}%, F1={corrected_f1_avg:.2f}%\n"
            f"Improvement: EM +{corrected_em_avg - em_avg:.2f}%, F1 +{corrected_f1_avg - f1_avg:.2f}%\n"
            f"Total Runtime: {time_str}\n"
            f"Log file saved to: {full_log_path}\n"
            f"{'='*60}\n"
        )
        
        # 记录到日志
        logging.info(final_log)
        # 强制刷新所有日志
        for handler in logging.getLogger().handlers:
            handler.flush()
        # 同时打印到控制台
        print(final_log)
        
        return evaluation_state

def parse_line_numbers(line_str: str) -> List[int]:
    """解析命令行传入的行号字符串，支持范围和单个数字"""
    result = []
    parts = line_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 处理范围，如 "1-10"
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            # 处理单个数字
            result.append(int(part))
    return sorted(set(result))  # 去重并排序

def read_lines_from_file(file_path: str) -> List[int]:
    """从文件中读取行号，每行一个数字"""
    try:
        with open(file_path, 'r') as f:
            return sorted(set(int(line.strip()) for line in f if line.strip().isdigit()))
    except Exception as e:
        logging.error(f"Error reading line numbers from file {file_path}: {str(e)}")
        return []

async def main_async():
    """异步主函数"""
    parser = argparse.ArgumentParser(description='LightRAG Selective Line Evaluation')
    parser.add_argument('--file', type=str, default=TEST_FILE_PATH, help='Test file path')
    parser.add_argument('--lines', type=str, 
                       help='Comma-separated line numbers or ranges to evaluate (e.g., "1,5,10-15")')
    parser.add_argument('--lines-file', type=str,
                       help='File containing line numbers to evaluate (one per line)')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU for embedding')
    
    args = parser.parse_args()
    
    # 设置全局变量
    global FORCE_CPU_FOR_EMBEDDING
    if args.force_cpu:
        FORCE_CPU_FOR_EMBEDDING = True
        logging.info("Forcing CPU for embedding operations")
    
    # 程序启动时同步清理基础工作目录
    clear_directory_sync(BASE_WORKING_DIR)
    
    # 设置PyTorch内存限制
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)
        logging.info(f"PyTorch memory limit set to 85% of GPU capacity")
    
    # 检查文件是否存在
    if not os.path.exists(args.file):
        print(f"错误: 测试文件 '{args.file}' 不存在!")
        sys.exit(1)
    
    # 获取要评估的行号
    line_numbers = []
    if args.lines:
        line_numbers = parse_line_numbers(args.lines)
    elif args.lines_file:
        line_numbers = read_lines_from_file(args.lines_file)
    else:
        print("错误: 必须指定 --lines 或 --lines-file 参数")
        sys.exit(1)
    
    if not line_numbers:
        print("错误: 没有有效的行号需要评估")
        sys.exit(1)
    
    print(f"开始 LightRAG 选择性测试，使用文件: {args.file}")
    print(f"将评估以下行号: {line_numbers}")
    print(f"日志文件将保存在: {LOG_DIR}/ 目录下")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Query Mode: {DEFAULT_QUERY_MODE}")
    print(f"Force CPU for Embedding: {FORCE_CPU_FOR_EMBEDDING}")
    
    # 运行评估
    await evaluate_specific_lines(args.file, line_numbers)

def main():
    """同步主函数"""
    try:
        # 设置asyncio策略以处理Windows和Linux差异
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        asyncio.run(main_async())
        exit_code = 0
    except KeyboardInterrupt:
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
            clear_directory_sync(BASE_WORKING_DIR)
        except:
            pass
        sys.exit(exit_code)

if __name__ == "__main__":
    main()