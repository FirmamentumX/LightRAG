#!/bin/bash
# 运行LightRAG评估，支持检查点恢复

LOG_FILE="evaluation_master_hard.log"
MAX_RESTARTS=5
RESTART_COUNT=0

# 清理函数，确保程序退出时进行清理
cleanup() {
    echo "[$(date)] Received signal, cleaning up..."
    # 这里可以添加任何需要的清理操作
}
trap cleanup SIGINT SIGTERM

# 记录开始时间
echo "[$(date)] Starting evaluation process" >> $LOG_FILE
echo "[$(date)] Starting evaluation process"

# 持续运行直到完成
while true; do
    echo "[$(date)] Starting Python evaluation script (restart count: $RESTART_COUNT)" >> $LOG_FILE
    
    # 运行Python脚本，追加输出到日志
    python promax_atc_hard.py >> $LOG_FILE 2>&1
    exit_code=$?
    
    echo "[$(date)] Python script exited with code: $exit_code" >> $LOG_FILE
    
    # 检查退出码
    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] Evaluation completed successfully!" >> $LOG_FILE
        echo "[$(date)] Evaluation completed successfully!"
        break
    elif grep -q "\[CHECKPOINT\] Progress saved, exiting for restart" $LOG_FILE; then
        echo "[$(date)] Checkpoint reached, restarting script..." >> $LOG_FILE
        echo "[$(date)] Checkpoint reached, restarting script..."
        # 等待几秒让系统清理资源
        sleep 5
        continue
    else
        RESTART_COUNT=$((RESTART_COUNT + 1))
        if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
            echo "[$(date)] Maximum restart attempts ($MAX_RESTARTS) reached. Exiting." >> $LOG_FILE
            echo "[$(date)] Maximum restart attempts ($MAX_RESTARTS) reached. Exiting."
            break
        fi
        
        echo "[$(date)] Script interrupted or crashed (exit code $exit_code). Attempting restart ($RESTART_COUNT/$MAX_RESTARTS)..." >> $LOG_FILE
        echo "[$(date)] Script interrupted or crashed (exit code $exit_code). Attempting restart ($RESTART_COUNT/$MAX_RESTARTS)..."
        
        # 等待几秒
        sleep 10
    fi
done

echo "[$(date)] Evaluation process finished" >> $LOG_FILE
echo "[$(date)] Evaluation process finished"