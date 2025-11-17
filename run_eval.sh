#!/bin/bash
# 运行LightRAG评估，支持检查点恢复

LOG_FILE="evaluation_master.log"
MAX_RESTARTS=5
RESTART_COUNT=0
USER_INTERRUPTED=0
PYTHON_PID=""

# 信号处理函数
handle_signal() {
    echo "[$(date)] Received SIGINT/SIGTERM, initiating clean shutdown..."
    USER_INTERRUPTED=1
    
    # 向当前Python进程发送SIGTERM
    if [ -n "$PYTHON_PID" ] && kill -0 $PYTHON_PID 2>/dev/null; then
        echo "[$(date)] Sending SIGTERM to Python process (PID: $PYTHON_PID)"
        kill -SIGTERM $PYTHON_PID 2>/dev/null
        # 等待Python进程优雅退出
        wait $PYTHON_PID 2>/dev/null
    else
        echo "[$(date)] No active Python process to terminate"
    fi
}

# 注册信号处理器
trap handle_signal SIGINT SIGTERM

# 记录开始时间
echo "[$(date)] Starting evaluation process" >> $LOG_FILE
echo "[$(date)] Starting evaluation process"

# 持续运行直到完成
while true; do
    if [ $USER_INTERRUPTED -eq 1 ]; then
        echo "[$(date)] User interrupted, exiting immediately." >> $LOG_FILE
        echo "[$(date)] User interrupted, exiting immediately."
        break
    fi

    echo "[$(date)] Starting Python evaluation script (restart count: $RESTART_COUNT)" >> $LOG_FILE
    
    # 运行Python脚本
    python promax_atc.py >> $LOG_FILE 2>&1 &
    PYTHON_PID=$!
    wait $PYTHON_PID
    exit_code=$?
    
    echo "[$(date)] Python script exited with code: $exit_code" >> $LOG_FILE

    # 检查是否用户中断
    if [ $USER_INTERRUPTED -eq 1 ]; then
        echo "[$(date)] User interrupted, exiting without restart." >> $LOG_FILE
        echo "[$(date)] User interrupted, exiting without restart."
        break
    fi

    # 检查退出状态
    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] Evaluation completed successfully!" >> $LOG_FILE
        echo "[$(date)] Evaluation completed successfully!"
        break
    elif [ $exit_code -eq 3 ]; then  # Restartable critical error
        echo "[$(date)] RESTARTABLE ERROR detected (exit code 3). Checkpoint saved, restarting script..." >> $LOG_FILE
        echo "[$(date)] RESTARTABLE ERROR detected (exit code 3). Checkpoint saved, restarting script..."
        # 等待几秒让系统清理资源
        sleep 5
        continue
    elif [ $exit_code -eq 2 ]; then  # Critical error that shouldn't be restarted
        echo "[$(date)] CRITICAL ERROR detected (exit code 2). Exiting immediately." >> $LOG_FILE
        echo "[$(date)] CRITICAL ERROR detected (exit code 2). Exiting immediately."
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

# 最终清理
if [ -n "$PYTHON_PID" ] && kill -0 $PYTHON_PID 2>/dev/null; then
    echo "[$(date)] Forcibly terminating any remaining Python process (PID: $PYTHON_PID)"
    kill -9 $PYTHON_PID 2>/dev/null
fi