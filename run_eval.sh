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

# 检查检查点文件是否存在
check_checkpoint_exists() {
    if [ -f "./checkpoint.easy.json" ]; then
        return 0  # 存在
    else
        return 1  # 不存在
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
    
    # 记录重启前的检查点状态
    if check_checkpoint_exists; then
        echo "[$(date)] Checkpoint file exists before starting Python script" >> $LOG_FILE
        # 记录检查点文件的详细信息
        ls -la ./checkpoint.easy.json >> $LOG_FILE 2>&1
    else
        echo "[$(date)] No checkpoint file found before starting Python script" >> $LOG_FILE
    fi
    
    # 运行Python脚本
    python promax_atc.py >> $LOG_FILE 2>&1 &
    PYTHON_PID=$!
    wait $PYTHON_PID
    exit_code=$?
    
    echo "[$(date)] Python script exited with code: $exit_code" >> $LOG_FILE

    # 检查退出后检查点状态
    if check_checkpoint_exists; then
        echo "[$(date)] Checkpoint file exists after Python script exit" >> $LOG_FILE
        ls -la ./checkpoint.easy.json >> $LOG_FILE 2>&1
    else
        echo "[$(date)] Checkpoint file DOES NOT exist after Python script exit" >> $LOG_FILE
    fi

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
        # 仅在成功完成时删除检查点
        if check_checkpoint_exists; then
            echo "[$(date)] Removing checkpoint file after successful completion" >> $LOG_FILE
            rm -f ./checkpoint.easy.json
        fi
        break
    elif [ $exit_code -eq 3 ]; then  # Restartable critical error
        echo "[$(date)] RESTARTABLE ERROR detected (exit code 3). Checkpoint saved, restarting script..." >> $LOG_FILE
        echo "[$(date)] RESTARTABLE ERROR detected (exit code 3). Checkpoint saved, restarting script..."
        
        # 额外确认检查点状态
        if ! check_checkpoint_exists; then
            echo "[$(date)] WARNING: Checkpoint file missing despite exit code 3!" >> $LOG_FILE
            echo "[$(date)] Attempting to recover from last known state..."
        else
            echo "[$(date)] Checkpoint file verified to exist before restart" >> $LOG_FILE
        fi
        
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
        
        # 验证检查点确实存在
        if ! check_checkpoint_exists; then
            echo "[$(date)] WARNING: Expected checkpoint file not found after checkpoint message!" >> $LOG_FILE
        fi
        
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
        
        # 验证检查点状态
        if check_checkpoint_exists; then
            echo "[$(date)] Checkpoint file exists despite unexpected exit code $exit_code" >> $LOG_FILE
        else
            echo "[$(date)] WARNING: No checkpoint file found with unexpected exit code $exit_code" >> $LOG_FILE
        fi
        
        # 等待几秒
        sleep 10
    fi
done

echo "[$(date)] Evaluation process finished" >> $LOG_FILE
echo "[$(date)] Evaluation process finished"

# 最终状态检查
if check_checkpoint_exists; then
    echo "[$(date)] Final check: Checkpoint file still exists" >> $LOG_FILE
    ls -la ./checkpoint.easy.json >> $LOG_FILE 2>&1
else
    echo "[$(date)] Final check: No checkpoint file exists" >> $LOG_FILE
fi

# 最终清理
if [ -n "$PYTHON_PID" ] && kill -0 $PYTHON_PID 2>/dev/null; then
    echo "[$(date)] Forcibly terminating any remaining Python process (PID: $PYTHON_PID)"
    kill -9 $PYTHON_PID 2>/dev/null
fi

# 安全删除检查点 - 只有在成功完成且用户未中断时才删除
if [ $exit_code -eq 0 ] && [ $USER_INTERRUPTED -eq 0 ]; then
    if check_checkpoint_exists; then
        echo "[$(date)] Final cleanup: Removing checkpoint file after successful completion" >> $LOG_FILE
        rm -f ./checkpoint.easy.json
    fi
else
    # 非正常退出，保留检查点
    if check_checkpoint_exists; then
        echo "[$(date)] Final cleanup: Preserving checkpoint file due to non-zero exit code ($exit_code) or user interruption" >> $LOG_FILE
    else
        echo "[$(date)] Final cleanup: No checkpoint file to preserve" >> $LOG_FILE
    fi
fi

echo "[$(date)] Process fully terminated" >> $LOG_FILE
echo "[$(date)] Process fully terminated"