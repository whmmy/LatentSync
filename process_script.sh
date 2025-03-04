#!/bin/sh
### BEGIN INIT INFO
# Provides:          my_python_script
# Required-Start:    $remote_fs $syslog
# Required-Stop:     $remote_fs $syslog
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: Start/stop my Python script
# Description:       Start/stop my Python script
### END INIT INFO

# 脚本的工作目录
WORKING_DIR="/workspace/LatentSync"
# Python 脚本的路径
SCRIPT="/workspace/LatentSync/run_process_redis.py"
# Python 解释器的路径
DAEMON="/root/miniforge3/bin/python3.10"
# 服务的名称
NAME="ls_process"
# 服务的 PID 文件路径
PIDFILE="/var/run/$NAME.pid"

case "$1" in
    start)
        echo "Starting $NAME"
        cd $WORKING_DIR
        $DAEMON $SCRIPT &
        echo $! > $PIDFILE
        ;;
    stop)
        echo "Stopping $NAME"
        if [ -f $PIDFILE ]; then
            PID=$(cat $PIDFILE)
            kill $PID
            rm $PIDFILE
        fi
        ;;
    restart)
        $0 stop
        $0 start
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac

exit 0