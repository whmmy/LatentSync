import argparse
from datetime import datetime
import os
import queue
import re
import urllib.request
from pathlib import Path
import logging
import redis
import requests
import yaml
from omegaconf import OmegaConf
from qcloud_cos import CosConfig, CosS3Client

from scripts.inference import main
from tools.ret import *

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(file_path):
    """加载配置文件"""
    with open(file_path) as f:
        return yaml.safe_load(f)


def create_directory(path):
    """创建目录"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_redis_client(config):
    """初始化 Redis 连接"""
    return redis.Redis(
        host=config['redis']['host'],
        port=config['redis']['port'],
        db=config['redis']['db'],
        password=config['redis']['password']
    )


def init_cos_client(config):
    """初始化 COS 客户端"""
    bucket = config['cos']['bucket']
    cosConfig = CosConfig(
        Region=config['cos']['region'],
        SecretId=config['cos']['secret_id'],
        SecretKey=config['cos']['secret_key'],
    )
    return CosS3Client(cosConfig), bucket


# 加载配置
config = load_config('./process_redis.yaml')
# 初始化 Redis 连接
redis_client = init_redis_client(config)
# 初始化 OSS 客户端
cosClient, bucket = init_cos_client(config)
redis_queue = 'LS:TaskQueue'

# 创建文件上传队列
fileUploadQueue = queue.Queue()


def download_file_bytes(url):
    try:
        # 打开 URL 并获取响应
        with urllib.request.urlopen(url) as response:
            # 读取响应内容，返回的是 bytes 类型
            return response.read()
    except Exception as e:
        logging.error(f"获取 URL 内容时出现错误: {e}")
        return None


def download_file(url, local_path):
    """
    下载文件
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        with open(local_path, 'wb') as f:
            f.write(response.content)
        logging.info(f'文件下载完成: {local_path}')
        return local_path
    except Exception as e:
        logging.error(f"下载文件 {url} 时出错: {e}")
        return None


def upload_to_cos(path, oss_key):
    """
    上传文件到 cos
    """
    try:
        logging.info(f'开始上传文件到 cos, oss_key:{oss_key}')
        response = cosClient.upload_file(
            Bucket=bucket,
            LocalFilePath=path,
            Key=oss_key,
            EnableMD5=False,
        )
        return True
    except Exception as e:
        logging.error(f"上传文件 {path} 到 COS 失败: {e}")
        return False


def process_redis_queue():
    """
    处理 redis 队列
    """
    logging.info('开始处理redis队列')
    while True:
        try:
            # 从redis队列中获取任务
            task = redis_client.blpop([redis_queue], timeout=0)
            if task:
                ret = JsonRet()
                logging.info(task)
                # 判断是否是json且合法格式
                task_data_str = task[1].decode('utf-8')
                task_dict = json.loads(task_data_str)
                audio_file_url = task_dict.get('audioFileUrl')
                video_file_url = task_dict.get('videoFileUrl')
                person_id = task_dict.get('personId')
                taskId = task_dict.get('taskId')

                task_result_key = f'LS:task_result:{taskId}'

                audio_dir = create_directory("./tempAudio")
                audioFileName = get_file_name_by_url(audio_file_url)
                # 音频每次重新下载，推理完后默认删除
                audioLocalPath = download_file(audio_file_url, str(audio_dir / audioFileName))
                if not audioLocalPath:
                    logging.warning(f'音频文件下载失败，跳过此任务')
                    continue

                video_dir = create_directory("./tempVideo")
                videoFileName = get_file_name_by_url(video_file_url)
                videoLocalPath = str(video_dir / videoFileName)
                if os.path.exists(videoLocalPath):
                    logging.info(f'视频文件已经存在无需下载，{videoLocalPath}')
                else:
                    videoLocalPath = download_file(video_file_url, videoLocalPath)
                    if not videoLocalPath:
                        logging.warning(f'视频文件下载失败，跳过此任务')
                        continue

                output_dir = create_directory("./fakeVideo")
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Set the output path for the processed video
                output_video_path = str(output_dir / f"{taskId}_{person_id}_{current_time}.mp4")
                if process_video(videoLocalPath, audioLocalPath, output_video_path):
                    fileUploadQueue.put({
                        'outPutVideoPath': output_video_path,
                        'tempAudioPath': audioLocalPath,
                        'taskId': taskId,
                        'personId': person_id,
                    })
                else:
                    logging.warning(f'视频处理失败，跳过此任务')
        except Exception as e:
            logging.error(f"处理 Redis 队列任务时出错: {e}")


CONFIG_PATH = Path("configs/unet/second_stage.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")


def process_video(
        video_path,
        audio_path,
        output_video_path
):
    try:
        uNetConfig = OmegaConf.load(CONFIG_PATH)

        uNetConfig["run"].update(
            {
                "guidance_scale": 1.5,
                "inference_steps": 20,
            }
        )

        # Parse the arguments
        args = create_args(video_path, audio_path, output_video_path, 20, 1.5, 1247)

        result = main(
            config=uNetConfig,
            args=args,
        )
        logging.info("Processing completed successfully.")
        return output_video_path  # Ensure the output path is returned
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        return False


def create_args(
        video_path: str, audio_path: str, output_path: str, inference_steps: int, guidance_scale: float, seed: int
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            CHECKPOINT_PATH.absolute().as_posix(),
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
        ]
    )


def putTaskStatus(key, ret: JsonRet):
    logging.info(f'更新redis状态,key:{key},ret:{ret()}')
    redis_client.set(key, ret())


# 功能2：通过上传文件的队列，获取文件地址并上传
def process_upload_queue():
    logging.info('开始处理文件上传队列')
    while True:
        try:
            # 从上传队列中获取文件地址
            uploadTask = fileUploadQueue.get()
            if uploadTask:
                logging.info('获取uploadTask', uploadTask)
                outPutVideoPath = uploadTask.get('outPutVideoPath')
                tempAudioPath = uploadTask.get('tempAudioPath')
                personId = uploadTask.get('personId')
                taskId = uploadTask.get('taskId')
                task_result_key = f'LS:task_result:{taskId}'

                # 进行COS文件上传
                fakeVideoName = generate_filename(personId, taskId, 'mp4')
                if not upload_to_cos(outPutVideoPath, fakeVideoName):
                    logging.warning(f'文件 {outPutVideoPath} 上传失败，重新加入队列')
                    fileUploadQueue.put(uploadTask)
                    continue

                # 上传完成后，改变redisKey中的状态值
                ret = JsonRet()
                ret.set_code(ret.RET_OK)
                ret.set_data({
                    "fileKey": cosClient.get_object_url(bucket, fakeVideoName),
                    "time": datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                })
                putTaskStatus(task_result_key, ret)
                # 上传完毕后删除音频和fake视频文件
                delete_file(tempAudioPath)
                delete_file(outPutVideoPath)
        except Exception as e:
            logging.error(f"处理文件上传队列任务时出错: {e}")


def generate_filename(person_id, taskId, file_format):
    """
   生成包含时间、person_id、taskId和文件格式的随机文件名
   :param person_id: 人员ID
   :param taskId: 任务ID
   :param file_format: 文件格式，如 'jpg', 'png', 'pdf' 等
   :return: 生成的随机文件名
   """
    # 获取当前时间并格式化为字符串，只保留到分钟
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    # 组合时间、person_id、taskId和文件格式生成文件名
    filename = f"/LS/TASK_RESULT/{current_time}_{person_id}_{taskId}.{file_format}"
    return filename


def get_file_name_by_url(cosUrl):
    file_name = re.search(r"/([^/]+)$", cosUrl).group(1)
    return file_name


def delete_file(file_path):
    """
    删除指定路径的文件，并对可能出现的错误进行控制处理。

    :param file_path: 要删除的文件的路径
    :return: 如果文件删除成功返回 True，否则返回 False
    """
    path = Path(file_path)
    try:
        if path.exists():
            path.unlink()
            logging.info(f"文件 {file_path} 已成功删除。")
            return True
        else:
            logging.info(f"文件 {file_path} 不存在，无需删除。")
            return False
    except PermissionError as pe:
        error_msg = f"没有权限删除文件 {file_path}。错误信息: {pe}"
        logging.error(error_msg)
        return False
    except Exception as e:
        error_msg = f"删除文件 {file_path} 时出现未知错误。错误信息: {e}"
        logging.error(error_msg)
        return False


if __name__ == "__main__":
    import threading

    # 启动两个线程分别处理两个功能
    t1 = threading.Thread(target=process_redis_queue)
    t2 = threading.Thread(target=process_upload_queue)

    t1.start()
    t2.start()

    t1.join()
    t2.join()
