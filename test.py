import json
import re

fileKey = "https://vvs-1259796308.cos.ap-guangzhou.myqcloud.com/TTS/TASK_RESULT/202502240823_1_2.wav"
file_name = re.search(r"/([^/]+)$", fileKey).group(1)

print("提取的文件名是:", file_name)
