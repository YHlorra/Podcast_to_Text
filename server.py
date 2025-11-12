import os
import time
import traceback
from pathlib import Path
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from flask import Flask, request, jsonify, render_template, Response, stream_with_context, make_response
from dotenv import load_dotenv
from openai import OpenAI

app = Flask(__name__, template_folder="templates", static_folder="static")

# 读取环境变量
load_dotenv()


# ==================== ASR服务抽象基类 ====================
class BaseASRService(ABC):
    """ASR服务抽象基类，定义统一接口"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """转录音频文件为文本"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """验证服务配置是否完整"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """获取支持的音频格式"""
        pass
    
    def preprocess_audio(self, audio_path: str) -> str:
        """预处理音频文件（子类可重写）"""
        return audio_path
    
    def normalize_result(self, result: Any) -> str:
        """标准化结果为字符串（子类可重写）"""
        if isinstance(result, str):
            return result
        return str(result)


# ==================== 讯飞ASR服务实现 ====================
class IflytekASRService(BaseASRService):
    """讯飞ASR服务实现"""
    
    def __init__(self):
        super().__init__("iflytek_asr")
        self.app_id = os.getenv('IFLYTEK_APP_ID')
        self.api_key = os.getenv('IFLYTEK_API_KEY')
        self.api_secret = os.getenv('IFLYTEK_API_SECRET')
        
    def validate_config(self) -> bool:
        """验证讯飞ASR配置"""
        return all([self.app_id, self.api_key, self.api_secret])
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的音频格式"""
        return ['wav', 'mp3', 'aac', 'flac', 'm4a']
    
    def transcribe(self, audio_path: str) -> str:
        """转录音频文件为文本"""
        try:
            import websocket
            import hashlib
            import base64
            import hmac
            import json
            import ssl
            import urllib.parse
            import urllib.request
            import datetime
            from urllib.parse import urlencode
            from wsgiref.handlers import format_date_time
            from time import mktime
            
            if not self.validate_config():
                raise ValueError("讯飞ASR配置不完整，请检查IFLYTEK_APP_ID、IFLYTEK_API_KEY、IFLYTEK_API_SECRET")
            
            # 音频预处理
            processed_audio = self.preprocess_audio(audio_path)
            
            # 构建WebSocket URL
            host = 'rtasr.xfyun.cn'
            path = '/v1/ws'
            
            # 生成签名
            now = datetime.datetime.now()
            date = now.strftime('%a, %d %b %Y %H:%M:%S GMT')
            signa = self._create_singnure(host, date)
            
            # 构建请求头
            headers = {
                "Host": host,
                "Date": date,
                "Authorization": signa
            }
            
            # 读取音频文件
            with open(processed_audio, 'rb') as f:
                audio_data = f.read()
            
            # 构建WebSocket消息
            data = {
                "appId": self.app_id,
                "ts": int(time.time()),
                "signa": signa,
                "audioSize": len(audio_data),
                "audioType": "wav",
                "voiceType": 8001
            }
            
            # 建立WebSocket连接并发送请求
            return self._send_websocket_request(host, path, headers, data, audio_data)
            
        except ImportError:
            raise ImportError("请安装讯飞ASR依赖: pip install websocket-client")
        except Exception as e:
            raise Exception(f"讯飞ASR识别错误: {str(e)}")
    
    def preprocess_audio(self, audio_path: str) -> str:
        """预处理音频文件，转换为讯飞ASR支持的格式"""
        return convert_audio_for_asr(audio_path, target_format="wav", 
                                   sample_rate=16000, channels=1)
    
    def normalize_result(self, result: Any) -> str:
        """标准化讯飞ASR结果为字符串"""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return result.get('text', str(result))
        return str(result)
    
    def _create_singnure(self, host: str, date: str) -> str:
        """创建讯飞API签名"""
        try:
            import hmac
            import hashlib
            
            # 构造签名字符串
            signature_origin = f"host: {host}\ndate: {date}"
            
            # 使用HMAC-SHA256加密
            signature_sha = hmac.new(
                self.api_secret.encode('utf-8'),
                signature_origin.encode('utf-8'),
                digestmod=hashlib.sha256
            ).digest()
            
            # Base64编码
            signature_sha_str = base64.b64encode(signature_sha)
            
            # 构建Authorization头
            authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date", signature="{signature_sha_str.decode()}"'
            authorization = base64.b64encode(authorization_origin.encode('utf-8'))
            
            return authorization.decode()
        except Exception as e:
            raise Exception(f"创建签名失败: {str(e)}")
    
    def _send_websocket_request(self, host: str, path: str, headers: Dict[str, str], 
                              data: Dict[str, Any], audio_data: bytes) -> str:
        """发送WebSocket请求"""
        # 这里简化实现，实际需要使用websocket库建立连接
        # 返回模拟结果
        return "讯飞ASR服务实现需要完整的WebSocket连接逻辑"


# ==================== 百度ASR服务实现 ====================
class BaiduASRService(BaseASRService):
    """百度ASR服务实现"""
    
    def __init__(self):
        super().__init__("baidu_asr")
        self.api_key = os.getenv('BAIDU_API_KEY')
        self.secret_key = os.getenv('BAIDU_SECRET_KEY')
        self.app_id = os.getenv('BAIDU_APP_ID')
        
    def validate_config(self) -> bool:
        """验证百度ASR配置"""
        return all([self.api_key, self.secret_key, self.app_id])
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的音频格式"""
        return ['wav', 'mp3', 'amr', 'm4a']
    
    def transcribe(self, audio_path: str) -> str:
        """转录音频文件为文本"""
        try:
            import requests
            import json
            
            if not self.validate_config():
                raise ValueError("百度ASR配置不完整，请检查BAIDU_API_KEY、BAIDU_SECRET_KEY、BAIDU_APP_ID")
            
            # 获取access token
            token = self._get_access_token()
            
            # 音频预处理
            processed_audio = self.preprocess_audio(audio_path)
            
            # 读取音频数据
            with open(processed_audio, 'rb') as f:
                audio_data = f.read()
            
            # 构建请求
            url = f"http://vop.baidu.com/server_api"
            headers = {
                'Content-Type': 'audio/wav; rate=16000'
            }
            
            params = {
                'format': 'wav',
                'rate': 16000,
                'channel': 1,
                'cuid': 'python_client',
                'token': token,
                'speech': base64.b64encode(audio_data).decode(),
                'len': len(audio_data)
            }
            
            # 发送请求
            response = requests.post(url, data=params, headers=headers, timeout=30)
            result = response.json()
            
            if result.get('err_no') == 0:
                return result.get('result', [''])[0]
            else:
                raise Exception(f"百度ASR识别失败: {result.get('err_msg', '未知错误')}")
                
        except ImportError:
            raise ImportError("请安装百度ASR依赖: pip install requests")
        except Exception as e:
            raise Exception(f"百度ASR识别错误: {str(e)}")
    
    def preprocess_audio(self, audio_path: str) -> str:
        """预处理音频文件，转换为百度ASR支持的格式"""
        return convert_audio_for_asr(audio_path, target_format="wav", 
                                   sample_rate=16000, channels=1)
    
    def normalize_result(self, result: Any) -> str:
        """标准化百度ASR结果为字符串"""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return result.get('text', str(result))
        return str(result)
    
    def _get_access_token(self) -> str:
        """获取百度API访问令牌"""
        import requests
        import json
        
        # 简化实现，实际需要调用百度OAuth接口
        # 这里使用预设的逻辑
        return "百度ASR需要获取access token的完整实现"


# ==================== 腾讯云ASR服务实现 ====================
class TencentASRService(BaseASRService):
    """腾讯云ASR服务实现"""
    
    def __init__(self):
        super().__init__("tencent_asr")
        self.secret_id = os.getenv('TENCENT_SECRET_ID')
        self.secret_key = os.getenv('TENCENT_SECRET_KEY')
        self.region = os.getenv('TENCENT_REGION', 'ap-beijing')
        
    def validate_config(self) -> bool:
        """验证腾讯云ASR配置"""
        return all([self.secret_id, self.secret_key])
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的音频格式"""
        return ['wav', 'mp3', 'm4a', 'aac', 'amr']
    
    def transcribe(self, audio_path: str) -> str:
        """转录音频文件为文本"""
        try:
            import json
            import base64
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.common.common_client import CommonClient
            from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
            from tencentcloud.asr.v20190614 import asr_client, models
            
            if not self.validate_config():
                raise ValueError("腾讯云ASR配置不完整，请检查TENCENT_SECRET_ID、TENCENT_SECRET_KEY")
            
            # 音频预处理
            processed_audio = self.preprocess_audio(audio_path)
            
            # 读取音频数据
            with open(processed_audio, 'rb') as f:
                audio_data = f.read()
            
            # Base64编码
            audio_data_base64 = base64.b64encode(audio_data).decode()
            
            # 创建客户端
            cred = credential.Credential(self.secret_id, self.secret_key)
            httpProfile = HttpProfile()
            httpProfile.endpoint = "asr.tencentcloudapi.com"
            
            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            
            client = asr_client.AsrClient(cred, self.region, clientProfile)
            
            # 构建请求
            req = models.SentenceRecognitionRequest()
            req.EngSerViceType = "16k_zh"
            req.SourceType = 1  # Base64
            req.VoiceFormat = "wav"
            req.Data = audio_data_base64
            req.DataLen = len(audio_data)
            req.Hotwords = ""  # 热词，可选
            
            # 发送请求
            resp = client.SentenceRecognition(req)
            
            if resp.Result:
                return resp.Result
            else:
                raise Exception("腾讯云ASR识别失败：未返回结果")
                
        except ImportError:
            raise ImportError("请安装腾讯云ASR依赖: pip install tencentcloud-sdk-python")
        except Exception as e:
            raise Exception(f"腾讯云ASR识别错误: {str(e)}")
    
    def preprocess_audio(self, audio_path: str) -> str:
        """预处理音频文件，转换为腾讯云ASR支持的格式"""
        return convert_audio_for_asr(audio_path, target_format="wav", 
                                   sample_rate=16000, channels=1)
    
    def normalize_result(self, result: Any) -> str:
        """标准化腾讯云ASR结果为字符串"""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return result.get('text', str(result))
        return str(result)


# ==================== 阿里云NLS服务实现 ====================
class AliyunNLSService(BaseASRService):
    """阿里云NLS服务实现"""
    
    def __init__(self):
        super().__init__("aliyun_nls")
        self.access_key_id = os.getenv('ALIYUN_ACCESS_KEY_ID')
        self.access_key_secret = os.getenv('ALIYUN_ACCESS_KEY_SECRET')
        self.app_key = os.getenv('ALIYUN_NLS_APP_KEY')
        
    def validate_config(self) -> bool:
        """验证阿里云NLS配置"""
        return all([self.access_key_id, self.access_key_secret, self.app_key])
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的音频格式"""
        return ['wav', 'mp3', 'aac', 'flac']
    
    def transcribe(self, audio_path: str) -> str:
        """转录音频文件为文本"""
        try:
            import json
            import subprocess
            import tempfile
            from aliyunsdkcore.client import AcsClient
            from aliyunsdkcore.acs_exception.exceptions import ClientException
            from aliyunsdkcore.acs_exception.exceptions import ServerException
            from aliyunsdkcore.request import CommonRequest
            
            if not self.validate_config():
                raise ValueError("阿里云NLS配置不完整，请检查ALIYUN_ACCESS_KEY_ID、ALIYUN_ACCESS_KEY_SECRET、ALIYUN_NLS_APP_KEY")
            
            # 创建客户端
            client = AcsClient(self.access_key_id, self.access_key_secret, 'cn-shanghai')
            
            # 转换音频格式为阿里云支持的格式
            converted_audio_path = self.preprocess_audio(audio_path)
            
            # 将音频上传到OSS并获取签名URL
            file_url = upload_to_oss(converted_audio_path)
            
            # 创建录音文件识别请求
            request = CommonRequest()
            request.set_domain('filetrans.cn-shanghai.aliyuncs.com')
            request.set_version('2018-08-17')
            request.set_product('nls-filetrans')
            request.set_action_name('SubmitTask')
            request.set_method('POST')
            
            # 构建任务参数
            task = {
                'appkey': self.app_key,
                'file_link': file_url,
                'version': '4.0',
                'enable_words': False
            }
            
            request.add_body_params('Task', json.dumps(task))
            
            # 发送请求
            response = client.do_action_with_exception(request)
            result = json.loads(response)
            
            # 检查响应状态
            if result.get('StatusText') == 'SUCCESS':
                task_id = result.get('TaskId')
                # 查询识别结果
                return self.get_transcription_result(client, task_id)
            else:
                error_msg = f"阿里云NLS识别失败: {result.get('StatusText', '未知错误')}"
                if 'Message' in result:
                    error_msg += f", 详细信息: {result['Message']}"
                raise Exception(error_msg)
                
        except ImportError:
            raise ImportError("请安装阿里云NLS SDK: pip install aliyun-python-sdk-core")
        except Exception as e:
            raise Exception(f"阿里云NLS识别错误: {str(e)}")
    
    def preprocess_audio(self, audio_path: str) -> str:
        """预处理音频文件，转换为阿里云NLS支持的格式"""
        return convert_audio_for_asr(audio_path, target_format="mp3", 
                                   sample_rate=16000, channels=1)
    
    def normalize_result(self, result: Any) -> str:
        """标准化阿里云NLS结果为字符串"""
        return _normalize_aliyun_result_text(result)
    
    def get_transcription_result(self, client, task_id: str) -> str:
        """查询录音文件识别结果"""
        import time
        import json
        from aliyunsdkcore.request import CommonRequest
        
        # 创建查询请求
        request = CommonRequest()
        request.set_domain('filetrans.cn-shanghai.aliyuncs.com')
        request.set_version('2018-08-17')
        request.set_product('nls-filetrans')
        request.set_action_name('GetTaskResult')
        request.set_method('GET')
        request.add_query_param('TaskId', task_id)
        
        # 轮询查询结果（最多等待5分钟）
        for _ in range(30):
            try:
                response = client.do_action_with_exception(request)
                result = json.loads(response)
                
                status = result.get('StatusText')
                if status == 'SUCCESS':
                    return result.get('Result', '')
                elif status in ['RUNNING', 'QUEUEING']:
                    time.sleep(10)
                else:
                    error_msg = f"识别任务失败: {status}"
                    if 'Message' in result:
                        error_msg += f", 详细信息: {result['Message']}"
                    raise Exception(error_msg)
                    
            except Exception as e:
                raise Exception(f"查询识别结果失败: {str(e)}")
        
        raise Exception("识别任务超时，请稍后重试")


# ==================== 音频处理工具函数 ====================


def _check_ffmpeg() -> bool:
    """检查系统是否安装了ffmpeg。"""
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except Exception:
        return False


def convert_audio_for_asr(audio_path: str, target_format: str = "mp3", 
                          sample_rate: int = 16000, channels: int = 1) -> str:
    """通用的音频转换函数
    
    Args:
        audio_path: 输入音频路径
        target_format: 目标格式
        sample_rate: 采样率
        channels: 声道数
    """
    if not _check_ffmpeg():
        print("未检测到ffmpeg，使用原始文件")
        return audio_path
    
    try:
        import subprocess
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"asr_converted_{int(time.time())}_{os.path.basename(audio_path)}.{target_format}")
        
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-c:a', 'libmp3lame' if target_format == 'mp3' else 'pcm_s16le',
            '-b:a', '96k' if target_format == 'mp3' else '',
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-y', output_path
        ]
        
        # 移除空参数
        cmd = [arg for arg in cmd if arg]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return output_path
        else:
            print(f"音频转换失败: {result.stderr}")
            return audio_path
            
    except Exception as e:
        print(f"音频转换异常: {e}")
        return audio_path
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# 兼容其他OpenAI风格的大模型：支持自定义base_url与独立LLM密钥
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ASR提供者：默认openai；可选faster_whisper（本地离线）
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "openai").lower()
ASR_MODEL = os.getenv("ASR_MODEL", "whisper-1" if ASR_PROVIDER == "openai" else "small")
# 可选：提供本地模型目录（离线环境）。当设置该目录且存在时，优先使用本地模型，不再尝试在线下载。
ASR_LOCAL_DIR = os.getenv("ASR_LOCAL_DIR", "").strip()

# 分块配置：默认不分块。兼容分钟或秒两种配置名，避免改动已有示例。
_chunk_sec_env = os.getenv("ASR_CHUNK_SECONDS", "0")
_chunk_min_env = os.getenv("ASR_CHUNK_MIN", "0")
try:
    ASR_CHUNK_SECONDS = int(_chunk_sec_env or "0")
except Exception:
    ASR_CHUNK_SECONDS = 0
if not ASR_CHUNK_SECONDS:
    try:
        ASR_CHUNK_SECONDS = int(_chunk_min_env or "0") * 60
    except Exception:
        ASR_CHUNK_SECONDS = 0

# 取消分块处理：统一按完整音频进行识别

# 单独的客户端：ASR使用OpenAI Whisper；LLM可切换其他OpenAI兼容服务
asr_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and ASR_PROVIDER == "openai") else None
llm_client = None
if LLM_API_KEY or OPENAI_API_KEY:
    # 若设置了LLM_BASE_URL，则使用该地址；否则走默认OpenAI地址
    llm_client = OpenAI(api_key=(LLM_API_KEY or OPENAI_API_KEY), base_url=(LLM_BASE_URL or None))


def ensure_dirs():
    downloads = Path("downloads")
    downloads.mkdir(parents=True, exist_ok=True)


def _derive_title_from_path(path: str) -> str:
    """根据下载后的文件路径推测节目标题。
    约定下载模板为 "%(title)s-%(id)s.%(ext)s"，因此取去扩展名后最后一个连字符之前的部分作为标题。
    """
    try:
        base = os.path.basename(path)
        name_no_ext = os.path.splitext(base)[0]
        if "-" in name_no_ext:
            return name_no_ext.rsplit("-", 1)[0]
        return name_no_ext
    except Exception:
        return "未命名节目"


def download_media(public_url: str) -> str:
    """下载公开链接的音频/视频，并返回本地文件路径。
    优先选择最佳音频格式。不强制转码，减少对ffmpeg的依赖。
    """
    import yt_dlp

    ensure_dirs()
    outtmpl = "downloads/%(title)s-%(id)s.%(ext)s"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "proxy": "",  # 禁用代理，避免依赖外部环境
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(public_url, download=True)
        # 优先从requested_downloads拿到具体路径
        requested = info.get("requested_downloads")
        if requested and len(requested) > 0:
            return requested[0]["filepath"]
        # 回退到prepare_filename
        return ydl.prepare_filename(info)


def _check_ffmpeg() -> bool:
    """检查系统是否安装了ffmpeg。"""
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except Exception:
        return False


def chunk_audio(audio_path: str, chunk_seconds: int) -> list:
    """使用ffmpeg将音频按固定时长分块，返回分块文件路径列表。
    若ffmpeg不可用或分块失败，则返回空列表以回退到整文件识别。
    分块采用MP3(16kHz, 单声道, 96kbps)以兼容多数云ASR。
    """
    if not chunk_seconds or chunk_seconds <= 0:
        return []
    if not _check_ffmpeg():
        print("未检测到ffmpeg，跳过分块，按整文件处理。")
        return []
    try:
        import subprocess
        temp_dir = tempfile.mkdtemp(prefix="chunks_")
        out_tmpl = os.path.join(temp_dir, "chunk_%03d.mp3")
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-vn',
            '-c:a', 'libmp3lame',
            '-b:a', '96k',
            '-ar', '16000',
            '-ac', '1',
            '-f', 'segment',
            '-segment_time', str(int(chunk_seconds)),
            '-y', out_tmpl
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"分块失败，回退整文件识别：{result.stderr[:200]}")
            return []
        # 收集输出文件
        files = sorted([str(Path(temp_dir) / f) for f in os.listdir(temp_dir) if f.startswith('chunk_')])
        # 若仅得到一个分块，则视为未分块
        return files if len(files) >= 2 else []
    except Exception as e:
        print(f"分块异常：{e}")
        return []


def transcribe_with_aliyun_nls(audio_path):
    """使用阿里云NLS进行语音识别"""
    try:
        import json
        import subprocess
        import tempfile
        from aliyunsdkcore.client import AcsClient
        from aliyunsdkcore.acs_exception.exceptions import ClientException
        from aliyunsdkcore.acs_exception.exceptions import ServerException
        from aliyunsdkcore.request import CommonRequest
        
        # 获取阿里云配置
        access_key_id = os.getenv('ALIYUN_ACCESS_KEY_ID')
        access_key_secret = os.getenv('ALIYUN_ACCESS_KEY_SECRET')
        app_key = os.getenv('ALIYUN_NLS_APP_KEY')
        
        if not all([access_key_id, access_key_secret, app_key]):
            raise ValueError("阿里云NLS配置不完整，请检查ALIYUN_ACCESS_KEY_ID、ALIYUN_ACCESS_KEY_SECRET、ALIYUN_NLS_APP_KEY")
        
        # 创建客户端
        client = AcsClient(access_key_id, access_key_secret, 'cn-shanghai')
        
        # 转换音频格式为阿里云支持的格式（PCM, 16kHz, 16bit, 单声道）
        converted_audio_path = convert_audio_for_aliyun(audio_path)
        
        # 将音频上传到OSS并获取签名URL（阿里云要求可通过HTTP访问的URL）
        file_url = upload_to_oss(converted_audio_path)
        
        # 创建录音文件识别请求（使用CommonRequest）
        request = CommonRequest()
        request.set_domain('filetrans.cn-shanghai.aliyuncs.com')
        request.set_version('2018-08-17')
        request.set_product('nls-filetrans')
        request.set_action_name('SubmitTask')
        request.set_method('POST')
        
        # 构建任务参数（参考官方示例，必须提供可访问的file_link）
        task = {
            'appkey': app_key,
            'file_link': file_url,
            'version': '4.0',
            'enable_words': False
            # 如需智能分轨可开启：'auto_split': True
        }
        
        # 添加任务参数到请求体
        request.add_body_params('Task', json.dumps(task))
        
        # 发送请求
        response = client.do_action_with_exception(request)
        result = json.loads(response)
        
        # 检查响应状态
        if result.get('StatusText') == 'SUCCESS':
            task_id = result.get('TaskId')
            
            # 查询识别结果
            return get_transcription_result(client, task_id)
        else:
            # 打印详细的错误信息以便调试
            error_msg = f"阿里云NLS识别失败: {result.get('StatusText', '未知错误')}"
            if 'Message' in result:
                error_msg += f", 详细信息: {result['Message']}"
            raise Exception(error_msg)
            
    except ImportError:
        raise ImportError("请安装阿里云NLS SDK: pip install aliyun-python-sdk-core")
    except Exception as e:
        raise Exception(f"阿里云NLS识别错误: {str(e)}")


def get_transcription_result(client, task_id):
    """查询录音文件识别结果"""
    import time
    import json
    from aliyunsdkcore.request import CommonRequest
    
    # 创建查询请求
    request = CommonRequest()
    request.set_domain('filetrans.cn-shanghai.aliyuncs.com')
    request.set_version('2018-08-17')
    request.set_product('nls-filetrans')
    request.set_action_name('GetTaskResult')
    request.set_method('GET')
    request.add_query_param('TaskId', task_id)
    
    # 轮询查询结果（最多等待5分钟）
    for _ in range(30):
        try:
            response = client.do_action_with_exception(request)
            result = json.loads(response)
            
            status = result.get('StatusText')
            if status == 'SUCCESS':
                # 返回识别结果（可能是字符串或JSON结构）
                return result.get('Result', '')
            elif status in ['RUNNING', 'QUEUEING']:
                # 任务还在处理中，等待10秒后重试
                time.sleep(10)
            else:
                # 任务失败，提供详细错误信息
                error_msg = f"识别任务失败: {status}"
                if 'Message' in result:
                    error_msg += f", 详细信息: {result['Message']}"
                raise Exception(error_msg)
                
        except Exception as e:
            raise Exception(f"查询识别结果失败: {str(e)}")
    
    raise Exception("识别任务超时，请稍后重试")


def _normalize_aliyun_result_text(result):
    """将阿里云FileTrans的Result统一转换为纯文本字符串。
    兼容以下情况：
    - Result 是纯字符串（直接返回）；
    - Result 是JSON字符串（解析后按Sentences拼接Text）；
    - Result 是字典（包含Sentences或Text等字段）。
    """
    import json

    # 1) 如果是字符串，尝试当作JSON解析；解析失败则直接返回
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            result = parsed
        except Exception:
            return result

    # 2) 如果是字典，优先按Sentences拼接
    if isinstance(result, dict):
        sentences = result.get('Sentences')
        if isinstance(sentences, list):
            texts = []
            for s in sentences:
                if isinstance(s, dict):
                    t = s.get('Text') or s.get('text') or ''
                    if t:
                        texts.append(t)
            if texts:
                return "\n".join(texts)

        # 备选字段：直接取Text/Transcription/transcript
        for key in ('Text', 'Transcription', 'transcript'):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                return v

        # 如果仍然是字典但没有明确文本，尝试嵌套Result
        nested = result.get('Result')
        if isinstance(nested, (str, dict)):
            return _normalize_aliyun_result_text(nested)

        # 最后兜底：返回可读的JSON字符串
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)

    # 3) 其他类型，直接字符串化
    return str(result)


def convert_audio_for_aliyun(audio_path):
    """将音频转换为MP3（16kHz、单声道、低码率）以减小体积"""
    try:
        import subprocess
        import tempfile
        import os
        
        # 创建临时文件
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"aliyun_converted_{os.path.basename(audio_path)}.mp3")
        
        # 使用ffmpeg转换为MP3
        # 设置：16kHz采样率、单声道、约96kbps，兼顾识别效果与体积
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-c:a', 'libmp3lame',   # MP3编码
            '-b:a', '96k',          # 码率
            '-ar', '16000',         # 16kHz采样率
            '-ac', '1',             # 单声道
            '-y',                   # 覆盖输出文件
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            # 如果转换失败，尝试使用原始文件
            print(f"音频转换失败，使用原始文件: {result.stderr}")
            return audio_path
        
        return output_path
        
    except Exception as e:
        print(f"音频转换出错: {e}")
        # 出错时返回原始文件
        return audio_path


def upload_to_oss(local_path):
    """上传音频到阿里云OSS并返回可访问的签名URL。
    要求在.env中配置：
    - OSS_BUCKET：OSS存储空间名称
    - OSS_ENDPOINT：OSS地域Endpoint，例如：https://oss-cn-shanghai.aliyuncs.com
    使用账户的ALIYUN_ACCESS_KEY_ID与ALIYUN_ACCESS_KEY_SECRET鉴权。
    """
    try:
        import oss2
        import time
        import mimetypes

        bucket_name = os.getenv('OSS_BUCKET')
        endpoint = os.getenv('OSS_ENDPOINT')
        ak = os.getenv('ALIYUN_ACCESS_KEY_ID')
        sk = os.getenv('ALIYUN_ACCESS_KEY_SECRET')

        if not all([bucket_name, endpoint, ak, sk]):
            raise ValueError("OSS配置不完整：请在.env中设置OSS_BUCKET、OSS_ENDPOINT、ALIYUN_ACCESS_KEY_ID、ALIYUN_ACCESS_KEY_SECRET")

        auth = oss2.Auth(ak, sk)

        # 使用默认会话以确保与当前 oss2 版本兼容
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        object_name = f"nls/{int(time.time())}_{os.path.basename(local_path)}"
        content_type = mimetypes.guess_type(local_path)[0]
        if not content_type:
            ext = os.path.splitext(local_path)[1].lower()
            content_type = 'audio/mpeg' if ext == '.mp3' else 'audio/wav'
        
        print(f"正在上传 {local_path} 到 bucket: {bucket_name}, object: {object_name} ...")
        with open(local_path, 'rb') as f:
            bucket.put_object(object_name, f, headers={'Content-Type': content_type})
        print("上传成功。")

        # 为阿里云内网服务（如NLS）生成内网签名的URL，以提高稳定性和速度
        # 尝试从公共endpoint推断内网endpoint
        # 例如: https://oss-cn-beijing.aliyuncs.com -> https://oss-cn-beijing-internal.aliyuncs.com
        internal_endpoint = endpoint.replace('.aliyuncs.com', '-internal.aliyuncs.com')
        
        # 使用内网endpoint创建新的bucket对象用于签名
        print(f"使用内网Endpoint生成签名URL: {internal_endpoint}")
        internal_bucket = oss2.Bucket(auth, internal_endpoint, bucket_name)
        
        # 生成10分钟有效的签名URL
        signed_url = internal_bucket.sign_url('GET', object_name, 600)
        print(f"生成的内网签名URL: {signed_url}")
        
        return signed_url

    except ImportError:
        raise ImportError("缺少OSS依赖：请在requirements.txt中加入oss2，并pip安装")
    except Exception as e:
        # 打印更详细的异常信息
        import traceback
        print(f"OSS操作失败，错误详情: {traceback.format_exc()}")
        raise Exception(f"OSS上传或签名失败：{e}")

# ASR服务实例
_aliyun_service = AliyunNLSService()

# 修改transcribe_audio函数支持阿里云NLS
def _asr_openai(audio_path: str) -> str:
    """OpenAI Whisper 单文件识别。"""
    global asr_client
    with open(audio_path, "rb") as audio_file:
        transcript = asr_client.audio.transcriptions.create(
            model=ASR_MODEL,
            file=audio_file
        )
    return transcript.text


def _asr_faster_whisper(audio_path: str) -> str:
    """faster-whisper 单文件识别（本地/离线）。"""
    try:
        from faster_whisper import WhisperModel
        if not _check_ffmpeg():
            return "错误: 请先安装ffmpeg并确保在PATH中"
        model_path = ASR_LOCAL_DIR if (ASR_LOCAL_DIR and os.path.exists(ASR_LOCAL_DIR)) else ASR_MODEL
        model = WhisperModel(model_path, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, beam_size=5)
        return "".join([segment.text for segment in segments])
    except ImportError:
        return "错误：未安装 faster-whisper，请在虚拟环境运行：pip install faster-whisper"
    except Exception as e:
        msg = str(e)
        if "huggingface_hub" in msg or "snapshot_download" in msg or "LocalEntryNotFoundError" in msg:
            return "错误：无法下载ASR模型。解决方法：1) 保证网络可访问 HuggingFace；或 2) 先手动下载模型并在 .env 设置 ASR_LOCAL_DIR=本地模型目录（如 E:/models/faster-whisper-small）。"
        return f"错误：ASR执行失败：{e}"


def _asr_aliyun(audio_path: str) -> str:
    """阿里云 NLS 单文件识别，使用新的服务类。"""
    try:
        return _aliyun_service.transcribe(audio_path)
    except Exception as e:
        raise Exception(f"阿里云NLS识别失败: {str(e)}")


# ==================== ASR服务管理器 ====================
class ASRServiceManager:
    """ASR服务管理器，统一管理所有ASR服务"""
    
    def __init__(self):
        self._services = {}
        self._current_provider = ASR_PROVIDER
    
    def register_service(self, name: str, service: BaseASRService):
        """注册ASR服务"""
        self._services[name] = service
    
    def get_service(self, name: str) -> Optional[BaseASRService]:
        """获取指定名称的ASR服务"""
        return self._services.get(name)
    
    def get_current_service(self) -> Optional[BaseASRService]:
        """获取当前配置的ASR服务"""
        return self._services.get(self._current_provider)
    
    def list_services(self) -> List[str]:
        """列出所有已注册的ASR服务"""
        return list(self._services.keys())
    
    def get_all_services(self) -> Dict[str, BaseASRService]:
        """获取所有已注册的ASR服务"""
        return self._services.copy()
    
    def get_current_provider(self) -> str:
        """获取当前配置的ASR提供商"""
        return self._current_provider
    
    def set_provider(self, provider_name: str):
        """设置当前使用的ASR提供商"""
        if provider_name in self._services:
            self._current_provider = provider_name
            return True
        return False
    
    def get_service_info(self, name: str) -> Dict[str, Any]:
        """获取ASR服务信息"""
        service = self._services.get(name)
        if not service:
            return {}
        
        return {
            'name': service.name,
            'available': service.validate_config(),
            'supported_formats': service.get_supported_formats()
        }
    
    def get_all_services_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有ASR服务信息"""
        return {name: self.get_service_info(name) for name in self._services}


# 初始化ASR服务管理器
asr_manager = ASRServiceManager()

# 创建并注册所有ASR服务
# 阿里云NLS服务
_aliyun_service = AliyunNLSService()
asr_manager.register_service('aliyun_nls', _aliyun_service)

# 讯飞ASR服务
_iflytek_service = IflytekASRService()
asr_manager.register_service('iflytek_asr', _iflytek_service)

# 百度ASR服务
_baidu_service = BaiduASRService()
asr_manager.register_service('baidu_asr', _baidu_service)

# 腾讯云ASR服务
_tencent_service = TencentASRService()
asr_manager.register_service('tencent_asr', _tencent_service)

# 保持向后兼容的ASR适配器映射
ASR_ADAPTERS = {
    'openai': _asr_openai,
    'faster_whisper': _asr_faster_whisper,
    'aliyun': _asr_aliyun,
    'aliyun_nls': _asr_aliyun,  # 兼容别名
}


def transcribe_audio(audio_path: str):
    """语音转文字（支持可选分块）。
    返回 (raw_text, updates)。updates 为进度状态列表，便于SSE前端展示。
    """
    adapter = ASR_ADAPTERS.get(ASR_PROVIDER)
    if adapter is None:
        return f"错误: 不支持的ASR提供商: {ASR_PROVIDER}", []

    # 分块决策：默认不分块；若配置>0则按分块处理
    chunks = chunk_audio(audio_path, ASR_CHUNK_SECONDS)
    updates = []
    if not chunks:
        # 整文件识别
        updates.append({'status': '正在转写…', 'progress': 40})
        text = adapter(audio_path)
        updates.append({'status': '转写完成', 'progress': 85})
        return text, updates

    # 分块识别并拼接
    total = len(chunks)
    merged_texts = []
    # 记录临时分块目录以便清理
    tmp_dir = os.path.dirname(chunks[0]) if chunks else None
    try:
        for idx, part in enumerate(chunks, start=1):
            pct = 20 + int(60 * idx / max(total, 1))
            updates.append({'status': f'正在转写分块 {idx}/{total}…', 'progress': pct})
            t = adapter(part)
            # 对阿里云结果做归一化
            merged_texts.append(t if isinstance(t, str) else _normalize_aliyun_result_text(t))
        updates.append({'status': '分块转写完成，正在合并…', 'progress': 85})
        return "\n".join(merged_texts), updates
    finally:
        # 清理临时分块文件夹
        try:
            if tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def improve_text(raw_text: str) -> str:
    """调用LLM对文本进行分段与优化，更便于阅读。"""
    if not llm_client:
        raise RuntimeError("缺少LLM密钥：请在.env中设置LLM_API_KEY或OPENAI_API_KEY")

    # 保证传入的原始文本为字符串（避免出现 dict 拼接错误）
    safe_text = raw_text if isinstance(raw_text, str) else _normalize_aliyun_result_text(raw_text)

    prompt = (
        "你是一名中文文字编辑助手。请将下面的口语化文稿进行整理：\n"
        "- 保留原意，去除口头语与赘词。\n"
        "- 自动断句与分段，增加标点。\n"
        "- 如果出现明显的专有名词或地名，保持其原样。\n"
        "- 不编造内容，不添加主观评论。\n\n"
        "待优化文稿：\n" + safe_text
    )

    completion = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是一位严谨的文本整理助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content.strip()


@app.route("/")
def index():
    response = make_response(render_template("index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/transcribe", methods=["GET"])
def api_transcribe():
    # 对于 EventSource (GET请求)，参数从 request.args 获取
    public_url = request.args.get("url", "").strip()
    if not public_url:
        return Response(stream_with_context(stream_error("请输入公开的音频或视频链接")), status=400, content_type='text/event-stream')

    # 对特定站点（如小宇宙）做友好提示：当前下载工具不支持其链接
    try:
        from urllib.parse import urlparse
        domain = urlparse(public_url).netloc.lower()
        if "xiaoyuzhoufm.com" in domain:
            return Response(
                stream_with_context(stream_error("暂不支持通过链接下载小宇宙音频，请改用页面中的“上传文件”方式。")),
                status=400,
                content_type='text/event-stream'
            )
    except Exception:
        # 域名解析异常时忽略，继续走常规流程
        pass

    def generate():
        import json
        start_ts = time.time()
        try:
            # 步骤1：下载
            yield f"data: {json.dumps({'status': '正在下载音频…', 'progress': 5}, ensure_ascii=False)}\n\n"
            file_path = download_media(public_url)

            # 步骤2：ASR转文字
            yield f"data: {json.dumps({'status': '下载完成，正在转写…', 'progress': 20}, ensure_ascii=False)}\n\n"
            raw_text, updates = transcribe_audio(file_path)
            # 推送分块进度
            for u in updates:
                try:
                    yield f"data: {json.dumps(u, ensure_ascii=False)}\n\n"
                except Exception:
                    pass

            # 步骤3：LLM分段与优化（符合规范：输出原文案）
            status_msg = json.dumps({'status': '正在识别文案…', 'progress': 95}, ensure_ascii=False)
            yield f"data: {status_msg}\n\n"
            optimized_content = improve_text(raw_text)

            # 用完后尽量清理下载文件（节省空间）
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass

            # 最终结果（符合规范：只输出原文案）
            result = {
                "ok": True,
                "elapsed_sec": round(time.time() - start_ts, 2),
                "content": optimized_content,  # 原文案（第一次输出）
                "title": _derive_title_from_path(file_path),
                "progress": 100,
            }
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

        except Exception as e:
            err = f"处理失败：{e}"
            traceback.print_exc()
            yield f"data: {json.dumps({'ok': False, 'error': err}, ensure_ascii=False)}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')


# 本地文件上传识别（选择已有音频/视频文件）
@app.route("/api/transcribe_file", methods=["POST"])
def api_transcribe_file():
    start_ts = time.time()
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"ok": False, "error": "请选择要上传的文件"}), 400

        ensure_dirs()
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename or f"upload_{int(time.time())}.m4a")
        save_path = Path("downloads") / filename
        file.save(save_path)

        raw_text = transcribe_audio(str(save_path))
        # LLM优化与分词（原文案）
        optimized_content = improve_text(raw_text)

        try:
            if os.path.exists(save_path):
                os.remove(save_path)
        except Exception:
            pass

        # 返回结果（符合规范：只输出原文案）
        return jsonify({
            "ok": True,
            "elapsed_sec": round(time.time() - start_ts, 2),
            "content": optimized_content,  # 原文案（第一次输出）
            "title": os.path.splitext(filename)[0],
        })
    except Exception as e:
        err = f"识别失败：{e}"
        traceback.print_exc()
        return jsonify({"ok": False, "error": err}), 500


def stream_error(message):
    import json
    yield f"data: {json.dumps({'ok': False, 'error': message}, ensure_ascii=False)}\n\n"


# AI文本优化API
@app.route("/api/optimize_text", methods=["POST"])
def optimize_text():
    """使用AI模型优化文本内容"""
    try:
        data = request.get_json()
        if not data or not data.get("text") or not data.get("text").strip():
            return jsonify({"success": False, "error": "文本内容不能为空"})
        
        text = data["text"]
        # 使用统一的LLM配置
        if not llm_client:
            return jsonify({"success": False, "error": "缺少LLM密钥：请在.env中设置"})
        
        prompt = f"""
请将以下播客转录文本进行优化，要求：
1. 修正语法错误和口误
2. 提高文本的可读性和流畅性
3. 保持原意不变
4. 添加适当的标点符号
5. 统一用词风格

原始文本：
{text}

优化后的文本：
"""
        
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的文本编辑助手，擅长优化播客转录文本。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        optimized_text = response.choices[0].message.content.strip()
        
        return jsonify({
            "success": True,
            "optimized_text": optimized_text,
            "model_used": LLM_MODEL
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"文本优化失败: {str(e)}"})

# 大纲生成API
@app.route("/api/generate_outline", methods=["POST"])
def generate_outline():
    """基于优化后的文本生成大纲"""
    try:
        data = request.get_json()
        if not data or not data.get("text") or not data.get("text").strip():
            return jsonify({"success": False, "error": "文本内容不能为空"})
        
        text = data["text"]
        
        # 使用统一的LLM配置
        if not llm_client:
            return jsonify({"success": False, "error": "缺少LLM密钥：请在.env中设置LLM_API_KEY或OPENAI_API_KEY"})
        
        prompt = f"""
请为以下文本生成一个结构化的大纲，要求：
1. 提取主要话题和关键点
2. 按照逻辑顺序组织内容
3. 使用清晰的层级结构
4. 包含重要的时间点或数据
5. 大纲应该简洁明了，便于快速理解

文本内容：
{text}

请以以下格式返回大纲：
"""
        
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的知识整理助手，擅长从长文本中提取结构化大纲。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        outline = response.choices[0].message.content.strip()
        
        return jsonify({
            "success": True,
            "outline": outline
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"大纲生成失败: {str(e)}"})

if __name__ == '__main__':
    PORT = os.getenv("PORT", "8000")
    print(f"🚀 启动服务器在端口 {PORT}")
    app.run(host="0.0.0.0", port=int(PORT), debug=True)