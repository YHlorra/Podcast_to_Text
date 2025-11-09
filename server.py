import os
import time
import traceback
from pathlib import Path
import shutil

from flask import Flask, request, jsonify, render_template, Response, stream_with_context, make_response
from dotenv import load_dotenv
from openai import OpenAI

app = Flask(__name__, template_folder="templates", static_folder="static")

# 读取环境变量
load_dotenv()
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

# 修改transcribe_audio函数支持阿里云NLS
def transcribe_audio(audio_path):
    """语音转文字"""
    global asr_client
    
    if ASR_PROVIDER == 'openai':
        # 原有的OpenAI Whisper代码
        with open(audio_path, "rb") as audio_file:
            transcript = asr_client.audio.transcriptions.create(
                model=ASR_MODEL,
                file=audio_file
            )
        return transcript.text
    
    elif ASR_PROVIDER == 'faster_whisper':
        # 原有的faster-whisper代码
        try:
            from faster_whisper import WhisperModel
            
            # 检查ffmpeg
            try:
                import subprocess
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return "错误: 请先安装ffmpeg并确保在PATH中"
            
            # 如果设置了本地模型目录且存在，则使用本地模型路径（离线使用）
            model_path = ASR_LOCAL_DIR if (ASR_LOCAL_DIR and os.path.exists(ASR_LOCAL_DIR)) else ASR_MODEL
            model = WhisperModel(model_path, device="cpu", compute_type="int8")

            # 直接处理完整音频
            segments, info = model.transcribe(audio_path, beam_size=5)
            text = "".join([segment.text for segment in segments])
            return text
            
        except ImportError:
            return "错误：未安装 faster-whisper，请在虚拟环境运行：pip install faster-whisper"
        except Exception as e:
            # 针对首次下载模型、无法访问 HuggingFace 的常见问题给出友好提示
            msg = str(e)
            if "huggingface_hub" in msg or "snapshot_download" in msg or "LocalEntryNotFoundError" in msg:
                return "错误：无法下载ASR模型。解决方法：1) 保证网络可访问 HuggingFace；或 2) 先手动下载模型并在 .env 设置 ASR_LOCAL_DIR=本地模型目录（如 E:/models/faster-whisper-small）。"
            return f"错误：ASR执行失败：{e}"
    
    elif ASR_PROVIDER == 'aliyun_nls':
        # 新增阿里云NLS支持
        return transcribe_with_aliyun_nls(audio_path)
    
    else:
        return f"错误: 不支持的ASR提供商: {ASR_PROVIDER}"


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
            yield f"data: {json.dumps({'status': '正在下载音频…'}, ensure_ascii=False)}\n\n"
            file_path = download_media(public_url)

            # 步骤2：ASR转文字
            yield f"data: {json.dumps({'status': '下载完成，正在转写…'}, ensure_ascii=False)}\n\n"
            raw_text = transcribe_audio(file_path)

            # 步骤3：LLM分段与优化
            yield f"data: {json.dumps({'status': '转写完成，正在优化文本…'}, ensure_ascii=False)}\n\n"
            clean_text = improve_text(raw_text)

            # 用完后尽量清理下载文件（节省空间）
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass

            # 最终结果
            result = {
                "ok": True,
                "elapsed_sec": round(time.time() - start_ts, 2),
                "raw_text": raw_text,
                "clean_text": clean_text,
                "title": _derive_title_from_path(file_path),
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
        clean_text = improve_text(raw_text)

        try:
            if os.path.exists(save_path):
                os.remove(save_path)
        except Exception:
            pass

        return jsonify({
            "ok": True,
            "elapsed_sec": round(time.time() - start_ts, 2),
            "raw_text": raw_text,
            "clean_text": clean_text,
            "title": os.path.splitext(filename)[0],
        })
    except Exception as e:
        err = f"识别失败：{e}"
        traceback.print_exc()
        return jsonify({"ok": False, "error": err}), 500


def stream_error(message):
    import json
    yield f"data: {json.dumps({'ok': False, 'error': message}, ensure_ascii=False)}\n\n"


if __name__ == "__main__":
    print("Server running at http://127.0.0.1:8000/")
    app.run(host="127.0.0.1", port=8000, debug=True)