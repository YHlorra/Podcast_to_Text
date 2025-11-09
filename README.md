# 🎙️ Podcast-to-Text

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Flask](https://img.shields.io/badge/Flask-3.1+-orange.svg)](https://flask.palletsprojects.com/)

一个简单易用的音频转文字工具，支持多种语音识别方案：

🎯 **核心功能**
- 🎵 自动下载公开链接中的音频/视频（支持YouTube、B站等）
- 📝 智能语音识别（ASR），支持多种方案
- ✨ AI文本优化，自动分段和提升可读性
- 🌐 简洁的Web界面，无需复杂配置

⚡ **支持的ASR方案**
- **OpenAI Whisper** - 云端识别，准确率高
- **国产云ASR** - 国内优化，价格实惠
- **本地模型** - 离线使用，隐私安全

## 🚀 快速开始

### 环境要求
- Python 3.9 或更高版本
- 至少 4GB 可用内存（本地ASR需要更多）
- 稳定的网络连接（使用云服务时）

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/your-username/podcast-to-text.git
   cd podcast-to-text
   ```

2. **创建虚拟环境并安装依赖**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **配置环境变量**
   ```bash
   cp .env.example
   # 编辑 .env 文件，填入必要的API密钥
   ```

4. **启动服务**
   ```bash
   python server.py
   ```

5. **访问应用**
   打开浏览器访问 http://127.0.0.1:8000

## 🎯 ASR方案对比与选择
### 1. OpenAI Whisper
- **准确率**: 英文>95%，中文>90%
- **处理速度**: 1小时音频约3-5分钟
- **优点**: 准确率高，支持多语言，无需本地硬件
- **缺点**: 需要OpenAI API密钥，网络要求稳定
- **成本**: 约$0.006/分钟（1小时约$0.36）
### 2.国产云ASR服务

国内云服务商提供的中文语音识别服务，适合中文场景：
**支持的服务商**（需自行注册和配置）：
- 阿里云智能语音交互
- 腾讯云语音识别
- 百度智能云语音
- 讯飞开放平台

**配置方法**：
- 注册对应服务商账号并开通语音识别服务
- 获取API密钥和相关配置信息
- 在`.env`文件中设置对应的环境变量

**注意事项**：
- 各服务商的SDK和鉴权方式不同
- 价格和免费额度各有差异
- 中文识别准确率普遍较好

> 提示：默认配置不变。只有当你设置了 `ASR_PROVIDER=faster_whisper` 或 `ASR_PROVIDER=aliyun_nls` 时，才会走对应ASR逻辑。

### 3. 本地模型（faster-whisper）
**⚠️ 重要提醒**: 本地ASR需要较强的硬件配置

| 模型大小 | 内存需求 | 显存需求 | 准确率 | 1小时处理时间 |
|----------|----------|----------|--------|---------------|
| tiny | 1GB | 无 | 70% | 5分钟 |
| base | 2GB | 无 | 78% | 8分钟 |
| small | 4GB | 无 | 85% | 15分钟 |
| medium | 8GB | 4GB | 90% | 30分钟 |
| large | 16GB | 8GB | 95% | 60分钟 |
**本地模型缺点**:
- 首次下载模型文件较大（small模型约500MB）
- 处理速度慢，特别是大文件
- 需要较好的CPU/内存配置
- 准确率通常低于云服务

## LLM文档转写

本项目默认使用 OpenAI。若在国内访问受限，可改用“OpenAI兼容”的服务，只需在 `.env` 设置：

```
LLM_API_KEY=你的其他服务密钥
LLM_BASE_URL=该服务的OpenAI兼容API地址
LLM_MODEL=该服务对应的模型名称
```

示例（任选其一，按服务商文档填写）：
- DeepSeek：
  - `LLM_BASE_URL=https://api.deepseek.com`
  - `LLM_MODEL=deepseek-chat`
- OpenRouter：
  - `LLM_BASE_URL=https://openrouter.ai/api/v1`
  - `LLM_MODEL=openrouter/auto`
- 硅基流动（SiliconFlow）：
  - `LLM_BASE_URL=https://api.siliconflow.cn/v1`
  - `LLM_MODEL=deepseek-ai/DeepSeek-V3`
- 本地 Ollama：
  - `LLM_BASE_URL=http://127.0.0.1:11434/v1`
  - `LLM_MODEL=llama3.1:8b`


## ⚙️ 配置选项

### 环境变量详解

| 变量名 | 说明 | 默认值 | 是否必需 |
|--------|------|--------|----------|
| `OPENAI_API_KEY` | OpenAI API密钥 | 空 | 使用OpenAI Whisper时必需 |
| `LLM_API_KEY` | 文本优化LLM密钥 | 空 | 可选，可与OPENAI_API_KEY共用 |
| `LLM_BASE_URL` | OpenAI兼容API地址 | 空 | 使用第三方LLM服务时设置 |
| `LLM_MODEL` | LLM模型名称 | gpt-4o-mini | 可选 |
| `ASR_PROVIDER` | ASR提供商选择 | openai | 可选 |
| `ASR_MODEL` | ASR模型选择 | whisper-1 | 可选 |

### 配置示例

#### 方案1：使用OpenAI Whisper
```bash
# .env 文件内容
OPENAI_API_KEY=sk-your-openai-key-here
ASR_PROVIDER=openai
ASR_MODEL=whisper-1
```

#### 方案2：使用国产云ASR服务
```bash
# .env 文件内容
# 以阿里云NLS为例，其他服务商请参考各自文档
ASR_PROVIDER=aliyun_nls
ALIYUN_ACCESS_KEY_ID=your-access-key
ALIYUN_ACCESS_KEY_SECRET=your-secret-key
ALIYUN_NLS_APP_KEY=your-app-key
OSS_BUCKET=your-oss-bucket
OSS_ENDPOINT=https://oss-cn-shanghai.aliyuncs.com
```

#### 方案3：使用本地模型
```bash
# .env 文件内容
ASR_PROVIDER=faster_whisper
ASR_MODEL=small
# 可选：指定本地模型目录
ASR_LOCAL_DIR=C:\models\faster-whisper-small
```

## 🔧 常见问题

### Q: 没有API密钥怎么办？
A: 你可以：
1. 注册OpenAI账号获取API密钥（推荐）
2. 使用国产云ASR服务（适合中文场景）
3. 使用本地模型（无需密钥，但需要较好硬件）

### Q: 下载失败怎么办？
A: 
- 检查网络连接是否稳定
- 确认链接是否为公开内容
- 尝试其他平台链接
- 检查是否被防火墙阻止

### Q: 本地模型运行缓慢怎么办？
A:
- 使用更小的模型（tiny/base）
- 升级硬件配置（CPU/内存）
- 考虑使用云服务（OpenAI/国产云ASR）
- 减少同时处理的任务数量

### Q: 识别准确率低怎么办？
A:
- 检查音频质量是否清晰
- 尝试使用更大的模型
- 考虑使用云服务获得更高准确率
- 确保音频语言与模型匹配

### Q: 如何处理长音频文件？
A:
- 设置音频分块参数：`ASR_CHUNK_MIN=10`（每10分钟切分）
- 确保有足够内存
- 考虑使用云服务处理大文件

## 📚 技术原理

### 工作流程
1. **音频下载**: 使用yt-dlp从公开平台下载音频
2. **语音识别**: 将音频转换为原始文本（ASR）
## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 开发环境搭建
```bash
git clone https://github.com/your-username/podcast-to-text.git
cd podcast-to-text
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 提交规范
- 使用清晰的提交信息
- 添加必要的测试
- 更新相关文档

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 优秀的语音识别模型
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - 强大的媒体下载工具
- [Flask](https://flask.palletsprojects.com/) - 轻量级Web框架

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 提交GitHub Issue
---

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！**