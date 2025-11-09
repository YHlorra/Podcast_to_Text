#!/usr/bin/env python3
"""
FFmpeg安装助手
自动检测操作系统并提供FFmpeg安装指导
"""

import os
import sys
import platform
import subprocess
import webbrowser

def detect_os():
    """检测操作系统类型"""
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Darwin":
        return "macos"
    elif system == "Linux":
        return "linux"
    else:
        return "unknown"

def check_ffmpeg():
    """检查FFmpeg是否已安装"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # 提取版本信息
            version_line = result.stdout.split('\n')[0]
            return True, version_line
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, None

def install_windows():
    """Windows系统安装指导"""
    print("""
🔧 Windows系统FFmpeg安装步骤：

1. 访问官方网站: https://www.gyan.dev/ffmpeg/builds/
2. 下载release版本（推荐full版本）
3. 解压到C:\ffmpeg目录
4. 添加环境变量：
   - 右键"此电脑" → 属性 → 高级系统设置
   - 环境变量 → 系统变量 → Path
   - 添加: C:\ffmpeg\bin
5. 重新打开命令行窗口

或者使用包管理器安装：
- 安装Chocolatey: https://chocolatey.org/
- 运行: choco install ffmpeg

💡 安装完成后，重新运行此脚本验证
""")
    
    # 提供直接下载链接
    webbrowser.open("https://www.gyan.dev/ffmpeg/builds/")

def install_macos():
    """macOS系统安装指导"""
    print("""
🔧 macOS系统FFmpeg安装步骤：

方案1: 使用Homebrew（推荐）
1. 安装Homebrew: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
2. 安装FFmpeg: brew install ffmpeg
3. 验证安装: ffmpeg -version

方案2: 使用MacPorts
1. 安装MacPorts: https://www.macports.org/install.php
2. 安装FFmpeg: sudo port install ffmpeg +nonfree

💡 安装完成后，重新运行此脚本验证
""")

def install_linux():
    """Linux系统安装指导"""
    print("""
🔧 Linux系统FFmpeg安装步骤：

Ubuntu/Debian:
sudo apt update
sudo apt install ffmpeg

CentOS/RHEL/Fedora:
sudo yum install epel-release
sudo yum install ffmpeg

Arch Linux:
sudo pacman -S ffmpeg

openSUSE:
sudo zypper install ffmpeg

💡 安装完成后，重新运行此脚本验证
""")

def main():
    """主函数"""
    print("🎬 FFmpeg安装助手")
    print("=" * 50)
    
    # 检测操作系统
    os_type = detect_os()
    print(f"检测到操作系统: {platform.system()} {platform.release()}")
    
    # 检查FFmpeg是否已安装
    print("\n🔍 检查FFmpeg安装状态...")
    is_installed, version = check_ffmpeg()
    
    if is_installed:
        print(f"✅ FFmpeg已安装!")
        print(f"📋 {version}")
        print("\n🎉 您的系统已准备好运行播客转文字工具！")
        return
    else:
        print("❌ FFmpeg未找到")
    
    # 根据操作系统提供安装指导
    print(f"\n📥 正在为{platform.system()}系统提供安装指导...")
    print("-" * 50)
    
    if os_type == "windows":
        install_windows()
    elif os_type == "macos":
        install_macos()
    elif os_type == "linux":
        install_linux()
    else:
        print("❌ 不支持的操作系统类型")
        print("请手动访问: https://ffmpeg.org/download.html")
        return
    
    print("\n⚠️  重要提醒:")
    print("- 安装完成后，请重新打开命令行窗口")
    print("- 重新运行此脚本验证安装")
    print("- 确保ffmpeg命令可在任何目录下运行")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1)