import os
import sys
from huggingface_hub import login, HfFolder

def main():
    """Hugging Face 登录助手"""
    print("Hugging Face 登录助手")
    print("=" * 40)
    
    # 检查是否已登录
    token = HfFolder.get_token()
    if token:
        print("✓ 检测到已登录 Hugging Face")
        print(f"Token 前10位: {token[:10]}...")
        
        choice = input("是否重新登录？(y/N): ").strip().lower()
        if choice != 'y':
            print("保持当前登录状态")
            return
    
    # 获取 token
    print("\n请提供 Hugging Face 访问 token:")
    print("1. 访问 https://huggingface.co/settings/tokens 获取 token")
    print("2. 确保 token 有读取权限")
    
    token = input("请输入 token: ").strip()
    
    if not token:
        print("未提供 token，退出")
        return
    
    # 尝试登录
    try:
        login(token=token)
        print("✓ 登录成功！")
        
        # 验证 token 是否保存
        saved_token = HfFolder.get_token()
        if saved_token:
            print("✓ Token 已保存到本地缓存")
        else:
            print("✗ Token 保存失败")
            
    except Exception as e:
        print(f"✗ 登录失败: {e}")
        print("请检查 token 是否正确且有访问数据集的权限")

if __name__ == "__main__":
    main()