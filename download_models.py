import os
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

class ModelDownloader:
    """模型下载器"""
    
    def __init__(self):
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
    
    def download_qwen_coder(self, model_size: str = "1.5B"):
        """下载Qwen Coder模型"""
        print(f"下载Qwen2.5-Coder-{model_size}...")
        
        model_map = {
            "1.5B": "Qwen/Qwen2.5-Coder-1.5B",
            "7B": "Qwen/Qwen2.5-Coder-7B", 
            "14B": "Qwen/Qwen2.5-Coder-14B"
        }
        
        if model_size not in model_map:
            print(f"❌ 不支持的模型大小: {model_size}")
            return
            
        model_name = model_map[model_size]
        local_path = self.model_dir / f"Qwen2.5-Coder-{model_size}"
        
        try:
            # 下载模型
            snapshot_download(
                repo_id=model_name,
                local_dir=local_path,
                local_dir_use_symlinks=False
            )
            print(f"✅ Qwen2.5-Coder-{model_size} 下载完成")
            
            # 测试加载
            tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            print(f"✅ 模型加载测试通过")
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
    
    def download_synthesis_model(self):
        """下载合成模型"""
        print("下载Qwen2.5-7B-Instruct...")
        
        local_path = self.model_dir / "Qwen2.5-7B-Instruct"
        
        try:
            snapshot_download(
                repo_id="Qwen/Qwen2.5-7B-Instruct",
                local_dir=local_path, 
                local_dir_use_symlinks=False
            )
            print("✅ Qwen2.5-7B-Instruct 下载完成")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
    
    def download_embedding_model(self):
        """下载嵌入模型"""
        print("下载gte-large-en-v1.5...")
        
        try:
            model = SentenceTransformer('thenlper/gte-large-en-v1.5')
            local_path = self.model_dir / "gte-large-en-v1.5"
            model.save(str(local_path))
            print("✅ gte-large-en-v1.5 下载完成")
        except Exception as e:
            print(f"❌ 下载失败: {e}")

def main():
    downloader = ModelDownloader()
    
    print("开始下载模型...")
    print("=" * 50)
    
    # 下载基础模型
    downloader.download_qwen_coder("1.5B")
    downloader.download_qwen_coder("7B")
    
    # 下载合成模型
    downloader.download_synthesis_model()
    
    # 下载嵌入模型
    downloader.download_embedding_model()
    
    print("=" * 50)
    print("模型下载完成！")

if __name__ == "__main__":
    main()