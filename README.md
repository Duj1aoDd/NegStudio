# NegStudio
NegStudio是一个用于底片去色罩的库

NegStudio is a library for negative film decolourisation masks

## 安装步骤
### 环境要求
- Python >= 3.8

### 创建虚拟环境
建议使用 Python 自带的 `venv` 来创建虚拟环境。
```bash
# 创建名为 myenv 的虚拟环境
python -m venv venv

# 在 Windows 上激活虚拟环境
venv\Scripts\activate

# 在 Linux 或 macOS 上激活虚拟环境
source venv/bin/activate
```
### 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法
创建NEG实例
```python
Neg = NEG("Image path")
```
检测底片边缘
```python
neg.FindBroaders()
```
```python
neg.broaderlinesDebug()#边线检测结果
neg.broaderDebug()#边框检测结果
```


