# 🧠 智能模型降级系统

## 🎯 功能概述

RAG Chat App 现在具备智能模型降级功能，根据以下条件自动选择最合适的模型：

1. **OpenAI API 余额**
2. **Token 配额限制**  
3. **API 可用性**
4. **请求大小**

## 🏆 模型层级 (按优先级排序)

| 优先级 | 模型 | 用途 | 成本/1K tokens | 最大tokens |
|--------|------|------|---------------|------------|
| 1️⃣ | **GPT-4o-mini** | 最佳质量 | $0.015 | 128K |
| 2️⃣ | **GPT-4 Turbo** | 高质量 | $0.01 | 128K |
| 3️⃣ | **GPT-3.5 Turbo** | 经济实惠 | $0.002 | 16K |
| 4️⃣ | **Local Llama2** | 完全免费 | $0.00 | 4K |

## ⚙️ 自动降级规则

### 📊 余额检查
- **余额 > $5**: 使用 GPT-4o-mini
- **余额 $1-$5**: 自动降级到 GPT-3.5 
- **余额 < $1**: 切换到本地模型

### 🔄 故障转移
1. **API 错误**: 自动尝试下一级模型
2. **网络问题**: 切换到本地模型
3. **请求过大**: 使用支持更大 context 的模型

### 💰 成本控制
- **预估成本 > 余额**: 自动选择更便宜的模型
- **每日限额**: 防止超出预算

## 📋 配置说明

### 1. 环境变量设置

```bash
# 复制配置文件
cp .env.example .env

# 编辑配置
nano .env
```

**关键配置项:**
```bash
# OpenAI 设置
OPENAI_API_KEY=your_key_here
OPENAI_ESTIMATED_BALANCE=25.50    # 当前余额

# 降级阈值
MIN_OPENAI_BALANCE=5.00           # 最低余额阈值
MAX_MODEL_RETRIES=2               # 失败重试次数

# 本地模型 (可选)
LOCAL_MODEL_ENDPOINT=http://localhost:11434/api/generate
LOCAL_MODEL_NAME=llama2:7b-chat
```

### 2. 安装本地模型 (备用方案)

```bash
# 运行安装脚本
./install_local_model.sh

# 手动安装 (macOS)
brew install ollama
ollama serve
ollama pull llama2:7b-chat
```

## 🚀 使用方法

### 启动应用
```bash
# 测试模式 (mock 后端)
./start.sh

# 完整 RAG 模式 (智能降级)
./start.sh full
```

### 监控模型状态
访问: `http://localhost:8000/api/model_status`

**响应示例:**
```json
{
  "current_model": {
    "tier": "gpt-4o-mini",
    "model_name": "gpt-4o-mini", 
    "max_tokens": 128000,
    "cost_per_1k": 0.015,
    "is_local": false,
    "estimated_balance": 25.50
  },
  "model_availability": {
    "gpt-4o-mini": {"available": true},
    "gpt-3.5-turbo": {"available": true}, 
    "local-llama": {"available": false}
  }
}
```

## 💡 智能特性

### 🎯 上下文感知
- **短对话**: 使用高质量模型 (GPT-4)
- **长文档**: 自动选择大 context 模型
- **代码请求**: 优先选择 code-optimized 模型

### 📈 学习优化
- **记录失败模式**: 避免重复调用失败的模型
- **成本追踪**: 优化 token 使用效率
- **性能监控**: 选择响应最快的可用模型

### 🔒 安全保护
- **余额保护**: 防止意外超支
- **速率限制**: 避免 API 限流
- **数据隐私**: 本地模型完全离线

## 📊 前端显示

用户界面会显示当前使用的模型信息：

```
🧠 GPT-4o-mini: 这是一个详细的回答...

*[Used: gpt-4o-mini - Selected GPT-4o-mini]*
```

## 🛠️ 故障排除

### 问题: 一直使用本地模型
**解决方案:**
1. 检查 `.env` 中的 `OPENAI_API_KEY`
2. 验证 `OPENAI_ESTIMATED_BALANCE` 设置
3. 检查网络连接

### 问题: 本地模型不可用
**解决方案:**
```bash
# 检查 Ollama 服务
curl http://localhost:11434/api/tags

# 重启 Ollama
ollama serve

# 重新下载模型
ollama pull llama2:7b-chat
```

### 问题: 频繁模型切换
**解决方案:**
1. 增加 `MIN_OPENAI_BALANCE` 阈值
2. 检查 API 配额限制
3. 优化请求 token 数量

## 🔧 高级配置

### 自定义降级策略
编辑 `backend/model_fallback.py`:

```python
# 修改模型优先级
MODEL_TIERS = [ModelTier.GPT35, ModelTier.GPT4, ModelTier.LOCAL]

# 调整成本阈值  
COST_THRESHOLDS = {
    ModelTier.GPT4: 0.50,      # $0.50 per request
    ModelTier.GPT35: 0.10,     # $0.10 per request  
    ModelTier.LOCAL: 0.0       # Free
}
```

### 添加新模型
```python
# 在 MODEL_CONFIGS 中添加
ModelTier.CLAUDE = ModelConfig(
    name="claude-3-sonnet",
    max_tokens=200000,
    cost_per_1k_tokens=0.003,
    requires_api=True
)
```

## 📈 监控和分析

### 查看使用统计
```bash
# 检查模型使用情况
tail -f backend.log | grep "Switching from"

# 监控成本
grep "Estimated cost" backend.log
```

### API 调用分析
访问 `/api/model_status` 获取实时统计:
- 当前模型状态
- 失败次数统计
- 可用性检查结果
- 预估余额信息

## 🎯 最佳实践

1. **设置合理阈值**: 根据使用频率调整 `MIN_OPENAI_BALANCE`
2. **监控成本**: 定期检查 API 使用情况
3. **本地备用**: 安装 Ollama 作为完全免费的备选方案
4. **测试环境**: 在测试时使用 mock 模式避免消耗 token
5. **批量处理**: 合并多个小请求减少 API 调用

## 🚨 注意事项

- ⚠️ 本地模型质量可能不如 GPT-4
- ⚠️ 模型切换可能影响回复一致性  
- ⚠️ 余额检查基于估算，非实时
- ⚠️ 首次使用本地模型需下载约 4GB 数据

通过这个智能降级系统，你的 RAG Chat App 将永远不会因为 token 不足而停止工作！🎉