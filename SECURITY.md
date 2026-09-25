# 安全与数据泄露报告

## 报告渠道

如果你在本仓库（含 git 历史、Issue、PR）中发现以下内容，请**不要**公开提 Issue 或在评论里引用原文：

- API key、token、私钥等凭证
- 疑似真实的客户数据、机构内部表结构或字段清单
- 可用于识别具体客户、机构或人员的信息

请通过 GitHub 的 [私密漏洞报告](https://github.com/JialuXu/corporate-ewi-derivation/security/advisories/new)（Security → Report a vulnerability）联系维护者，说明文件路径或提交号即可，无需附上敏感内容本身。

## 维护者的处理流程

1. 确认后立即轮换泄露的凭证。
2. 删除相关内容；若已进入 git 历史，改写历史并请求 GitHub 清除缓存。
3. 在修复完成后告知报告者。

## 本仓库的数据边界

- 仓库只包含合成数据（`synthetic.py` 生成）和用公开通用术语编写的示例指标清单。
- 真实的原子指标清单在运行时通过 `AD_REGISTRY_CSV` 加载，永远不入库。
- 每次提交与 CI 都会运行 `scripts/check_public_safety.py` 检查私有清单、凭证和内部文档是否被误提交。
