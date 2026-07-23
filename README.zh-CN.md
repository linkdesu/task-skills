# task-skills

本仓库是一个面向 `npx skills add` 的多 skill 目录。

这里的许多 skill 来自复杂、一次性且高度依赖环境的任务。过去，这类经验通常以独立 shell 或 Python 脚本的形式分享，帮助了很多人，但也会占用热心开发者大量时间。本仓库延续这种分享精神，但采用 AI skill 作为载体。

在个人设备都能轻易达到今天 SOTA 模型能力之前，分享可复用 skill 仍是最高效的路径。长期目标很明确：把实践经验沉淀为可复用的 skill，让更多用户可以用高效模型更快地解决困难的构建与配置问题。

## Translation Links

- English: [README.md](README.md)
- Japanese: [README.ja.md](README.ja.md)
- Spanish: [README.es.md](README.es.md)

## 安装命令

通常不建议一次性安装全部 skill，这会浪费大量上下文。更好的方式是让 AI 先阅读 README，再根据当前任务只安装需要的 skill。

按名称安装单个 skill：

```bash
npx skills add <owner>/<repo> --skill <skill>
```

通过直接路径安装单个 skill：

```bash
npx skills add https://github.com/<owner>/<repo>/tree/main/skills/<skill>
```

## Skill 索引

- build-sageattention-rocm-on-win11: [skills/build-sageattention-rocm-on-win11/SKILL.md](skills/build-sageattention-rocm-on-win11/SKILL.md)
- dashboard-https-proxy: [skills/hermes/dashboard-https-proxy/SKILL.md](skills/hermes/dashboard-https-proxy/SKILL.md)

## 如何贡献新 skill

1. 用足够可靠的 AI 模型完整解决你的真实任务。
2. 让它把成功流程总结为可复用的 skill。
3. 将 skill 保存到 `skills/<your-skill-name>/`。
4. 更新 Skill 索引，方便其他 agent 快速发现。

## 说明

- 优先按任务安装，而不是批量安装。
- 每个 skill 聚焦一个明确问题。
- 尽量包含已知陷阱与验证步骤。
