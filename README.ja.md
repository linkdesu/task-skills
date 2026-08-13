# task-skills

このリポジトリは、`npx skills add` 向けのマルチスキルカタログです。

ここにある多くの skill は、複雑で一度きりになりがちな、環境依存の強い作業から作られています。以前はこの種の知見を shell や Python スクリプトとして共有することが一般的で、多くのユーザーを助けましたが、善意の開発者の時間を大きく消費する面もありました。本リポジトリはその精神を引き継ぎ、AI skill という形式で共有します。

個人デバイスが現在の SOTA モデル級の能力に容易に到達できるようになるまでは、実用的な skill 共有が最も効率的な道です。長期的な目的はシンプルです。実践知を再利用可能な skill に変換し、より多くのユーザーが効率的なモデルで難しいセットアップやビルド問題を素早く解決できるようにすることです。

## Translation Links

- English: [README.md](README.md)
- Chinese: [README.zh-CN.md](README.zh-CN.md)
- Spanish: [README.es.md](README.es.md)

## インストールコマンド

すべての skill を一度にインストールするのは、通常はおすすめしません。コンテキストを無駄に消費するためです。まず AI に README を読ませ、現在のタスクに必要な skill だけをプロジェクト単位でインストールするのが良い運用です。

名前を指定して 1 つの skill をインストール:

```bash
npx skills add <owner>/<repo> --skill <skill>
```

skill の直接パスでインストール:

```bash
npx skills add https://github.com/<owner>/<repo>/tree/main/skills/<skill>
```

## Skill Index

- build-sageattention-rocm-on-win11: [skills/build-sageattention-rocm-on-win11/SKILL.md](skills/build-sageattention-rocm-on-win11/SKILL.md)
- comfyui-minimax-h3-rocm-tuning: [skills/comfyui-minimax-h3-rocm-tuning/SKILL.md](skills/comfyui-minimax-h3-rocm-tuning/SKILL.md)
- dashboard-https-proxy: [skills/hermes/dashboard-https-proxy/SKILL.md](skills/hermes/dashboard-https-proxy/SKILL.md)

## 新しい skill の追加方法

1. 信頼できる AI モデルで実タスクを最後まで解決します。
2. 成功した手順を再利用可能な skill として要約させます。
3. `skills/<your-skill-name>/` に保存します。
4. 他の agent が見つけやすいよう Skill Index を更新します。

## Notes

- 一括導入ではなく、タスク単位の導入を優先してください。
- 各 skill は 1 つの具体的な問題に集中させてください。
- 可能な限り既知の落とし穴と検証手順を含めてください。
