# task-skills

This repository is a multi-skill catalog for `npx skills add`.

Many skills here come from hard one-off tasks that used to be fragile and environment-dependent. In the past, people often shared these solutions as standalone shell or Python scripts. Those scripts helped many users, but they also consumed substantial time from volunteer developers, and this repository follows the same spirit while using AI skills as the sharing format.

Until personal devices can easily match the capability of today's SOTA models, practical skill sharing remains the most efficient path. The long-term goal is simple: convert practical experience into reusable skills, so more users can solve difficult setup and build issues quickly with efficient models.

## Translations

- Chinese: [README.zh-CN.md](README.zh-CN.md)
- Japanese: [README.ja.md](README.ja.md)
- Spanish: [README.es.md](README.es.md)

## Install commands

Installing every skill at once is usually a bad idea because it wastes context. A better workflow is to let your AI read this README, select only the relevant skill for the current task, and install it at the project level.

Install one named skill:

```bash
npx skills add <owner>/<repo> --skill <skill>
```

Install a direct skill path:

```bash
npx skills add https://github.com/<owner>/<repo>/tree/main/skills/<skill>
```

## Skill Index

- build-sageattention-rocm-on-win11: [skills/build-sageattention-rocm-on-win11/SKILL.md](skills/build-sageattention-rocm-on-win11/SKILL.md)

## Contribute a new skill

1. Use a capable AI model to complete your real task end-to-end.
2. Ask it to summarize the successful workflow as a reusable skill.
3. Save that skill under `skills/<your-skill-name>/`.
4. Update the Skill Index so other agents can discover it quickly.

## Notes

- Prefer task-specific installation over bulk installation.
- Keep each skill focused on one concrete problem.
- Include known pitfalls and verification steps whenever possible.
