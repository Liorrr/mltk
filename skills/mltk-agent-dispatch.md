# Dispatching subagents in mltk

Two skill files exist. The orchestrator MUST read and include them in agent prompts per the matrix below.

| Skill | Path | Content |
|-------|------|---------|
| **Index** | `~/.claude/skills/mltk-index.md` (generated) | assertions, MCP tools, CLI commands + groups, scanners, classes with file:line |
| **Templates** | `skills/mltk-templates.md` (repo) → `~/.claude/skills/` | Patterns for adding assertions, scanners, MCP tools, CLI commands |

Regenerate the index after each sprint: `python scripts/generate_skill_index.py`
Detailed reference with full signatures: `docs/reference/full-api-index.md`

## Which agents get which skills

| Agent Type | Index | Templates | Why |
|------------|:-----:|:---------:|-----|
| Builder | Y | Y | Needs file locations + code patterns |
| Test hardening | Y | N | Needs assertion names + test file mapping |
| Wiring/integration | Y | Y | Needs module structure + export patterns |
| Documentation | Y | N | Needs to know what exists |
| Reviewer/auditor | Y | N | Needs to navigate during review |
| Researcher | N | N | Searches web, not codebase |

## How to include

Read both skill files and paste their content into the agent prompt:

```
## Codebase Index
{content of ~/.claude/skills/mltk-index.md}

## Development Templates
{content of ~/.claude/skills/mltk-templates.md}
```
