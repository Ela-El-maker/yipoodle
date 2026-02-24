# Domain Source Configs

Use these files with `--sources-config` to switch ingestion behavior by domain.

## Available presets
- `config/domains/sources_nlp.yaml`
- `config/domains/sources_biomed_ai.yaml`
- `config/domains/sources_robotics.yaml`
- `config/domains/sources_cybersecurity.yaml`
- `config/domains/sources_multimodal_ai.yaml`
- `config/domains/sources_speech_audio.yaml`
- `config/domains/sources_reinforcement_learning.yaml`
- `config/domains/sources_data_engineering.yaml`
- `config/domains/sources_cloud_infrastructure.yaml`
- `config/domains/sources_software_engineering.yaml`
- `config/domains/sources_finance_markets.yaml`

## Quick usage
```bash
python -m src.cli sync-papers \
  --query "your domain query" \
  --max-results 20 \
  --with-semantic-scholar \
  --sources-config config/domains/sources_nlp.yaml
```

To use automation with a domain preset, set this in `config/automation.yaml`:
```yaml
paths:
  sources_config: config/domains/sources_nlp.yaml
```

## Domain OCR defaults
`extract-corpus` can read OCR defaults from the same domain file:
```yaml
ocr:
  enabled: false
  timeout_sec: 30
  min_chars_trigger: 120
  max_pages: 20
  min_output_chars: 200
  min_gain_chars: 40
  min_confidence: 45.0
  lang: eng+spa
  profile: sparse
  noise_suppression: true
```
Precedence is: `CLI flags` > `sources-config ocr block` > `built-in defaults`.

## Auto-generate a new domain config
```bash
python -m src.cli scaffold-domain-config --domain "marketing growth"
```
Optional:
```bash
python -m src.cli scaffold-domain-config --domain "marketing growth" --profile industry --overwrite
```
