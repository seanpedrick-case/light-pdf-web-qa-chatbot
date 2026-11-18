# Bedrock KB metadata filtering (app side)

This chatbot does **not** prepare or upload knowledge-base source files. Configure ingestion in your data pipeline / other repo. Here you only set env vars that match **your** KB metadata schema.

**Local files (gitignored):** `config/app_config.env`, `config/kb_filter_groups.txt`, `config/kb_filter_prompt.txt`. Copy from `*.example.txt` templates where provided.

## App env (typical)

```env
USE_KB_METADATA_FILTER=1
KB_METADATA_FILTER_KEYS=Category,Department,Status,DocumentId,YearPublished
KB_DATE_METADATA_KEYS=PublishedAt,UpdatedAt
KB_DATE_FILTER_STRATEGY=semantic_only
KB_GROUP_FILTER_KEY=Department
KB_GROUP_OPTIONS_PATH=config/kb_filter_groups.example.txt
KB_FILTER_PROMPT_PATH=config/kb_filter_prompt.example.txt
```

## Date strategies (`KB_DATE_FILTER_STRATEGY`)

| Value | Use when |
|--------|-----------|
| `semantic_only` | Default safe choice: no metadata date clauses; dates via semantic search |
| `year_in` | KB has string year columns; set `KB_YEAR_METADATA_KEYS` and `KB_DATE_TO_YEAR_MAP` |
| `numeric_yyyymmdd` | KB stores filterable **NUMBER** dates |
| `iso_range` | OpenSearch-style KB with ISO date strings |

## Year-only filtering (`year_in`)

```env
KB_DATE_FILTER_STRATEGY=year_in
KB_YEAR_METADATA_KEYS=YearPublished,YearUpdated
KB_DATE_TO_YEAR_MAP=PublishedAt:YearPublished,UpdatedAt:YearUpdated
```

Include `Year*` keys in `KB_METADATA_FILTER_KEYS`. Day-level range filters on raw date keys are dropped.

## Group allow-list

`KB_GROUP_FILTER_KEY` + `KB_GROUP_OPTIONS_PATH` + placeholders `{{GROUP_FILTER_KEY}}` / `{{GROUP_OPTIONS}}` in `KB_FILTER_PROMPT_PATH`.

See [S3 Vectors metadata filtering](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-metadata-filtering.html) and [Bedrock KB queries](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html).

## CSV record sidecar size

For `source.csv.metadata.json` (and similar `RECORD_BASED_STRUCTURE_METADATA` sidecars), ingestion can fail if the file is **over ~1 KB** when pretty-printed. Bedrock then **ignores** the sidecar (`MaximumFileSizeSupported: 1024 bytes`) and the CSV may show **Failed to ingest**. Keep the JSON **minified** (no extra whitespace); avoid long `metadataAttributes` strings. Re-upload the sidecar to S3 and sync.
