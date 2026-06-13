# openai-test

Standalone OpenAI probing utilities built with `uv`.

This project is intentionally small and read-only except for the two probe commands
that create temporary OpenAI resources and clean them up afterward.

## Requirements

- Python 3.12 or newer
- `uv`
- An OpenAI API key with access to Files and Vector Stores

## Setup

```bash
uv sync --group dev
```

## Environment

Set this environment variable before running a command:

- `OPENAI_API_KEY`

Example:

```bash
export OPENAI_API_KEY=sk-proj-...
```

## Commands

### `vector-stores`

Lists every vector store in the account and reports how many files are attached.

```bash
uv run openai-test vector-stores
```

Output includes:

- total vector store count
- per-store id, name, and creation time
- attached file count per store
- end-of-list summary with vector store name, id, and total files, each on its own line

### `files`

Lists the total number of uploaded OpenAI files in the account.

```bash
uv run openai-test files
```

Output includes:

- total file count

### `vision-probe`

Uploads a local file and then retries a multimodal vision call until it succeeds.
This is useful for reproducing `file_id` visibility delays between `files.create`
and the vision/Responses path.

```bash
uv run openai-test vision-probe --file-path /path/to/image.jpg
```

Options:

- `--model`: OpenAI model to use, default `gpt-5-mini`
- `--prompt`: prompt sent with the image
- `--max-attempts`: number of retries, default `0` means retry forever

This probe:

- uploads the file
- waits for file processing
- retries file visibility checks
- retries the vision call until it succeeds
- deletes the uploaded file afterward

### `attach-probe`

Creates a temporary vector store, uploads a local file, attaches it, and cleans up
the temporary resources afterward.

```bash
uv run openai-test attach-probe --file-path /path/to/image.jpg
```

Options:

- `--max-attempts`: number of retries, default `0` means retry forever

This probe:

- creates a fresh temporary vector store
- uploads the file with `purpose=assistants`
- waits for file processing
- retries file visibility checks
- retries the vector-store attach call until it succeeds
- deletes the vector-store file, uploaded file, and temporary vector store afterward

### `attach-existing-probe`

Attaches a local file to an existing vector store and cleans up only the uploaded
file plus the attachment. The vector store itself is left untouched.

```bash
uv run openai-test attach-existing-probe \
  --file-path /path/to/image.jpg \
  --vector-store vs_...
```

Options:

- `--vector-store`: existing vector store id to test against
- `--max-attempts`: number of retries, default `0` means retry forever

This probe:

- uses the existing vector store you pass in
- uploads the file with `purpose=assistants`
- waits for file processing
- retries file visibility checks
- retries the vector-store attach call until it succeeds
- deletes the attached test file from the store and deletes the uploaded file afterward

## Retry behavior

The probing helpers retry OpenAI status codes that often represent transient
backend propagation or temporary failure conditions:

- `404`
- `408`
- `409`
- `425`
- `429`
- `500`
- `502`
- `503`
- `504`

The retry delay starts at `1s` and grows exponentially up to `60s`.

## Examples

List all files:

```bash
uv run openai-test files
```

Check vector store inventory:

```bash
uv run openai-test vector-stores
```

Probe vision with a specific prompt:

```bash
uv run openai-test vision-probe \
  --file-path /path-to/photo.jpg \
  --prompt "Describe the document contents for procurement review."
```

Probe a fresh vector store attach:

```bash
uv run openai-test attach-probe \
  --file-path /path-to/photo.jpg
```

## Notes

- The inventory commands can be slow in large accounts because they page through
  all files and vector stores.
- The retrying probe commands are intended for debugging propagation delays and
  attach behavior, not for normal application flows.
- Temporary resources created by the probe commands are cleaned up best-effort.
