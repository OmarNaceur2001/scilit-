# SciLit optional Groq integration

This patch adds an opt-in Groq-backed claim-rewriting stage to the provided SciLit Bronze template generator.
It keeps template generation as the default and does not change retrieval, claim extraction, source IDs, BibTeX creation, or the existing NLI stage.

## Apply it

From PowerShell, unpack the ZIP somewhere and run the setup script with your repository root:

```powershell
python .\scilit_groq_setup.py E:\scilit
```

The script creates `.pre-groq.bak` backup copies for changed files and does not modify `.env`. Review changes with `git diff` before committing.

## Install and configure

```powershell
python -m pip install -e ".[dev,llm]"
```

Add these values to your local `E:\scilit\.env`:

```dotenv
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=your_real_key_here
```

Keep the real key out of GitHub. Restart the API process after changing `.env`.

To keep the original deterministic template output, use:

```dotenv
LLM_PROVIDER=template
```

## Behavior and safety

- Groq receives only the selected retrieved claims and their extracted fields.
- The model is instructed not to add facts, figures, conclusions, or citations.
- A strict JSON schema requires one rewritten sentence for each source claim ID.
- SciLit continues to assign article citations and supporting claim IDs itself.
- The existing NLI audit remains a separate step and is not a guarantee of factual correctness.
- If the configured Groq request fails or the response fails validation, generation reports an error; it does not silently label template output as LLM output.
- The Groq SDK is optional; the default `template` mode does not import or call it.

Default model ID is documented by Groq as `openai/gpt-oss-20b`, including support for JSON Schema mode. Check Groq's current model documentation before changing it:
https://console.groq.com/docs/models
https://console.groq.com/docs/structured-outputs
