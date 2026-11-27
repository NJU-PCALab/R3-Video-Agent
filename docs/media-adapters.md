# Hosted media workers

Install the local stack first if you need the reference configuration. A hosted adapter is an optional executable supplied by the operator. Each of the three routes can be replaced independently in the private workspace's `config.json`. For example, set `media_adapters.image.executable` to your executable's absolute path and `media_adapters.image.model` to the public model ID enabled on your account. Audio and video use the same two fields. Keep vendor credentials in the worker environment.

## Invocation

The harness invokes the executable as an argument vector, without a shell.

```text
adapter --kind KIND --model MODEL --check
adapter --kind KIND --model MODEL --request REQUEST --response RESPONSE --output-dir DIR
```

`--check` verifies local installation, credentials presence and capability configuration without submitting a generation. Exit zero on success. On a job, read the UTF-8 JSON request, validate every required reference and the requested operation, materialize outputs under DIR, then atomically rename the final JSON receipt into RESPONSE. The worker and controller must share access to the referenced files. Submit the operation once. Preserve the vendor job ID and use the vendor's idempotency or status mechanism to resolve interruptions. Strip authorization headers and signed URLs from logs and receipts.

| Route | Inputs to preserve | Success receipt and media |
|---|---|---|
| Image | Prompt, ordered parents with checksums, dimensions, seed where supported, optional character age edit | `{"ok":true,"images":[{"path":"..."}]}` with a real, decodable image. For age edits verify the actual parent before upload and supply `parent_hashes` and `character_age_condition` including the original edit specification and source SHA-256. |
| Audio | Generator mode (voice design, clone, or music), language, full text, voice description, isolated reference for cloning, requested duration | `{"ok":true,"path":"..."}` with a decodable audio file of at least two seconds. Preserve the audible speaker and requested line. A speech-only service needs another implementation for requested music or must reject it. |
| Video | Compiled prompt and ordered conditions, including stills, video, audio and optional frame indices, target duration and aspect | Write `native.mp4` into DIR, then `{"ok":true,"status":"completed"}`. The controller checks the real frames and native audio, transcribes when configured and creates the 24 fps presentation copy. |

Use absolute local file paths in receipts. For a known failure return `{"ok":false,"status":"failed","error":"..."}`. If submission may have been accepted but the result is unknown, preserve the vendor receipt and return `{"ok":false,"status":"failed","needs_reconciliation":true,"error":"..."}`. This leaves the job pending for reconciliation. Report unsupported capabilities explicitly. A new model may have different sampling or reference behavior. Record these differences when comparing results.

## Vendor implementation references

- **Image:** GPT Image supports both generation and edits with image inputs. Use edits when parents are present and preserve their order. [OpenAI Images API](https://developers.openai.com/api/docs/guides/image-generation)
- **Voice:** a MiniMax Speech adapter can use Voice Design to create a voice, save its authorized voice identifier and pass it to speech synthesis. Voice cloning needs an authorized reference and the provider's dedicated clone workflow. [MiniMax API documentation](https://platform.minimax.io/docs)
- **Video:** a Seedance adapter can create a video task, retain its identifier, poll it and materialize the result. Select a deployment that accepts the story's reference modalities and produces native audio. [BytePlus ModelArk video generation task API](https://docs.byteplus.com/en/docs/ModelArk/create-video-generation-task-api)

These links describe vendor integration points. The repository includes the transport contract and offline executable tests. Each operator supplies the vendor implementation and tests it against their account, enabled model and reference requirements.
