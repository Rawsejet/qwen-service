# VL-4B Image Inference Test Scripts

## Important Note

The VL-4B model requires `--allowed-local-media-path` to be set when starting the server to allow local file access. This is now configured in `start-qwen.sh` for the VL-4B model.

**Note:** VL-4B uses `image_url` type (not `image`) with either remote URLs or `file://` paths.

## Process a Single Remote Image

```bash
curl -s http://127.0.0.1:8088/v1/chat/completions -X POST -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-VL-4B","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/receipt.png"}},{"type":"text","text":"Read all the text in the image."}]}],"max_tokens":512}'
```

## Process a Single Local Image

```bash
curl -s http://127.0.0.1:8088/v1/chat/completions -X POST -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-VL-4B","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///home/teja/Downloads/image.jpg"}},{"type":"text","text":"Describe this image."}]}],"max_tokens":256}'
```

## Process All Images in a Folder

```bash
for img in /Documents/pics/screenshots/*.{jpg,png,jpeg}; do
  if [ -f "$img" ]; then
    echo "=== Processing: $img ==="
    curl -s http://127.0.0.1:8088/v1/chat/completions -X POST -H "Content-Type: application/json" \
      -d '{"model":"Qwen3-VL-4B","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file://'"$img"'"}},{"type":"text","text":"Read all the text in the image."}]}],"max_tokens":512}'
    echo ""
  fi
done
```

## Process with a Specific Prompt

Replace `"Read all the text in the image."` with your desired prompt, such as:
- `"What is in this image?"`
- `"Can you identify and describe the objects?"`
- `"Extract any text you see."`
- `"Describe the coding problem shown."`

## GPU Monitoring

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Check current GPU memory per process
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```
