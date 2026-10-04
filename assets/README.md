# Brand assets

| File | Size | Use |
|---|---|---|
| `logo/mojo-dllm.svg` | 256 × 256 | README, docs, favicon source |
| `logo/wordmark.svg` | 720 × 200 | logo plus name, for slides and posts |
| `social-preview.svg` / `.png` | 1280 × 640 | GitHub social preview (upload by hand in repository settings) |
| `logo/mojo-dllm-256.png`, `logo/mojo-dllm-1024.png` | | raster copies |

The mark is a 5 × 5 canvas of tokens. Teal dashed squares are still masked;
ember squares are resolved and form a flame. It reads as a diffusion model
denoising its canvas, in Mojo. Colours: plate `#0F1419`, ember `#FFB45A` to
`#FF6B1A`, core `#FFE3B0`, teal `#5EEAD4`.

`wordmark.svg` and `social-preview.svg` are generated from the logo:

```bash
python3 scripts/gen_brand.py
```

Rasterize with any headless Chromium (Brave shown). The logo SVG declares
256 × 256, so larger rasters come from a resized copy:

```bash
shot() {  # svg width height png
  brave-browser --headless=new --disable-gpu --hide-scrollbars \
    --default-background-color=00000000 --window-size="$2,$3" \
    --screenshot="$PWD/assets/$4" "file://$1"
}
sed 's/width="256" height="256"/width="1024" height="1024"/' \
  assets/logo/mojo-dllm.svg > /tmp/logo1024.svg
shot "$PWD/assets/logo/mojo-dllm.svg" 256 256 logo/mojo-dllm-256.png
shot /tmp/logo1024.svg 1024 1024 logo/mojo-dllm-1024.png
shot "$PWD/assets/logo/wordmark.svg" 720 200 logo/wordmark.png
shot "$PWD/assets/social-preview.svg" 1280 640 social-preview.png
```
