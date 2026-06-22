$src = "<experiment-dir>\workdir\Task302_anonymisation_ner_aug-fold0\checkpoints"
$dst = "<local-model-dir>"

$files = @(
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin"
)

New-Item -ItemType Directory -Force -Path $dst | Out-Null

foreach ($file in $files) {
    $srcPath = Join-Path $src $file
    $dstPath = Join-Path $dst $file
    Write-Host "Copying $file..."
    Copy-Item -Path $srcPath -Destination $dstPath
}

Write-Host "Done. Model copied to $dst"
