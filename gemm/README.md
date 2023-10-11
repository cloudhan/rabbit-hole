## Build

```bash
bazel build -c opt --config=linux cpu/...

# Generate compile_commands.json
bazel run @hedron_compile_commands//:refresh_all --symlink_prefix=linux/bazel- -- --config=linux
```

## Test

```
pytest cpu/test_driver.py
```
