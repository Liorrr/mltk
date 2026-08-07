# Install mltk (honest path)

| What | Name |
|------|------|
| **PyPI distribution** | `mlspec` |
| **Python import** | `mltk` |
| **CLI** | `mltk` |
| **GitHub** | https://github.com/Liorrr/mltk |

## Recommended

```bash
pip install "mlspec[cli,report]"
python -c "import mltk; print(mltk.__version__)"
mltk doctor
mltk list | head
```

## Do not

```bash
pip install mltk   # WRONG — different package on PyPI
```

## From source

```bash
pip install "git+https://github.com/Liorrr/mltk"
# or
git clone https://github.com/Liorrr/mltk
cd mltk && pip install -e ".[dev,cli,report]"
```

## Homebrew

```bash
brew tap Liorrr/mltk
brew install mltk
```

## Docker

See `Dockerfile` — image installs **`mlspec[all]`**.

## Extras

Any extra that used to be documented as `mltk[...]` is now:

```bash
pip install "mlspec[scipy]"
pip install "mlspec[server]"
pip install "mlspec[all]"
```

Import paths stay `from mltk...`.

## Commercial

See [COMMERCIAL.md](COMMERCIAL.md). Payments after Osek Patur; free OSS use is ELv2.
