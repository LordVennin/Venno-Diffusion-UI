import os.path

folder = os.path.dirname(__file__)  # huggingface


class Token:
    # Anima's Qwen3 tokenizer is shipped compressed (shared with Z-Image / Flux.2 Klein upstream)
    z_compress = os.path.join(folder, "z.tokenizer.json.xz")
    anima = os.path.join(folder, "circlestone-labs", "Anima", "tokenizer", "tokenizer.json")


class sha256:
    anima = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


def decompress(source: str, target: str):
    import lzma

    with lzma.open(source, "rb") as input_file:
        with open(target, "wb") as output_file:
            output_file.write(input_file.read())


def compress(source: str, target: str):
    import lzma

    with open(source, "rb") as input_file:
        with lzma.open(target, "wb") as output_file:
            output_file.write(input_file.read())


def process():
    if not os.path.isfile(Token.anima):
        decompress(Token.z_compress, Token.anima)
        compare_sha256(Token.anima, sha256.anima)


def compare_sha256(path: str, target: str):
    import hashlib

    with open(path, "rb") as f:
        data = f.read()
    _hash = hashlib.sha256(data)
    assert _hash.hexdigest() == target
