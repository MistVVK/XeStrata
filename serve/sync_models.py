# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sync pack file sizes and hashes from the manifests at setup's pinned revisions."""

PACK_FILES = {
    'Sync-XS': {
        'compat-bf16.json': (
            40854, '53b2bf856b9321312904751bd31923bdaed70f3cdeb2f850d2962b6b96704e9c'),
        'conversions.json': (
            151849, 'd582d347cb951bde742347f812edd58cee7433bfeacf9a034fe4ff650645fe72'),
        'dense.bin': (
            1538035200, '23f06527ca2ebd2dc44d25b61f48d7c259e891719e28b9f59a60a4098d65b5c5'),
        'dense_nq.bin': (
            600883200, '22ef24d20049a0dd5ab22780c3f5bcd962af8c4a9b0e23d9c7ef45fa004ee83b'),
        'dense_nq.txt': (
            2013, 'cabbb27132dac4488fb901c9c13df2b7a64893175a3ab920b9bae28c2e4d1fb9'),
        'experts.v2.bin': (
            21976774656, 'a2af351608e995c47443c891b9e9c95754895dcb130546776e3b633c084a413f'),
        'experts_k.txt': (
            358331, '6779afcb6edd896112393e68ef7771ee6cf9f8c1fd7a31e70efda290c7e40874'),
        'hadamard.txt': (
            8938, '0ac6508d90038f53cb193cc91ae9c06e9352f154861f2777dcdba842be89668d'),
        'index.txt': (
            93669, 'a2150ca2d15ba59f8d9b299d122e3ad0474517ed4a0c97ec63c828201f9704f6'),
        'native_experts.txt': (
            3539, 'b0857627c97ac086b338e79598ce0258fa4be8343303115c6f261f51969b68d0'),
        'r1q2.txt': (
            3823, '163a5907f6b80997b74dd455fb03484668d0e504c80dbbc6d4dbdcda25dd9f5c'),
    },
    'Sync-S': {
        'compat-bf16.json': (
            36116, '02efffb529190f127881e165703b4780a5311bd39f8d7c4f18dd152f5c6a47f1'),
        'conversions.json': (
            151849, 'd582d347cb951bde742347f812edd58cee7433bfeacf9a034fe4ff650645fe72'),
        'dense.bin': (
            1538035200, '23f06527ca2ebd2dc44d25b61f48d7c259e891719e28b9f59a60a4098d65b5c5'),
        'dense_nq.bin': (
            755548160, '26c68e7391f15fae9241f0b7afb5dcbe6d73eedc9906e30d4b57c766bf7cca4c'),
        'dense_nq.txt': (
            2013, '486b6ad57421e2d3f5fd3adee5ae5591287208b31f0f813a6a3284ad4428d35c'),
        'experts.v2.bin': (
            25846112256, '1864ef8ab8be5a23bbe6b6f7ad8ee28a913fdf819f2c56ff90fbd73b017e60f8'),
        'experts_k.txt': (
            358331, '3375e87bfec8eaf104b2096fd39927d9ff7b2aad3f5e1d3f858e99bf36688311'),
        'hadamard.txt': (
            8938, '0ac6508d90038f53cb193cc91ae9c06e9352f154861f2777dcdba842be89668d'),
        'index.txt': (
            93669, 'a2150ca2d15ba59f8d9b299d122e3ad0474517ed4a0c97ec63c828201f9704f6'),
        'native_experts.txt': (
            3543, '3894883c55a7ff63956eb1fe62d0e2521e7250adf0a09c8fc7a0d71abc316604'),
        'r1q2.txt': (
            3818, 'f94fa7cf8a5bbcfb7da912297275a834ef27933726b387e7d98566b0eb8e3407'),
    },
    'Sync-M': {
        'compat-bf16.json': (
            36116, '02efffb529190f127881e165703b4780a5311bd39f8d7c4f18dd152f5c6a47f1'),
        'conversions.json': (
            151849, 'd582d347cb951bde742347f812edd58cee7433bfeacf9a034fe4ff650645fe72'),
        'dense.bin': (
            1538035200, '23f06527ca2ebd2dc44d25b61f48d7c259e891719e28b9f59a60a4098d65b5c5'),
        'dense_nq.bin': (
            755548160, '26c68e7391f15fae9241f0b7afb5dcbe6d73eedc9906e30d4b57c766bf7cca4c'),
        'dense_nq.txt': (
            2013, '486b6ad57421e2d3f5fd3adee5ae5591287208b31f0f813a6a3284ad4428d35c'),
        'experts.v2.bin': (
            32294701056, 'e6f2e60942274be59e5acc19c75c0708013eb48ebb7e580611d9c9bad385483c'),
        'experts_k.txt': (
            358331, 'a72d7e27b893b1e6b2e61c1a65ef8c653884bb203855a0c2684b2e024b5c5b26'),
        'hadamard.txt': (
            8938, '0ac6508d90038f53cb193cc91ae9c06e9352f154861f2777dcdba842be89668d'),
        'index.txt': (
            93669, 'a2150ca2d15ba59f8d9b299d122e3ad0474517ed4a0c97ec63c828201f9704f6'),
        'native_experts.txt': (
            3547, '1313b6683d6ca036e7d9f88944c718016113051e362435568b72d9853149637f'),
        'r1q2.txt': (
            3823, '4b3330558d148e320cef4281aef834af302f56ca84eae26268ce6fe0819400c0'),
    },
}

SHARED_FILES = {
    'native-gguf/Qwen3.8-Flash-Next-Sync-common.gguf': (
        32692038016, '82a78508d291130fdf4654b90acb5780e8f7f44b28b0ac3495aed59163a00e69'),
    'mtp-rt/dense.bin': (
        116099072, 'c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a'),
    'mtp-rt/dense.txt': (
        1880, '8773c81ebb0986e37fe94a8a9933e87be48b1fabc6889a0106bcdab179d1c2ac'),
    'mtp-rt/draft_vocab.bin': (
        425196, 'b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91'),
    'mtp-rt/experts.bin': (
        707788800, '09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f'),
    'tokenizer/chat_template.jinja': (
        9993, '12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce'),
    'tokenizer/merges.txt': (
        3600844, '080cf95432173729d160346b7882bd2a644b07f7a3813a3eb73f4e66787bfde9'),
    'tokenizer/token_type.json': (
        744960, '5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b'),
    'tokenizer/tokenizer.json': (
        554, '87be2ac47d8bc7393b7435df6ccd276740019a393941a2585d5edce59f450c54'),
    'tokenizer/vocab.json': (
        5737005, '4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4'),
}


def sha256(path, size):
    """Hash large files without the Linux page cache, falling back to a normal read."""
    import hashlib
    import mmap
    import os
    import sys

    if not sys.platform.startswith("linux") or size < 32 << 20:
        from serve.download import sha256_file
        return sha256_file(path)
    chunk = 4 << 20
    aligned = size - size % 4096
    digest = hashlib.sha256()
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError:
        return _sha256_buffered(path, size)
    try:
        with mmap.mmap(-1, chunk) as buf:
            offset = 0
            while offset < aligned:
                want = min(chunk, aligned - offset)
                view = memoryview(buf)[:want]
                try:
                    n = os.readv(fd, [view])
                    if n != want:
                        raise OSError(f"short direct read at {offset}: {n} of {want}")
                    digest.update(view)
                finally:
                    view.release()
                offset += n
        if aligned != size:
            with open(path, "rb") as f:
                f.seek(aligned)
                tail = f.read(size - aligned)
            if len(tail) != size - aligned:
                raise OSError(f"short tail at {aligned}")
            digest.update(tail)
    except OSError:
        return _sha256_buffered(path, size)
    finally:
        os.close(fd)
    return digest.hexdigest()


def _sha256_buffered(path, size):
    """Hash the pinned prefix through the page cache when direct I/O is refused."""
    import hashlib

    digest = hashlib.sha256()
    left = size
    with open(path, "rb") as f:
        while left:
            data = f.read(min(1 << 20, left))
            if not data:
                raise OSError(f"short read at {size - left} of {size}")
            digest.update(data)
            left -= len(data)
    return digest.hexdigest()
