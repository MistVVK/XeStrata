#!/usr/bin/env python3
"""Language identification (fork of langid.py by Marco Lui)."""

import logging
import math
import unicodedata
from collections import Counter
from operator import itemgetter
from pathlib import Path

import numpy as np

from .modelio import load_model as _load_model_file

LOGGER = logging.getLogger(__name__)

IDENTIFIER = None
MODEL_FILE = 'data/model.npz.xz'
MODEL_DIR = Path(__file__).parent
RAW_FLOOR = float(np.finfo(np.float32).min)  # finite floor for featureless input


def decode_trimmed(data):
    """Decode UTF-8, trimming ≤3 partial trailing bytes; None if undecodable.
    Shared train/inference contract (also used by train.common.nfc_bytes)."""
    for trim in range(4):
        chunk = data[:len(data) - trim] if trim else data
        try:
            return chunk.decode('utf8')
        except UnicodeDecodeError as e:
            if e.start < len(data) - 3:  # not fixable by trimming the tail
                return None
    return None


def visit_counts(nm, rowbase, out, text):
    """DFA-walk feature counts over bytes; None if none.
    Shared by inference (_raw_score) and training (train.stages)."""
    state, indexes = 0, []
    append = indexes.append
    for letter in text:
        state = nm[rowbase[state] + letter]
        f = out[state]
        if f >= 0:
            append(f)
    return Counter(indexes) if indexes else None


def _load_identifier(model_path=None, norm_probs=False, langs=None):
    if model_path:
        identifier = LanguageIdentifier.from_modelpath(model_path, norm_probs=norm_probs)
        LOGGER.info("Using external model: %s", model_path)
    else:
        identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=norm_probs)
    if langs:
        identifier.set_languages(langs)
    return identifier


def _get_identifier():
    global IDENTIFIER
    if IDENTIFIER is None:
        LOGGER.debug('initializing identifier')
        IDENTIFIER = _load_identifier()
    return IDENTIFIER


def set_languages(langs=None):
    return _get_identifier().set_languages(langs)


def classify(instance):
    return _get_identifier().classify(instance)


def rank(instance):
    return _get_identifier().rank(instance)


def _init_worker(model_path, norm_probs, langs):
    global IDENTIFIER
    if IDENTIFIER is None:  # forked workers inherit the parent's identifier
        IDENTIFIER = _load_identifier(model_path, norm_probs, langs)


def _process_file(path, dist=False):
    with open(path, 'rb') as f:
        text = f.read()
    return path, (rank(text) if dist else classify(text))


class LanguageIdentifier:
    __slots__ = [
        '_alias_pairs',
        '_full_model',
        '_norm_probs',
        '_rowbase',
        'min_confidence',
        'nb_classes',
        'nb_pc',
        'nb_ptc',
        'tk_nextmove',
        'tk_output',
        'tk_row',
    ]

    @classmethod
    def from_model_file(cls, model_file, *args, **kwargs):
        filepath = Path(model_file)
        if not filepath.is_absolute():
            filepath = MODEL_DIR / filepath
        ptc, pc, classes, nextmove, row, output = _load_model_file(filepath)
        return cls(np.asarray(ptc), np.asarray(pc), classes, nextmove, output,
                   *args, tk_row=row, **kwargs)

    @classmethod
    def from_modelpath(cls, path, *args, **kwargs):
        return cls.from_model_file(Path(path).absolute(), *args, **kwargs)

    def __init__(self, nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output,
                 norm_probs=False, min_confidence=None, *, tk_row):
        if min_confidence is not None and not norm_probs:
            raise ValueError("min_confidence requires norm_probs=True")
        self.min_confidence = min_confidence
        self.nb_ptc = nb_ptc
        self.nb_pc = nb_pc
        self.nb_classes = nb_classes
        self.tk_nextmove = tk_nextmove
        self.tk_row = tk_row
        self._rowbase = [r << 8 for r in tk_row]  # pre-shifted row offsets
        self.tk_output = tk_output
        self._norm_probs = norm_probs
        self._full_model = nb_ptc, nb_pc, nb_classes
        self._set_alias_pairs()

    def _set_alias_pairs(self):
        """(first, dupe) column pairs for labels appearing more than once."""
        first, pairs = {}, []
        for i, c in enumerate(self.nb_classes):
            if c in first:
                pairs.append((first[c], i))
            else:
                first[c] = i
        self._alias_pairs = pairs

    @property
    def labels(self):
        "Distinct output labels; script aliases (srl->sr) share one."
        return list(dict.fromkeys(self.nb_classes))

    def set_languages(self, langs=None):
        """Restrict classification to *langs* (ISO 639 codes), or reset to all."""
        LOGGER.debug("restricting languages to: %s", langs)
        nb_ptc, nb_pc, nb_classes = self._full_model
        if langs is None:
            self.nb_classes, self.nb_ptc, self.nb_pc = nb_classes, nb_ptc, nb_pc
        else:
            lang_set = set(langs)
            unknown = lang_set - set(nb_classes)
            if unknown:
                raise ValueError(f"Unknown language code(s): {unknown}")

            indices = [i for i, c in enumerate(nb_classes) if c in lang_set]
            self.nb_classes = [nb_classes[i] for i in indices]
            self.nb_ptc = nb_ptc[:, indices]
            self.nb_pc = nb_pc[indices]
        self._set_alias_pairs()

    @staticmethod
    def _encode(text):
        if isinstance(text, bytes):
            decoded = decode_trimmed(text)
            if decoded is not None:
                text = decoded
        if isinstance(text, str):
            if text.isupper():
                text = text.lower()
            text = unicodedata.normalize('NFC', text)
            text = text.encode('utf8', errors='surrogatepass')
        return text

    def _sparse_score(self, visits, table):
        """NB log-posterior from sparse {feature: count}."""
        idx = np.fromiter(visits.keys(), dtype=np.intp, count=len(visits))
        counts = np.fromiter(visits.values(), dtype=np.float32, count=len(visits))
        return np.log1p(counts) @ table[idx] + self.nb_pc

    def _raw_score(self, text):
        """Raw NB scores via DFA walk over encoded bytes."""
        visits = visit_counts(self.tk_nextmove, self._rowbase, self.tk_output,
                              text)
        if visits:
            return self._sparse_score(visits, self.nb_ptc)

        # no features: 0.0 under norm_probs (uniform → abstain), RAW_FLOOR otherwise
        fill = 0.0 if self._norm_probs else RAW_FLOOR
        return np.full(len(self.nb_classes), fill, dtype=np.float32)

    def _decide(self, text):
        """Score per class, optionally normalized to probabilities."""
        text = self._encode(text)
        scores = self._raw_score(text)
        if self._norm_probs:
            # T = sqrt(bytes) keeps softmax calibrated across lengths
            scores *= 1.0 / math.sqrt(len(text) or 1)
            np.exp(scores - scores.max(), out=scores)
            scores /= scores.sum()
        # aliased columns (srl->sr): fold the dupe into the first occurrence and
        # mask it, so argmax and rank agree on one score per label
        for i, j in self._alias_pairs:
            if self._norm_probs:
                scores[i] += scores[j]
                scores[j] = 0.0
            else:
                scores[i] = max(scores[i], scores[j])
                scores[j] = RAW_FLOOR
        return scores

    def classify(self, text):
        """Return *(language, confidence)* for *text* (str or UTF-8 bytes)."""
        scores = self._decide(text)
        i = int(scores.argmax())
        conf = float(scores[i])
        if self.min_confidence is not None and conf < self.min_confidence:
            return 'und', conf
        return self.nb_classes[i], conf

    def rank(self, text):
        """All languages by likelihood, best first, one entry per label."""
        merged = {}
        for lang, score in zip(self.nb_classes, self._decide(text).tolist()):
            merged.setdefault(lang, score)  # first column holds the merged score
        return sorted(merged.items(), key=itemgetter(1), reverse=True)


def main():

    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--serve', action='store_true', help='launch web service')
    parser.add_argument('--host', help='host/ip to bind to')
    parser.add_argument('--port', default=9008, type=int, help='port to listen on')
    parser.add_argument('-v', action='count', dest='verbosity', help='increase verbosity (repeat for greater effect)')
    parser.add_argument('-m', dest='model', help='load model from file')
    parser.add_argument('-l', '--langs', help='comma-separated set of target ISO639 language codes (e.g en,de)')
    parser.add_argument('-r', '--remote', action='store_true', help='auto-detect IP address for remote access')
    parser.add_argument('-b', '--batch', action='store_true', help='read file paths from stdin and classify in parallel')
    parser.add_argument('-d', '--dist', action='store_true', help='show full distribution over languages')
    parser.add_argument('-u', '--url', help='classify text from URL')
    parser.add_argument('--line', action='store_true', help='process pipes line-by-line rather than as a document')
    parser.add_argument('-n', '--normalize', action='store_true', help='normalize confidence scores to probability values')
    options = parser.parse_args()

    if options.verbosity:
        logging.basicConfig(level=max((5-options.verbosity)*10, 0))
    else:
        logging.basicConfig()

    if options.batch and options.serve:
        parser.error("cannot specify both batch and serve at the same time")

    global IDENTIFIER

    langs = options.langs.split(",") if options.langs else None
    IDENTIFIER = _load_identifier(options.model, options.normalize, langs)

    _process = IDENTIFIER.rank if options.dist else IDENTIFIER.classify

    if options.url:
        from urllib.request import urlopen
        with urlopen(options.url) as url:
            text = url.read()
            output = _process(text)
            print(options.url, len(text), output)

    elif options.serve:
        import socket
        from wsgiref.simple_server import make_server

        from .server import application

        if options.remote and options.host is None:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("google.com", 80))
                hostname = s.getsockname()[0]
        elif options.host is None:
            hostname = socket.gethostbyname(socket.gethostname())
        else:
            hostname = options.host

        print(f"Listening on {hostname}:{options.port}")
        print("Press Ctrl+C to exit")
        httpd = make_server(hostname, options.port, application)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

    elif options.batch:
        import csv
        import multiprocessing as mp
        from functools import partial

        def paths():
            for line in sys.stdin:
                p = line.strip()
                if p and Path(p).is_file():
                    yield p

        writer = csv.writer(sys.stdout, lineterminator='\n')
        ctx = mp.get_context('fork') if sys.platform == 'darwin' else mp
        with ctx.Pool(processes=mp.cpu_count(),
                      initializer=_init_worker,
                      initargs=(options.model, options.normalize, langs)) as pool:
            if options.dist:
                header = IDENTIFIER.labels
                writer.writerow(['path', 'language'] + header)
                for path, ranking in pool.imap_unordered(partial(_process_file, dist=True), paths()):
                    scores = dict(ranking)
                    row = [path, ranking[0][0]] + [scores[c] for c in header]
                    writer.writerow(row)
            else:
                for path, (lang, conf) in pool.imap_unordered(_process_file, paths()):
                    writer.writerow((path, lang, conf))
    else:
        if sys.stdin.isatty():
            while True:
                try:
                    print(">>>", end=' ')
                    text = input()
                except (KeyboardInterrupt, EOFError):
                    break
                print(_process(text))
        else:
            if options.line:
                for line in sys.stdin:
                    print(_process(line))
            else:
                print(_process(sys.stdin.read()))


if __name__ == "__main__":
    main()
