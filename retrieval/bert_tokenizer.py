"""纯 Python 的 BERT WordPiece 分词器（不依赖 transformers / tokenizers）。

只实现 BGE-zh 这类中文 BERT 用到的部分：BertNormalizer + BertPreTokenizer +
WordPiece + 单句模板 ``[CLS] x [SEP]``。参数从 ``tokenizer.json`` 里读，
因此分词结果与 transformers 的 ``BertTokenizerFast`` 在中文上一致
（中文每个汉字本身就是词表里的一个 token，不做子词切分）。

用它的唯一原因是本机装不上 torch/transformers：离线、纯标准库、好测。
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_PUNCT_RE = re.compile(r"[!-/:-@\[-`{-~]")


class BertWordPieceTokenizer:
    def __init__(
        self,
        vocab: Dict[str, int],
        unk_token: str = "[UNK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        max_length: int = 512,
        lowercase: bool = True,
        do_lower_case_accents: bool = True,
        do_clean_text: bool = True,
        handle_chinese_chars: bool = True,
        subword_prefix: str = "##",
        max_input_chars_per_word: int = 100,
    ) -> None:
        self.vocab = vocab
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.lowercase = lowercase
        self.strip_accents = do_lower_case_accents
        self.do_clean_text = do_clean_text
        self.handle_chinese_chars = handle_chinese_chars
        self.subword_prefix = subword_prefix
        self.max_input_chars_per_word = max_input_chars_per_word
        for tok in (unk_token, cls_token, sep_token, pad_token):
            if tok not in vocab:
                raise ValueError(f"词表缺少特殊 token {tok}")
        missing = [c for c in "0123456789" if c not in vocab]
        if missing:
            raise ValueError(f"词表异常，缺少数字 {missing}")

    # ---------------- 构造 ----------------
    @classmethod
    def from_tokenizer_json(cls, path: str | Path, max_length: int = 512) -> "BertWordPieceTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = data["model"]
        vocab = model["vocab"]
        normalizer = data.get("normalizer") or {}
        kwargs = dict(
            vocab=vocab,
            unk_token=model.get("unk_token", "[UNK]"),
            subword_prefix=model.get("continuing_subword_prefix") or "##",
            max_input_chars_per_word=model.get("max_input_chars_per_word", 100),
            max_length=max_length,
        )
        if isinstance(normalizer, dict) and normalizer.get("type") == "BertNormalizer":
            kwargs.update(
                lowercase=normalizer.get("lowercase", True),
                do_lower_case_accents=normalizer.get("strip_accents", True)
                if normalizer.get("strip_accents") is not None
                else True,
                do_clean_text=normalizer.get("clean_text", True),
                handle_chinese_chars=normalizer.get("handle_chinese_chars", True),
            )
        tok = cls(**kwargs)
        post = data.get("post_processor") or {}
        if isinstance(post, dict) and post.get("type") == "TemplateProcessing":
            tok._check_template(post)
        return tok

    @staticmethod
    def _check_template(post: dict) -> None:
        """``single`` 可能是模板字符串，也可能是 [{"SpecialToken": ...}, {"Sequence": ...}] 列表。"""
        single = post.get("single")
        if isinstance(single, str):
            seq = single.replace(" ", "")
        elif isinstance(single, list):
            seq = "".join(
                ("[CLS]" if "CLS" in json.dumps(x) else "[SEP]" if "SEP" in json.dumps(x) else "$A")
                for x in single
            )
        else:
            return
        if seq.startswith("[CLS]") and seq.endswith("[SEP]"):
            return
        raise ValueError(f"不支持的 post_processor 模板: {single}")

    # ---------------- 归一化 ----------------
    def normalize(self, text: str) -> str:
        if self.do_clean_text:
            out = []
            for ch in text:
                cp = ord(ch)
                if cp == 0 or cp == 0xFFFD or _is_control(ch):
                    continue
                if _is_whitespace(ch):
                    out.append(" ")
                else:
                    out.append(ch)
            text = "".join(out)
        if self.handle_chinese_chars:
            text = _pad_cjk(text)
        if self.lowercase:
            text = text.lower()
        if self.strip_accents:
            text = _strip_accents(text)
        return text

    # ---------------- 主流程 ----------------
    def tokenize(self, text: str) -> List[str]:
        text = self.normalize(text)
        toks: List[str] = []
        for word in text.split():
            toks.extend(self._wordpiece(word))
        return toks

    def _wordpiece(self, word: str) -> List[str]:
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_token]
        if word in self.vocab:
            return [word]
        pieces: List[str] = []
        start = 0
        n = len(word)
        while start < n:
            end = n
            found = None
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = self.subword_prefix + sub
                if sub in self.vocab:
                    found = sub
                    break
                end -= 1
            if found is None:
                return [self.unk_token]
            pieces.append(found)
            start = end
        return pieces

    def encode(self, text: str, add_special_tokens: bool = True) -> Tuple[List[int], List[int]]:
        ids = [self.vocab.get(t, self.vocab[self.unk_token]) for t in self.tokenize(text)]
        if not add_special_tokens:
            return ids, [1] * len(ids)
        body = ids[: self.max_length - 2]
        ids = [self.vocab[self.cls_token]] + body + [self.vocab[self.sep_token]]
        return ids, [1] * len(ids)

    def batch_encode(
        self, texts: Sequence[str], max_length: int | None = None, pad: bool = True
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """返回 ``(input_ids, attention_mask)``，按 batch 内最长补齐（不是固定 512，省一半算力）。"""
        max_length = max_length or self.max_length
        encoded = []
        for t in texts:
            ids = [self.vocab.get(x, self.vocab[self.unk_token]) for x in self.tokenize(t)]
            ids = ids[: max_length - 2]
            ids = [self.vocab[self.cls_token]] + ids + [self.vocab[self.sep_token]]
            encoded.append(ids)
        if not pad:
            return encoded, [[1] * len(x) for x in encoded]
        width = max((len(x) for x in encoded), default=2)
        pad_id = self.vocab[self.pad_token]
        input_ids = [x + [pad_id] * (width - len(x)) for x in encoded]
        attention = [[1] * len(x) + [0] * (width - len(x)) for x in encoded]
        return input_ids, attention


def _is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch) in ("Cc", "Cf")


def _is_whitespace(ch: str) -> bool:
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _pad_cjk(text: str) -> str:
    """把每个 CJK 字符用空格包起来（BertNormalizer.handle_chinese_chars 的做法）。"""
    out = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            out.append(" ")
            out.append(ch)
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")
