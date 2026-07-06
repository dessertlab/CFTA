from __future__ import annotations

import os
import re
import time

def parse_float_0_1(text):
    text = (text or "").strip()
    m = re.search(r"(?:0(?:\.\d+)?|1(?:\.0+)?)$", text) or re.search(r"(0\.\d+|1\.0+|1|0)", text)
    return float(m.group(0)) if m else None

def parse_float_direct(text):
    try:
        return float((text or "").strip())
    except Exception:
        return None

def parse_float_regex(text):
    m = re.search(r"0\.\d+|1\.0+|1|0", (text or "").strip())
    return float(m.group(0)) if m else None

def extract_scores_multiline(text, n_expected):
    lines = (text or "").strip().split("\n")
    out = []
    for line in lines:
        m = re.search(r"[-+]?\d*\.\d+|\d+", line)
        if m:
            try:
                out.append(max(0.0, min(1.0, float(m.group(0)))))
            except Exception:
                out.append(None)
        else:
            out.append(None)
    while len(out) < n_expected:
        out.append(None)
    return out[:n_expected]

class Scorer:
    kind = "base"

    def __init__(self, key, out_columns, file_prefix, batch_size=1, delay=0.0):
        self.key = key
        self.out_columns = out_columns
        self.file_prefix = file_prefix
        self.batch_size = batch_size
        self.delay = delay
        self._ready = False

    def filename(self, s):
        return f"{self.file_prefix}_{s}.csv"

    def prepare(self):
        self._ready = True

    def score(self, texts):
        raise NotImplementedError

class HFScorer(Scorer):
    kind = "hf"

    def __init__(self, key, model_id, out_columns, logit_indices, activation,
                 file_prefix, hf_login=False, batch_size=16, max_length=128):
        super().__init__(key, out_columns, file_prefix, batch_size=batch_size, delay=0.0)
        self.model_id = model_id
        self.logit_indices = logit_indices
        self.activation = activation
        self.hf_login = hf_login
        self.max_length = max_length

    def prepare(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if self.hf_login and os.getenv("HF_TOKEN"):
            try:
                from huggingface_hub import login
                login(token=os.getenv("HF_TOKEN"))
            except Exception:
                pass
        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device).eval()
        self._ready = True

    def score(self, texts):
        import gc
        torch = self._torch
        try:
            enc = self.tokenizer(list(texts), return_tensors="pt", padding=True,
                                  truncation=True, max_length=self.max_length)
            with torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
                if self.activation == "sigmoid":
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=-1)
                probs = probs.cpu().numpy()
            return [[float(probs[i][idx]) for idx in self.logit_indices] for i in range(len(texts))]
        finally:
            try:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

class _ApiError(Exception):
    def __init__(self, msg, retryable=True, rate=False):
        super().__init__(msg)
        self.retryable = retryable
        self.rate = rate

class ApiScorer(Scorer):
    kind = "api"

    def __init__(self, key, model_id, file_prefix, transport, mode, system_prompt,
                 user_template=None, batch_user_builder=None, parser=None,
                 multiline_parser=None, base_url=None, url=None,
                 temperature=0.0, max_tokens=None, timeout=60, delay=0.0,
                 batch_size=1, max_retries=1, model_env=None, groq=False,
                 fallback=None, backoff=0.5):
        super().__init__(key, ["sentiment_score"], file_prefix, batch_size=batch_size, delay=delay)
        self.model_id = model_id
        self.transport = transport
        self.mode = mode
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.batch_user_builder = batch_user_builder
        self.parser = parser
        self.multiline_parser = multiline_parser
        self.base_url = base_url
        self.url = url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.model_env = model_env
        self.groq = groq
        self.fallback = fallback
        self.backoff = backoff
        self._fallback_count = 0

    def _api_keys(self):
        if self.groq:
            raw = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        else:
            raw = os.getenv("OPENAI_API_KEYS") or os.getenv("OPENAI_API_KEY", "")
        keys = [k.strip() for k in re.split(r"[,\s]+", raw or "") if k.strip()]
        if not keys:
            raise RuntimeError(
                f"Missing API key for '{self.key}'. Set "
                + ("GROQ_API_KEY/GROQ_API_KEYS (or OPENAI_API_KEY)." if self.groq
                   else "OPENAI_API_KEY/OPENAI_API_KEYS.")
            )
        return keys

    def prepare(self):
        self.model_id = os.getenv(self.model_env, self.model_id) if self.model_env else self.model_id
        self._keys = self._api_keys()
        self._key_idx = 0
        if self.transport == "openai":
            from openai import OpenAI
            self._clients = [
                (OpenAI(base_url=self.base_url, api_key=k) if self.base_url else OpenAI(api_key=k))
                for k in self._keys
            ]
        else:
            import requests
            self._requests = requests
            self._headers_pool = [
                {"Authorization": f"Bearer {k}", "Content-Type": "application/json"} for k in self._keys
            ]
        self._activate_key()
        self._ready = True

    def _activate_key(self):
        i = self._key_idx % len(self._keys)
        if self.transport == "openai":
            self.client = self._clients[i]
        else:
            self._headers = self._headers_pool[i]

    def _rotate_key(self):
        if len(self._keys) > 1:
            self._key_idx += 1
            self._activate_key()

    def _chat(self, messages):
        if self.transport == "openai":
            kw = dict(model=self.model_id, messages=messages, temperature=self.temperature, timeout=self.timeout)
            if self.max_tokens is not None:
                kw["max_tokens"] = self.max_tokens
            try:
                resp = self.client.chat.completions.create(**kw)
            except Exception as e:
                name = type(e).__name__.lower()
                msg = str(e).lower()
                rate = (("ratelimit" in name) or ("429" in msg) or ("rate limit" in msg)
                        or ("quota" in msg) or ("authentication" in name)
                        or ("permission" in name) or ("401" in msg) or ("403" in msg))
                retryable = (rate or ("timeout" in name) or ("connection" in name)
                             or ("internalserver" in name) or ("apierror" in name)
                             or ("500" in msg) or ("502" in msg) or ("503" in msg))
                raise _ApiError(str(e), retryable=retryable, rate=rate)
            return (resp.choices[0].message.content or "").strip()
        payload = {"model": self.model_id, "messages": messages, "temperature": self.temperature}
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        try:
            r = self._requests.post(self.url, headers=self._headers, json=payload, timeout=self.timeout)
        except Exception as e:
            raise _ApiError(f"request error: {e}", retryable=True, rate=False)
        if r.status_code == 200:
            try:
                return (r.json()["choices"][0]["message"]["content"] or "").strip()
            except Exception as e:
                raise _ApiError(f"bad 200 body: {e}", retryable=False, rate=False)
        rate = r.status_code in (401, 403, 429)
        retryable = rate or r.status_code >= 500
        raise _ApiError(f"HTTP {r.status_code}", retryable=retryable, rate=rate)

    def _messages_one(self, sentence):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(sentence=sentence)},
        ]

    def _sleep_backoff(self, attempt):
        import random
        wait = min(self.backoff * (2 ** attempt), 20.0) + random.uniform(0, 0.4)
        time.sleep(max(self.delay, wait))

    def _request_content(self, messages):
        self._rotate_key()
        for attempt in range(max(1, self.max_retries)):
            try:
                return self._chat(messages)
            except _ApiError as e:
                if e.rate or not e.retryable:
                    self._rotate_key()
            except Exception:
                self._rotate_key()
            if attempt < self.max_retries - 1:
                self._sleep_backoff(attempt)
        return None

    def _parse_or_fallback(self, content):
        if content is None:
            return None
        v = self.parser(content)
        if v is not None:
            return v
        if self.fallback is not None and not os.getenv("SA_NO_FALLBACK"):
            self._fallback_count += 1
            if self._fallback_count <= 5 or self._fallback_count % 50 == 0:
                print(f"        [fallback] '{self.key}': unparseable reply "
                      f"{content!r} -> {self.fallback} (count={self._fallback_count})")
            return self.fallback
        return None

    def _score_one(self, sentence):
        return self._parse_or_fallback(self._request_content(self._messages_one(sentence)))

    def _score_batched(self, sentences):
        user = self.batch_user_builder(sentences)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user}]
        content = self._request_content(messages)
        if content is None:
            return [None] * len(sentences)
        vals = self.multiline_parser(content, len(sentences))
        if self.fallback is not None and not os.getenv("SA_NO_FALLBACK"):
            n_fb = sum(1 for v in vals if v is None)
            if n_fb:
                self._fallback_count += n_fb
            vals = [self.fallback if v is None else v for v in vals]
        return vals

    def score(self, texts):
        texts = list(texts)
        if self.mode == "batched":
            vals = self._score_batched(texts)
            return [[v] if v is not None else None for v in vals]
        out = []
        for t in texts:
            v = self._score_one(t)
            out.append([v] if v is not None else None)
        return out

def _build_registry():
    reg = {}

    reg["cardiffnlp_twitter_roberta_base_sentiment_latest"] = HFScorer(
        "cardiffnlp_twitter_roberta_base_sentiment_latest",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        ["sentiment_score_negative", "sentiment_score_neutral", "sentiment_score_positive"],
        [0, 1, 2], "softmax",
        "dataset_with_sentiment_scores_robertaTwitter", hf_login=True)
    reg["distilbert_base_uncased_finetuned_sst2_english"] = HFScorer(
        "distilbert_base_uncased_finetuned_sst2_english",
        "distilbert-base-uncased-finetuned-sst-2-english",
        ["sentiment_score_negative", "sentiment_score_positive"], [0, 1], "softmax",
        "dataset_with_sentiment_scores_distilbert")
    reg["siebert_sentiment_roberta_large_english"] = HFScorer(
        "siebert_sentiment_roberta_large_english",
        "siebert/sentiment-roberta-large-english",
        ["sentiment_score"], [1], "softmax",
        "dataset_with_sentiment_scores_siebert", hf_login=True)
    reg["textattack_bert_base_uncased_SST_2"] = HFScorer(
        "textattack_bert_base_uncased_SST_2",
        "textattack/bert-base-uncased-SST-2",
        ["sentiment_score_negative", "sentiment_score_positive"], [0, 1], "softmax",
        "dataset_with_sentiment_scores_bertbase")
    reg["unitary_toxic_bert"] = HFScorer(
        "unitary_toxic_bert", "unitary/toxic-bert",
        ["toxicity_score"], [0], "sigmoid",
        "sentiment_score_toxicbert")

    reg["gpt_3_5_turbo"] = ApiScorer(
        "gpt_3_5_turbo", "gpt-3.5-turbo", None,
        transport="openai", mode="batched",
        system_prompt="You are a sentiment analysis model. For each sentence, respond only with a float between 0 and 1 (no text, no explanation), one per line.",
        batch_user_builder=lambda sents: "Return one float between 0 and 1 for each sentence below, in order, one per line. No extra text:\n\n"
                                         + "\n".join(f"{i+1}. {s.strip()}" for i, s in enumerate(sents)),
        multiline_parser=extract_scores_multiline,
        temperature=0.0, max_tokens=200, timeout=60,
        batch_size=20, delay=1.5, max_retries=3, model_env="OPENAI_MODEL")

    reg["allam_2_7b"] = ApiScorer(
        "allam_2_7b", "allam-2-7b", None,
        transport="openai", mode="per_sentence", groq=True,
        base_url="https://api.groq.com/openai/v1",
        system_prompt="Respond only with a float between 0 and 1 for the sentiment positivity of the sentence.",
        user_template="Sentence: {sentence}\nSentiment score (0=negative, 1=positive):",
        parser=parse_float_0_1, temperature=0.0, max_tokens=8, timeout=30,
        delay=0.05, max_retries=4, model_env="GROQ_MODEL")
    reg["llama_3_1_8b_instant"] = ApiScorer(
        "llama_3_1_8b_instant", "llama-3.1-8b-instant", None,
        transport="openai", mode="per_sentence", groq=True,
        base_url="https://api.groq.com/openai/v1",
        system_prompt="Respond only with a single number between 0 and 1 (inclusive) representing the sentence's positive sentiment.",
        user_template="Sentence: {sentence}\nSentiment score (0=negative, 1=positive):",
        parser=parse_float_0_1, temperature=0.0, max_tokens=8, timeout=30,
        delay=0.05, max_retries=6, fallback=0.5)

    reg["gemma2_9b_it"] = ApiScorer(
        "gemma2_9b_it", "gemma2-9b-it", None,
        transport="requests", mode="per_sentence", groq=True,
        url="https://api.groq.com/openai/v1/chat/completions",
        system_prompt=("You are a helpful assistant that performs sentiment analysis. "
                       "For each input, provide only a single positive sentiment score between 0 and 1. "
                       "Respond only with the number (e.g., 0.85)."),
        user_template="Sentence: {sentence}",
        parser=parse_float_direct, temperature=0.0, max_tokens=None, timeout=60,
        delay=1.5, max_retries=4)
    reg["llama3_70b"] = ApiScorer(
        "llama3_70b", "llama3-70b-8192", None,
        transport="requests", mode="per_sentence", groq=True,
        url="https://api.groq.com/openai/v1/chat/completions",
        system_prompt=("You are a sentiment analysis model. "
                       "Respond only with a single float between 0 and 1 representing positivity."),
        user_template=('Analyze the following sentence and return only a decimal number between 0 and 1 '
                       '(at least two decimal places). No text, no explanation.\n\nSentence: "{sentence}"'),
        parser=parse_float_regex, temperature=0.0, max_tokens=8, timeout=60,
        delay=2.0, max_retries=4)

    reg["gpt_3_5_turbo"].file_prefix = "generated_sentences_PLACEHOLDER_with_sentiment_gpt-3.5-turbo"
    reg["allam_2_7b"].file_prefix = "generated_sentences_PLACEHOLDER_with_sentiment_allam-2-7b"
    reg["llama_3_1_8b_instant"].file_prefix = "generated_sentences_PLACEHOLDER_with_sentiment_llama-3.1-8b-instant"
    reg["gemma2_9b_it"].file_prefix = "generated_sentences_PLACEHOLDER_with_sentiment_gemma2-9b-it"
    reg["llama3_70b"].file_prefix = "generated_sentences_PLACEHOLDER_with_sentiment_llama3-70b"
    return reg

REGISTRY = _build_registry()

def _api_filename(self, s):
    return self.file_prefix.replace("PLACEHOLDER", s) + ".csv"

for _sc in REGISTRY.values():
    if isinstance(_sc, ApiScorer):
        import types
        _sc.filename = types.MethodType(_api_filename, _sc)

ALIASES = {
    "cardiff": "cardiffnlp_twitter_roberta_base_sentiment_latest",
    "cardiffnlp": "cardiffnlp_twitter_roberta_base_sentiment_latest",
    "robertatwitter": "cardiffnlp_twitter_roberta_base_sentiment_latest",
    "distilbert": "distilbert_base_uncased_finetuned_sst2_english",
    "siebert": "siebert_sentiment_roberta_large_english",
    "bertbase": "textattack_bert_base_uncased_SST_2",
    "textattack": "textattack_bert_base_uncased_SST_2",
    "toxic": "unitary_toxic_bert",
    "toxicbert": "unitary_toxic_bert",
    "unitary": "unitary_toxic_bert",
    "gpt": "gpt_3_5_turbo",
    "gpt35": "gpt_3_5_turbo",
    "allam": "allam_2_7b",
    "gemma2": "gemma2_9b_it",
    "gemma": "gemma2_9b_it",
    "llama8b": "llama_3_1_8b_instant",
    "llama-3.1": "llama_3_1_8b_instant",
    "llama70b": "llama3_70b",
    "llama3-70b": "llama3_70b",
}

HF_KEYS = [k for k, v in REGISTRY.items() if v.kind == "hf"]
API_KEYS = [k for k, v in REGISTRY.items() if v.kind == "api"]

def resolve_models(spec):
    if spec is None:
        raise SystemExit("No --model given. Use a key, alias, 'all', 'hf' or 'api'. See --list.")
    spec = str(spec).strip().lower()
    if spec == "all":
        return list(REGISTRY.keys())
    if spec == "hf":
        return list(HF_KEYS)
    if spec == "api":
        return list(API_KEYS)
    keys = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in REGISTRY:
            keys.append(tok)
        elif tok in ALIASES:
            keys.append(ALIASES[tok])
        else:
            raise SystemExit(f"Unknown model '{tok}'. Run with --list to see options.")
    return keys

def get_scorer(key):
    if key not in REGISTRY:
        raise SystemExit(f"Unknown model key '{key}'.")
    return REGISTRY[key]
