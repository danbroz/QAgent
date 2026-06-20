from openai import OpenAI,RateLimitError
from huggingface_hub import InferenceClient
import os
import requests
import shutil
import subprocess
import time
import random

class LLM_model:
    OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4")
    MODEL_PROVIDER_PREFIXES = ("codex", "openai", "nebius", "nscale", "huggingface")

    def __init__(
        self,
        llm_choice='codex:gpt-5.5',
        llm_key='',
        temp=None,
        provider='auto',
        reasoning_effort=None,
        codex_timeout=600,
    ):
        provider, llm_choice = self._parse_provider_prefix(provider, llm_choice)
        self.llm_choice = llm_choice or 'gpt-5.5'
        self.provider = self._resolve_provider(provider, self.llm_choice)
        self.llm_key = llm_key or (os.getenv("OPENAI_API_KEY", "") if self.provider == "openai" else "")
        self.use_openai = self.provider == "openai"
        self.use_codex = self.provider == "codex"
        self.temp = temp
        self.reasoning_effort = reasoning_effort
        self.codex_timeout = codex_timeout
        
        self.backoff_factor =3
        self.jitter_factor = 1

        if self.provider == 'codex':
            print('using provider CODEX CLI')
            self.client = None
            self.codex_bin = shutil.which(os.getenv("CODEX_CLI", "codex"))
        elif self.provider == 'openai':
            print('using provider OPENAI')
            self.client = OpenAI(api_key=self.llm_key) if self.llm_key else None
        elif self.provider == 'nscale':
            print('using provider NSCALE')
            self.client = OpenAI(
                base_url="https://inference.api.nscale.com/v1",
                api_key=self.llm_key,
            )
        elif self.provider == 'huggingface':
            print('using provider HuggingFace')
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=self.llm_key,
            )
        elif self.provider == 'nebius':
            print('using provider NEBIUS')
            self.client = OpenAI(
                base_url="https://api.studio.nebius.com/v1/",
                api_key=self.llm_key,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

        self._snap_prompt = 0
        self._snap_completion = 0
        self._snap_total = 0

    @classmethod
    def _is_openai_model(cls, llm_choice):
        model = (llm_choice or "").lower()
        return model.startswith(cls.OPENAI_MODEL_PREFIXES)

    @classmethod
    def _parse_provider_prefix(cls, provider, llm_choice):
        choice = llm_choice or ""
        if ":" not in choice:
            return provider, llm_choice

        prefix, model = choice.split(":", 1)
        prefix = prefix.lower()
        if prefix in cls.MODEL_PROVIDER_PREFIXES:
            return prefix, model or None
        return provider, llm_choice

    @classmethod
    def _resolve_provider(cls, provider, llm_choice):
        provider = (provider or "auto").lower()
        if provider == "auto":
            return "codex" if cls._is_openai_model(llm_choice) else "nebius"
        return provider

    @staticmethod
    def _usage_value(usage, *names):
        for name in names:
            if isinstance(usage, dict):
                value = usage.get(name)
            else:
                value = getattr(usage, name, None)
            if value is not None:
                return value
        return 0

    def _record_usage(self, response):
        try:
            usage = getattr(response, "usage", None)
            if usage is None and isinstance(response, dict):
                usage = response.get("usage")
            if usage:
                p = self._usage_value(usage, "prompt_tokens", "input_tokens")
                c = self._usage_value(usage, "completion_tokens", "output_tokens")
                t = self._usage_value(usage, "total_tokens") or (p or 0) + (c or 0)
                self.total_prompt_tokens += int(p or 0)
                self.total_completion_tokens += int(c or 0)
                self.total_tokens += int(t or 0)
        except Exception:
            pass

    def snapshot_usage(self):
        self._snap_prompt = self.total_prompt_tokens
        self._snap_completion = self.total_completion_tokens
        self._snap_total = self.total_tokens

    def usage_since_snapshot(self):
        return {
            "prompt": self.total_prompt_tokens - self._snap_prompt,
            "completion": self.total_completion_tokens - self._snap_completion,
            "total": self.total_tokens - self._snap_total,
        }

    def generate(self, prompt, system_prompt, max_tokens=200):
        if self.use_codex:
            out, resp = self._generate_codex(prompt, system_prompt, max_tokens)
        elif self.use_openai:
            out, resp = self._generate_openai(prompt, system_prompt, max_tokens)
        else:
            out, resp = self._generate_compatible_chat(prompt, system_prompt, max_tokens)
        self._record_usage(resp)
        return out

    def _retry_sleep(self, attempt):
        sleep_time = self.backoff_factor * min(attempt + 1, 10) * random.uniform(0.5, 1.5)
        print(f"Rate limit error. Retrying in {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)

    @staticmethod
    def _content_part_text(part):
        if isinstance(part, dict):
            return part.get("text") or ""
        return getattr(part, "text", "") or ""

    @classmethod
    def _extract_responses_text(cls, response):
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        if isinstance(response, dict):
            text = response.get("output_text")
            if text:
                return text.strip()
            output = response.get("output", [])
        else:
            output = getattr(response, "output", [])

        chunks = []
        for item in output or []:
            content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
            for part in content or []:
                part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if part_type in (None, "output_text", "text"):
                    chunks.append(cls._content_part_text(part))

        return "".join(chunks).strip()

    def _build_codex_prompt(self, prompt, system_prompt, max_tokens):
        return f"""You are being used as a non-interactive text generation backend for QAgent.
Do not inspect files, run shell commands, edit files, or use tools. Answer only from the instructions and user input below.
Return only the requested final content with no progress notes.
If the requested output is code, JSON, or QASM, return only that artifact.
Keep the response within approximately {max_tokens} output tokens.

Developer instructions:
{system_prompt}

User input:
{prompt}
"""

    def _generate_codex(self, prompt, system_prompt, max_tokens=200):
        if not self.codex_bin:
            return "Error: Codex CLI not found. Install Codex or set CODEX_CLI to its executable path.", {"usage": None}

        cmd = [
            self.codex_bin,
            "exec",
            "--ephemeral",
            "--color",
            "never",
            "--sandbox",
            "read-only",
            "--ask-for-approval",
            "never",
            "-C",
            os.getcwd(),
        ]
        if self.llm_choice:
            cmd.extend(["--model", self.llm_choice])
        cmd.append("-")

        try:
            completed = subprocess.run(
                cmd,
                input=self._build_codex_prompt(prompt, system_prompt, max_tokens),
                text=True,
                capture_output=True,
                timeout=self.codex_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"Error: Codex CLI timed out after {self.codex_timeout} seconds.", {"usage": None}
        except Exception as e:
            return f"Error during Codex CLI call: {str(e)}", {"usage": None}

        if completed.returncode != 0:
            err = completed.stderr.strip() or completed.stdout.strip()
            return f"Error: Codex CLI exited with status {completed.returncode}: {err}", {"usage": None}

        return completed.stdout.strip(), {"usage": None}

    def _generate_openai(self, prompt, system_prompt, max_tokens=200):
        if not self.llm_key:
            return "Error: missing OpenAI API key. Set OPENAI_API_KEY or pass llm_key.", {"usage": None}

        kwargs = {
            "model": self.llm_choice,
            "instructions": system_prompt,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        if self.temp is not None:
            kwargs["temperature"] = self.temp
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}

        attempt = 0
        while True:
            try:
                if self.client is not None and hasattr(self.client, "responses"):
                    response = self.client.responses.create(**kwargs)
                    return self._extract_responses_text(response), response

                response = requests.post(
                    "https://api.openai.com/v1/responses",
                    json=kwargs,
                    headers={
                        "Authorization": f"Bearer {self.llm_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=300,
                )

                if response.status_code == 200:
                    j = response.json()
                    return self._extract_responses_text(j), j
                if response.status_code == 429:
                    self._retry_sleep(attempt)
                    attempt += 1
                    continue

                return f"Error: {response.status_code}, {response.text}", {"usage": None}
            except RateLimitError:
                self._retry_sleep(attempt)
                attempt += 1
            except Exception as e:
                return f"Error during OpenAI Responses call: {str(e)}", {"usage": None}


    def _generate_compatible_chat(self, prompt, system_prompt, max_tokens=200):
        kwargs = dict(
            model=self.llm_choice,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
        )
        if self.temp is not None:
            kwargs["temperature"] = self.temp
            
        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content.strip(), response
            except RateLimitError:
                self._retry_sleep(attempt)
                attempt += 1
            except Exception as e:
                return f"Error during chat call: {str(e)}", {"usage": None}

if __name__ == "__main__":
    pass
