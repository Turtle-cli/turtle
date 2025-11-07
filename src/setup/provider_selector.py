import os
import sys
import termios
import tty
from typing import List, Dict, Optional, Tuple


class ProviderSelector:

    def __init__(self):
        self.providers = self._load_providers()
        self.current_index = 0
        self.search_query = ""
        self.filtered_providers = self.providers.copy()
        self.items_per_page = 10
        self.current_page = 0

    def _load_providers(self) -> List[Dict[str, str]]:
        return [
            {
                "id": "openai",
                "name": "OpenAI",
                "description": "GPT models including GPT-4, GPT-3.5-turbo",
                "tier": "paid",
                "popular_models": "gpt-4, gpt-4-turbo, gpt-3.5-turbo"
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "description": "Claude models for advanced reasoning",
                "tier": "paid",
                "popular_models": "claude-3-opus, claude-3-sonnet, claude-3-haiku"
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "description": "Google's multimodal AI models",
                "tier": "free",
                "popular_models": "gemini-pro, gemini-pro-vision"
            },
            {
                "id": "vertex_ai",
                "name": "Vertex AI",
                "description": "Google's enterprise AI platform",
                "tier": "enterprise",
                "popular_models": "gemini-pro, palm-2"
            },
            {
                "id": "azure",
                "name": "Azure OpenAI",
                "description": "Microsoft's hosted OpenAI models",
                "tier": "enterprise",
                "popular_models": "gpt-4, gpt-35-turbo"
            },
            {
                "id": "cohere",
                "name": "Cohere",
                "description": "Enterprise language models",
                "tier": "paid",
                "popular_models": "command, command-r, embed"
            },
            {
                "id": "huggingface",
                "name": "Hugging Face",
                "description": "Open source and custom models",
                "tier": "free",
                "popular_models": "mistral-7b, llama-2, codellama"
            },
            {
                "id": "groq",
                "name": "Groq",
                "description": "Ultra-fast inference for LLMs",
                "tier": "free",
                "popular_models": "llama2-70b, mixtral-8x7b"
            },
            {
                "id": "ollama",
                "name": "Ollama",
                "description": "Run models locally on your machine",
                "tier": "free",
                "popular_models": "llama2, mistral, codellama"
            },
            {
                "id": "mistral",
                "name": "Mistral AI",
                "description": "High-performance open models",
                "tier": "paid",
                "popular_models": "mistral-large, mistral-medium, mistral-small"
            },
            {
                "id": "perplexity",
                "name": "Perplexity AI",
                "description": "Search-augmented language models",
                "tier": "paid",
                "popular_models": "pplx-7b-online, pplx-70b-online"
            },
            {
                "id": "fireworks",
                "name": "Fireworks AI",
                "description": "Fast inference for open source models",
                "tier": "paid",
                "popular_models": "llama-v2-70b, mixtral-8x7b"
            },
            {
                "id": "together",
                "name": "Together AI",
                "description": "Distributed inference platform",
                "tier": "paid",
                "popular_models": "llama-2-70b, falcon-40b"
            },
            {
                "id": "replicate",
                "name": "Replicate",
                "description": "Run ML models via API",
                "tier": "paid",
                "popular_models": "llama-2, stable-diffusion, whisper"
            },
            {
                "id": "anyscale",
                "name": "Anyscale Endpoints",
                "description": "Scalable model serving",
                "tier": "paid",
                "popular_models": "llama-2-70b, mistral-7b"
            },
            {
                "id": "deepinfra",
                "name": "DeepInfra",
                "description": "Serverless inference for open models",
                "tier": "paid",
                "popular_models": "llama2-70b, code-llama-34b"
            },
            {
                "id": "palm",
                "name": "Google PaLM",
                "description": "Google's Pathways Language Model",
                "tier": "paid",
                "popular_models": "palm-2, palm-2-chat"
            },
            {
                "id": "ai21",
                "name": "AI21 Labs",
                "description": "Jurassic language models",
                "tier": "paid",
                "popular_models": "j2-ultra, j2-mid, j2-light"
            },
            {
                "id": "nlpcloud",
                "name": "NLP Cloud",
                "description": "Production-ready NLP API",
                "tier": "paid",
                "popular_models": "chatdolphin, finetuned-llama-2"
            },
            {
                "id": "aleph_alpha",
                "name": "Aleph Alpha",
                "description": "European AI models",
                "tier": "paid",
                "popular_models": "luminous-supreme, luminous-extended"
            }
        ]

    def _get_tier_indicator(self, tier: str) -> str:
        indicators = {
            "free": "[FREE]",
            "paid": "[PAID]",
            "enterprise": "[ENT]"
        }
        return indicators.get(tier, "[PAID]")

    def _filter_providers(self):
        if not self.search_query:
            self.filtered_providers = self.providers.copy()
        else:
            query = self.search_query.lower()
            self.filtered_providers = [
                p for p in self.providers
                if query in p["name"].lower() or
                   query in p["description"].lower() or
                   query in p["id"].lower()
            ]

        self.current_index = min(self.current_index, len(self.filtered_providers) - 1)
        if self.current_index < 0:
            self.current_index = 0

        total_pages = (len(self.filtered_providers) - 1) // self.items_per_page
        self.current_page = min(self.current_page, total_pages)

    def _get_current_page_items(self) -> Tuple[List[Dict[str, str]], int]:
        start_idx = self.current_page * self.items_per_page
        end_idx = start_idx + self.items_per_page
        page_items = self.filtered_providers[start_idx:end_idx]

        global_current_index = self.current_index
        local_current_index = global_current_index - start_idx

        if local_current_index < 0 or local_current_index >= len(page_items):
            local_current_index = -1

        return page_items, local_current_index

    def _clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')

    def _display_header(self):
        print("=" * 80)
        print("LiteLLM Provider Selection")
        print("=" * 80)
        print("Use arrow keys to navigate, type to search, Enter to select, Esc/q to quit")
        print()

    def _display_search_bar(self):
        if self.search_query:
            print(f"Search: {self.search_query}")
        else:
            print("Search: (type to filter providers)")
        print("-" * 40)
        print()

    def _display_providers(self):
        page_items, local_current_index = self._get_current_page_items()

        if not page_items:
            print("No providers found matching your search.")
            return

        for i, provider in enumerate(page_items):
            prefix = "> " if i == local_current_index else "  "
            tier_indicator = self._get_tier_indicator(provider["tier"])

            print(f"{prefix}{provider['name']} {tier_indicator}")
            print(f"   {provider['description']}")
            print(f"   Models: {provider['popular_models']}")
            print()

    def _display_pagination(self):
        if len(self.filtered_providers) <= self.items_per_page:
            return

        total_pages = (len(self.filtered_providers) - 1) // self.items_per_page + 1
        current_page_num = self.current_page + 1

        print("-" * 80)
        print(f"Page {current_page_num} of {total_pages} | Total: {len(self.filtered_providers)} providers")
        print("Use Left/Right arrows to change pages")

    def _get_key_input(self) -> str:
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setraw(fd)

            key = sys.stdin.read(1)

            if key == '\x1b':
                key += sys.stdin.read(2)

            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return key
        except:
            return input()

    def _handle_navigation(self, key: str) -> bool:
        if key == '\x1b[A':  # Up arrow
            if self.current_index > 0:
                self.current_index -= 1
                if self.current_index < self.current_page * self.items_per_page:
                    self.current_page = max(0, self.current_page - 1)
        elif key == '\x1b[B':  # Down arrow
            if self.current_index < len(self.filtered_providers) - 1:
                self.current_index += 1
                if self.current_index >= (self.current_page + 1) * self.items_per_page:
                    total_pages = (len(self.filtered_providers) - 1) // self.items_per_page
                    self.current_page = min(total_pages, self.current_page + 1)
        elif key == '\x1b[D':  # Left arrow
            if self.current_page > 0:
                self.current_page -= 1
                self.current_index = self.current_page * self.items_per_page
        elif key == '\x1b[C':  # Right arrow
            total_pages = (len(self.filtered_providers) - 1) // self.items_per_page
            if self.current_page < total_pages:
                self.current_page += 1
                self.current_index = min(
                    self.current_page * self.items_per_page,
                    len(self.filtered_providers) - 1
                )
        elif key == '\r' or key == '\n':  # Enter
            if self.filtered_providers and 0 <= self.current_index < len(self.filtered_providers):
                return True
        elif key == '\x1b' or key == 'q':  # Esc or q
            return None
        elif key == '\x7f':  # Backspace
            if self.search_query:
                self.search_query = self.search_query[:-1]
                self._filter_providers()
        elif key.isprintable() and len(key) == 1:
            self.search_query += key
            self._filter_providers()

        return False

    def select_provider(self) -> Optional[str]:
        try:
            while True:
                self._clear_screen()
                self._display_header()
                self._display_search_bar()
                self._display_providers()
                self._display_pagination()

                key = self._get_key_input()
                result = self._handle_navigation(key)

                if result is True:
                    if self.filtered_providers:
                        return self.filtered_providers[self.current_index]["id"]
                elif result is None:
                    return None

        except KeyboardInterrupt:
            return None
        except Exception:
            print("\nError: Unable to use interactive mode. Falling back to simple selection.")
            return self._fallback_selection()

    def _fallback_selection(self) -> Optional[str]:
        print("\nAvailable providers:")
        print("-" * 40)

        for i, provider in enumerate(self.providers[:10], 1):
            tier_indicator = self._get_tier_indicator(provider["tier"])
            print(f"{i:2d}. {provider['name']} {tier_indicator}")
            print(f"    {provider['description']}")

        if len(self.providers) > 10:
            print(f"\n... and {len(self.providers) - 10} more providers")

        try:
            choice = input(f"\nEnter provider number (1-{min(10, len(self.providers))}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < min(10, len(self.providers)):
                return self.providers[idx]["id"]
        except ValueError:
            pass

        print("Invalid selection.")
        return None

    def get_provider_info(self, provider_id: str) -> Optional[Dict[str, str]]:
        for provider in self.providers:
            if provider["id"] == provider_id:
                return provider
        return None