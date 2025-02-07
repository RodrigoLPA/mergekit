import json
import math
import os
import random
import time
import typing
from typing import List, Optional

import numpy as np
import requests
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
)
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

load_dotenv()

WANDB_TOKEN = os.environ.get("WANDB_API_KEY", None)


class SubsetLoader(IterableDataset):
    """Base class for data-specific subset loader classes.

    Handles core dataset loading functionality like:
    - Fetching pages of data from HF datasets API
    - Tokenizing text into sequences
    - Buffering and padding sequences
    - Batching samples for training

    Args:
        batch_size (int, optional): Size of batches to return
        sequence_length (int, optional): Length of sequences after padding
        num_pages (int, optional): Number of pages to fetch
        tokenizer (AutoTokenizer, optional): Tokenizer to use
        pack_samples (bool): Whether to pack sequences without padding
        random_seed (int, optional): Random seed for reproducibility
        config (str): Dataset config name
        split (str): Dataset split name
        requires_auth (bool): Whether HF auth token is needed
        add_bos (bool): Whether to add BOS token to the sequences
    """

    name: str = None  # Dataset name
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"
    max_pages: int = None

    def __init__(
        self,
        batch_size=None,
        sequence_length=None,
        num_pages=None,
        tokenizer: AutoTokenizer = None,
        pack_samples: bool = False,
        random_seed: typing.Optional[int] = None,
        config: str = "default",
        split: str = "train",
        requires_auth: bool = False,
        add_bos: bool = True,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_pages = num_pages
        self.tokenizer = tokenizer
        self.pack_samples = pack_samples
        self.config = config
        self.split = split
        self.requires_auth = requires_auth

        # Configure tokenizer based on add_bos
        if add_bos:
            self.tokenizer.add_special_tokens({"bos_token": self.tokenizer.bos_token})

        # Initialize with seed if provided
        if random_seed is not None:
            random.seed(random_seed)

        self.num_rows_per_page = 50
        self.duplicate_page_threshold = 100
        self.retry_limit = 10
        self.retry_delay = 5

        # Buffers
        self.buffer = []
        self.used_buffer = []
        self.padded_buffer = []

        # Get HF token if needed
        self.hf_token = None
        if self.requires_auth:
            self.hf_token = os.getenv("HF_TOKEN")
            if not self.hf_token:
                raise ValueError("HF_TOKEN environment variable not found")

        # Initialize request params
        self.params = self._get_default_params()

        # Fetch pages if specified
        if self.num_pages:
            self._initialize_pages()

    def _get_default_params(self):
        """Get default request parameters. Override if needed."""
        return {
            "dataset": self.name,
            "config": self.config,
            "split": self.split,
        }

    def _get_request_headers(self):
        """Get request headers. Override if needed."""
        headers = {}
        if self.requires_auth:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers

    def _initialize_pages(self):
        """Initialize pages based on loader type"""
        if hasattr(self, "fetch_dataset_configs"):
            # For FineWebEdu2 style loaders
            self.configs_data = self.fetch_dataset_configs()
            self._fetch_data_to_buffer(self.num_pages)
        else:
            # For simple page-based loaders
            pages = self._sample_pages()
            self.fetch_data_for_pages(pages)

    def fetch_data_for_pages(self, pages):
        """Set the pages and fetch their data to the buffer."""
        self.pages = pages
        self.buffer = []
        for page in self.pages:
            self._fetch_data_for_page(page)

    def _fetch_data_for_page(self, page):
        """Fetch data for a single page"""
        # Handle different page types (tuple vs int)
        if isinstance(page, tuple):
            config_name, page_num, split = page
            self.params.update(
                {
                    "config": config_name,
                    "split": split,
                    "offset": page_num,
                }
            )
        else:
            self.params["offset"] = page

        self.params["length"] = self.num_rows_per_page

        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = requests.get(
                    self.rows_base_url,
                    params=self.params,
                    headers=self._get_request_headers(),
                )
                response.raise_for_status()

                for row in response.json()["rows"]:
                    content = self._get_content_from_row(row)
                    input_ids = self.tokenizer(
                        content, truncation=True, add_special_tokens=True
                    )["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

                break

            except requests.exceptions.RequestException as e:
                attempt += 1
                print(
                    f"Failed to fetch data for page {page}, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)
                else:
                    print("Maximum retry limit reached. Unable to fetch data.")
                    raise

    def _get_content_from_row(self, row):
        """Extract content from row based on dataset format. Override if needed."""
        return row["row"].get("text", row["row"].get("content"))

    def _sample_pages(self):
        """Sample random pages. Override for custom sampling logic."""
        return [random.randint(1, self.max_pages) for _ in range(self.num_pages)]

    def get_page_names(self):
        """Get page names in consistent format"""
        if not hasattr(self, "pages"):
            return []

        if isinstance(self.pages[0], tuple):
            return [
                f"{cfg_name}_{num_rows}_{split}"
                for cfg_name, num_rows, split in self.pages
            ]
        return self.pages

    def _get_pad_size(self, input_ids):
        """Get padding size for input tokens."""
        if self.pack_samples:
            return 1

        sample_size = len(input_ids)
        remainder = sample_size % self.sequence_length
        pad_size = self.sequence_length - remainder
        pad_size = pad_size % self.sequence_length
        return pad_size

    def _refill_padded_buffer(self):
        """Refill the padded buffer from the main buffer."""
        while self.buffer and len(self.padded_buffer) < self.sequence_length:
            input_ids = []
            EOS_index = self.buffer.index(self.tokenizer.eos_token_id)
            input_ids = self.buffer[: EOS_index + 1]
            self.buffer = self.buffer[EOS_index + 1 :]
            self.used_buffer += input_ids
            self.padded_buffer += input_ids[:-1]
            self.padded_buffer += [self.tokenizer.eos_token_id] * self._get_pad_size(
                input_ids=input_ids[:-1]
            )

    def __iter__(self):
        self.buffer = self.used_buffer + self.buffer
        self.padded_buffer = []
        self._refill_padded_buffer()
        return self

    def __next__(self):
        batch = []
        while len(self.padded_buffer) >= self.sequence_length:
            batch.append(self.padded_buffer[: self.sequence_length])
            self.padded_buffer = self.padded_buffer[self.sequence_length :]
            self._refill_padded_buffer()
            if len(batch) == self.batch_size:
                return np.stack(batch)
        raise StopIteration


class SubsetPes2oXLoader(SubsetLoader):
    """Loader for the Pes2oX dataset"""

    max_pages: int = 8242000
    name: str = "laion/Pes2oX-fulltext"

    def __init__(self, **kwargs):
        super().__init__(config="pes2ov2", **kwargs)


class SubsetStackV1DedupLoader(SubsetLoader):
    """Loader for The Stack deduped dataset"""

    max_pages: int = 236655813
    name: str = "bigcode/the-stack-dedup"

    def __init__(self, **kwargs):
        super().__init__(requires_auth=True, **kwargs)


class SubsetFalconLoader(SubsetLoader):
    """Loader for the Falcon RefinedWeb dataset"""

    max_pages: int = 968000015
    name: str = "tiiuae/falcon-refinedweb"


class SubsetFineWebEdu2Loader(SubsetLoader):
    """Loader for the FineWeb Edu 2 dataset"""

    name: str = "HuggingFaceFW/fineweb-edu-score-2"

    def fetch_dataset_configs(self) -> typing.Dict[str, typing.Dict]:
        """
        Fetch dataset configs and their metadata.
        Returns a dictionary with config names as keys and metadata as values.
        """
        params = dict(dataset=self.name)

        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = requests.get(self.size_base_url, params=params)
                response.raise_for_status()

                configs_dict = response.json()["size"]["splits"]
                configs_data = {
                    entry["config"]: {
                        "num_rows": entry["num_rows"],
                        "split": entry["split"],
                    }
                    for entry in configs_dict
                    if entry["config"] != "default"
                }

                return configs_data

            except requests.exceptions.RequestException as e:
                attempt += 1
                print(
                    f"Failed to fetch dataset configs, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)
                else:
                    print("Maximum retry limit reached. Unable to fetch data.")
                    raise

    def _fetch_data_to_buffer(self, num_pages):
        """Fetch data to buffer with support for multiple configs."""
        self.pages = []
        attempts = 0
        duplicates = 0
        initial_offset = random.randint(0, self.num_rows_per_page - 1)

        while len(self.pages) < num_pages:
            page = self.get_random_pages(num_pages=1, initial_offset=initial_offset)[0]

            if page in self.pages:
                duplicates += 1
                if duplicates >= self.duplicate_page_threshold:
                    print(
                        f"Hit duplicate page threshold of {self.duplicate_page_threshold}. "
                        f"Stopping early at: {len(self.pages)} pages."
                    )
                    break
                continue

            config_name, page_row_start, split = page
            params = {
                "dataset": self.name,
                "config": config_name,
                "split": split,
                "offset": page_row_start,
                "length": self.num_rows_per_page,
            }

            try:
                response = requests.get(self.rows_base_url, params=params)
                response.raise_for_status()
                self.pages.append(page)

                for row in response.json()["rows"]:
                    content = row["row"]["text"]
                    input_ids = self.tokenizer(
                        content, truncation=True, add_special_tokens=True
                    )["input_ids"]
                    self.buffer += input_ids
                    self.buffer += [self.tokenizer.eos_token_id]

            except requests.exceptions.RequestException as e:
                attempts += 1
                print(
                    f"Failed to fetch data, retrying. Attempt {attempts}/{self.retry_limit * num_pages}"
                )
                if attempts >= num_pages * self.retry_limit:
                    print("Maximum retry limit reached. Unable to fetch data.")
                    raise

    def get_random_pages(self, num_pages, initial_offset):
        """Get random pages across different configs."""
        pages = []
        for _ in range(num_pages):
            config_name = random.choice(list(self.configs_data.keys()))
            data_row_count = self.configs_data[config_name]["num_rows"] - initial_offset
            data_page_count = (data_row_count + 1) // self.num_rows_per_page
            selected_page_start = initial_offset + (
                random.randint(0, data_page_count - 1) * self.num_rows_per_page
            )
            split = self.configs_data[config_name]["split"]
            pages.append((config_name, selected_page_start, split))
        return pages

    def fetch_data_to_rows(self, num_pages):
        """
        Fetch data and return raw text rows instead of adding to buffer.

        Args:
            num_pages (int): Number of pages to fetch

        Returns:
            List[str]: List of text samples from the fetched pages

        Raises:
            RequestException: If data cannot be fetched after retries
        """
        downloaded_pages = set()
        rows = []
        attempts = 0
        duplicates = 0
        initial_offset = random.randint(0, self.num_rows_per_page - 1)

        while len(downloaded_pages) < num_pages:
            page = self.get_random_pages(num_pages=1, initial_offset=initial_offset)[0]

            if page in downloaded_pages:
                duplicates += 1
                if duplicates >= self.duplicate_page_threshold:
                    print(
                        f"Hit duplicate page threshold of {self.duplicate_page_threshold}. "
                        f"Stopping early at: {len(downloaded_pages)} pages."
                    )
                    break
                continue

            config_name, page_row_start, split = page
            params = {
                "dataset": self.name,
                "config": config_name,
                "split": split,
                "offset": page_row_start,
                "length": self.num_rows_per_page,
            }

            try:
                response = requests.get(self.rows_base_url, params=params)
                response.raise_for_status()
                downloaded_pages.add(page)

                for row in response.json()["rows"]:
                    rows.append(row["row"]["text"])

            except requests.exceptions.RequestException as e:
                attempts += 1
                print(
                    f"Failed to fetch data, retrying with a newly sampled page. "
                    f"Attempt {attempts}/{self.retry_limit * num_pages}"
                )
                if attempts >= num_pages * self.retry_limit:
                    print("Maximum retry limit reached. Unable to fetch data.")
                    raise

        return rows


# Todo Add stackv2 dataloader


def get_batches_from_loader(
    dataset_name: str,
    num_pages: int,
    tokenizer: AutoTokenizer,
    batch_size: int = 1,
    sequence_length: int = 2048,
    random_seed: Optional[int] = None,
    pack_samples: bool = False,
    add_bos: bool = True,
) -> List[np.ndarray]:
    """
    Creates batches using the appropriate dataset loader.

    Args:
        dataset_name (str): Name of dataset ('pes2ox', 'stack', 'falcon', or 'fineweb')
        num_pages (int): Number of pages to fetch
        tokenizer (AutoTokenizer): Tokenizer to use
        batch_size (int): Size of batches to return
        sequence_length (int): Maximum sequence length
        random_seed (Optional[int]): Random seed for reproducibility

    Returns:
        List[np.ndarray]: List of batched token sequences
    """
    # Map dataset names to loader classes
    dataset_map = {
        "Pes2oX": SubsetPes2oXLoader,
        "StackV1Dedup": SubsetStackV1DedupLoader,
        "RefinedWeb": SubsetFalconLoader,
        "FineWeb Edu 2": SubsetFineWebEdu2Loader,
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Dataset must be one of: {list(dataset_map.keys())}")

    # Initialize loader
    loader_class = dataset_map[dataset_name]
    loader = loader_class(
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_pages=num_pages,
        tokenizer=tokenizer,
        random_seed=random_seed,
        pack_samples=pack_samples,
        add_bos=add_bos,
    )

    # Collect batches
    batches = []
    for batch in loader:
        batches.append(batch)

    return batches


def get_wikitext103() -> str:
    """Returns the wikitext103 dataset.

    Args:
        cache_dir (str): The directory to cache the dataset.
    """
    wikitext_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    return "\n\n".join(wikitext_dataset["text"])


def prepare_wikitext_batches(
    text: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 1,
    sequence_length: int = 2048,
    add_bos: bool = True,
) -> List[np.ndarray]:
    """
    Prepares batches from the WikiText dataset.

    Args:
        text (str): Full WikiText dataset text
        tokenizer (AutoTokenizer): Tokenizer to use
        batch_size (int): Batch size for processing
        sequence_length (int): Maximum sequence length

    Returns:
        List[np.ndarray]: List of token batches
    """
    # Configure tokenizer based on add_bos
    if add_bos:
        tokenizer.add_special_tokens({"bos_token": tokenizer.bos_token})

    # Tokenize with special tokens
    tokens = tokenizer(
        text, return_tensors="np", truncation=False, add_special_tokens=True
    )["input_ids"][0]

    # Calculate number of sequences and create batches
    n_sequences = len(tokens) // sequence_length
    if n_sequences == 0:
        raise ValueError("Text is shorter than sequence_length")

    # Reshape into sequences
    sequences = tokens[: n_sequences * sequence_length].reshape(-1, sequence_length)

    # Create batches
    n_batches = len(sequences) // batch_size
    batches = [
        sequences[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)
    ]

    return batches

def _fetch_page_data(loader, page):
    """Helper function to fetch data for a single page."""
    loader.params["offset"] = page
    loader.params["length"] = loader.num_rows_per_page

    attempt = 0
    while attempt < loader.retry_limit:
        try:
            response = requests.get(
                loader.rows_base_url,
                params=loader.params,
                headers=loader._get_request_headers(),
            )
            response.raise_for_status()
            
            return [
                {"text": loader._get_content_from_row(row)}
                for row in response.json()["rows"]
            ]

        except requests.exceptions.RequestException as e:
            attempt += 1
            print(f"Failed to fetch data for page {page}, retrying. Attempt {attempt}/{loader.retry_limit}")
            if attempt < loader.retry_limit:
                time.sleep(loader.retry_delay)
            else:
                print("Maximum retry limit reached. Unable to fetch data.")
                raise

def save_pages_to_jsonl(
    dataset_name: str,
    num_pages: int,
    output_path: str,
    random_seed: Optional[int] = None,
    max_workers: int = 6,
) -> None:
    """
    Saves dataset pages to a JSONL file compatible with Hugging Face datasets.
    Each line will be a JSON object containing a single page.

    Args:
        dataset_name (str): Name of dataset ('Pes2oX', 'StackV1Dedup', 'RefinedWeb', or 'FineWeb Edu 2')
        num_pages (int): Number of pages to fetch
        output_path (str): Path where to save the JSONL file
        random_seed (Optional[int]): Random seed for reproducibility
        max_workers (int): Maximum number of parallel workers for fetching data
    """
    # Map dataset names to loader classes
    dataset_map = {
        "Pes2oX": SubsetPes2oXLoader,
        "StackV1Dedup": SubsetStackV1DedupLoader,
        "RefinedWeb": SubsetFalconLoader,
        "FineWeb Edu 2": SubsetFineWebEdu2Loader,
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Dataset must be one of: {list(dataset_map.keys())}")

    print(f"Saving {num_pages} pages from {dataset_name} to {output_path}")

    # Create a minimal loader instance without tokenizer
    class MinimalLoader(dataset_map[dataset_name]):
        def __init__(self, random_seed=None):
            self.num_rows_per_page = 50
            self.duplicate_page_threshold = 100
            self.retry_limit = 10
            self.retry_delay = 5
            self.requires_auth = False
            self.hf_token = None
            self.config = "default"
            self.split = "train"
            self.params = self._get_default_params()
            
            if random_seed is not None:
                random.seed(random_seed)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pages = []
    
    # For FineWeb Edu 2, we can use the existing fetch_data_to_rows method
    if dataset_name == "FineWeb Edu 2":
        loader = MinimalLoader(random_seed=random_seed)
        print("Fetching dataset configs...")
        loader.configs_data = loader.fetch_dataset_configs()
        print("Fetching pages...")
        rows = loader.fetch_data_to_rows(num_pages)
        pages = [{"text": text} for text in rows]
    else:
        # Initialize loader and fetch pages
        loader = MinimalLoader(random_seed=random_seed)
        page_offsets = loader._sample_pages()
        
        pages = []
        fetched_pages = 0
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create progress bar
            pbar = tqdm(total=num_pages, desc=f"Fetching {dataset_name} pages")
            
            # Submit only the number of pages we need
            future_to_page = {
                executor.submit(_fetch_page_data, loader, page): page 
                for page in page_offsets[:num_pages]  # Limit to num_pages
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                if fetched_pages >= num_pages:  # Stop if we have enough pages
                    break
                    
                page = future_to_page[future]
                try:
                    page_data = future.result()
                    pages.extend(page_data)
                    fetched_pages += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"Page {page} generated an exception: {e}")
            
            pbar.close()

    # Write pages as JSONL, one page per line
    print(f"Writing {len(pages)} pages to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for page in tqdm(pages, desc="Writing to file"):
            f.write(json.dumps(page, ensure_ascii=False) + '\n')
    
    print("Done!")