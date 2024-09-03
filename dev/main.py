# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import logging
import os

os.environ["HF_HOME"] = "/home/scratch/mingzhul/.cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
import glob
from sklearn.preprocessing import StandardScaler
import pandas as pd


import re
import sys

# sys.path.append("chronos_forecasting")


import json
import itertools
import random
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import List, Iterator, Optional, Dict

import typer
from typer_config import use_yaml_config
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    T5Config,
    Trainer,
    TrainingArguments,
)
import accelerate
import gluonts
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic, Map, Filter
from gluonts.transform import (
    FilterTransformation,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
    LeavesMissingValues,
    LastValueImputation,
)

from chronos import ChronosConfig, ChronosTokenizer


from transformers import T5Config
from t5_multivariate_prefix import T5StackWithPrefixMulti, T5ForConditionalGenerationWithPrefixMulti

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss, MSE, MAE
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm
from typing import Iterable, Optional
from chronos import ChronosPipeline, ChronosModel





def control_randomness(seed: int = 13):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




app = typer.Typer(pretty_exceptions_enable=False)


def is_main_process() -> bool:
    """
    Check if we're on the main process.
    """
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0


def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    """
    Log the given message using the given logger, if we're on the main process.
    """
    if is_main_process():
        logger.log(log_level, msg)


def get_training_job_info() -> Dict:
    """
    Returns info about this training job.
    """
    job_info = {}

    # CUDA info
    job_info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        job_info["device_count"] = torch.cuda.device_count()

        job_info["device_names"] = {
            idx: torch.cuda.get_device_name(idx)
            for idx in range(torch.cuda.device_count())
        }
        job_info["mem_info"] = {
            idx: torch.cuda.mem_get_info(device=idx)
            for idx in range(torch.cuda.device_count())
        }

    # DDP info
    job_info["torchelastic_launched"] = dist.is_torchelastic_launched()

    if dist.is_torchelastic_launched():
        job_info["world_size"] = dist.get_world_size()

    # Versions
    job_info["python_version"] = sys.version.replace("\n", " ")
    job_info["torch_version"] = torch.__version__
    job_info["numpy_version"] = np.__version__
    job_info["gluonts_version"] = gluonts.__version__
    job_info["transformers_version"] = transformers.__version__
    job_info["accelerate_version"] = accelerate.__version__

    return job_info


def save_training_info(ckpt_path: Path, training_config: Dict):
    """
    Save info about this training job in a json file for documentation.
    """
    assert ckpt_path.is_dir()
    with open(ckpt_path / "training_info.json", "w") as fp:
        json.dump(
            {"training_config": training_config, "job_info": get_training_job_info()},
            fp,
            indent=4,
        )


def get_next_path(
    base_fname: str,
    base_dir: Path,
    file_type: str = "yaml",
    separator: str = "-",
):
    """
    Gets the next available path in a directory. For example, if `base_fname="results"`
    and `base_dir` has files ["results-0.yaml", "results-1.yaml"], this function returns
    "results-2.yaml".
    """
    if file_type == "":
        # Directory
        items = filter(
            lambda x: x.is_dir() and re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob("*"),
        )
    else:
        # File
        items = filter(
            lambda x: re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob(f"*.{file_type}"),
        )
    run_nums = list(
        map(lambda x: int(x.stem.replace(base_fname + separator, "")), items)
    ) + [-1]

    next_num = max(run_nums) + 1
    fname = f"{base_fname}{separator}{next_num}" + (
        f".{file_type}" if file_type != "" else ""
    )

    return base_dir / fname


def load_model(
    model_id="google/t5-efficient-tiny",
    model_type="seq2seq",
    vocab_size=4096,
    random_init=False,
    tie_embeddings=False,
    pad_token_id=0,
    eos_token_id=1,
    n_channels=1,
    prompt=False,
    lora=False,
    forecast_length=24,
):
    """
    Load the specified HuggingFace model, adjusting the vocabulary
    size, special token IDs, and initialization options.

    This allows to set a model up for training on a new vocabulary
    of tokens.
    """
    assert model_type in ["seq2seq", "causal"]
    AutoModelClass = (
        AutoModelForSeq2SeqLM if model_type == "seq2seq" else AutoModelForCausalLM
    )
    if random_init:
        log_on_main("Using random initialization", logger)
        config = AutoConfig.from_pretrained(model_id)
        if isinstance(config, T5Config):
            # The default initializer_factor (1.0) in transformers is too large
            config.initializer_factor = 0.05
        config.tie_word_embeddings = tie_embeddings
        model = AutoModelClass.from_config(config)
    else:
        log_on_main(f"Using pretrained initialization from {model_id}", logger)
        # model = AutoModelClass.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


    model.resize_token_embeddings(vocab_size)

    model.config.pad_token_id = model.generation_config.pad_token_id = pad_token_id
    model.config.eos_token_id = model.generation_config.eos_token_id = eos_token_id


    if prompt:
        model_config = T5Config.from_pretrained(model_id)
        setattr(model_config, 'num_prefix', 16)
        setattr(model_config, 'reparam', True)
        setattr(model_config, 'reparam_dim', 32)
        setattr(model_config, 'no_decoder_self_attn', False)
        setattr(model_config, 'MPT', False)
        setattr(model_config, 'seq_len', 513)
        setattr(model_config, 'multivariate_projection', 'attention')
        setattr(model_config, 'visualize_attention', False)
        setattr(model_config, 'forecast_length', forecast_length)

        # TODO: calculate this
        # num_patches = 64
        # setattr(model_config, 'num_patches', num_patches)
        # setattr(model_config, 'prefix_tuning', True)

        setattr(model_config, 'n_channels', n_channels)

        transformer_backbone = T5ForConditionalGenerationWithPrefixMulti(model_config)

        model = transformer_backbone.from_pretrained(model_id, config=model_config)

        # freeze params
        for n, param in model.named_parameters():
            if 'prefix' not in n and 'prompt' not in n and 'head' not in n and 'layer_norm' not in n and 'shared' not in n and 'embed_tokens' not in n:
                param.requires_grad = False

        # print unfrozen params
        for n, param in model.named_parameters():
            if param.requires_grad:
                print(n)


    if lora:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["lm_head"],
        )
        model = get_peft_model(model, config)



    return model


def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    Check if the given entry has enough observations in the ``"target"`` attribute.

    Parameters
    ----------
    entry
        The data entry (dictionary) to be tested.
    min_length
        The minimum length the ``"target"`` attribute must have.
    max_missing_prop
        The maximum proportion of missing data allowed in the ``"target"``
        attribute.
    """
    if (
        len(entry["target"]) >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class ChronosDataset(IterableDataset, ShuffleMixin):
    """
    Dataset wrapper, using a ``ChronosTokenizer`` to turn data from a time series
    into a HuggingFace-compatible set of ``input_ids``, ``attention_mask`` and
    ``labels``.

    Entries from the original datasets are assumed to have a ``"start"`` attribute
    (of type ``pd.Period``), and a ``"target"`` attribute (of type ``np.ndarray``).

    Parameters
    ----------
    datasets
        Datasets containing the original time series data.
    probabilities
        In training mode, data will be sampled from each of the original datasets
        with these probabilities.
    tokenizer
        Tokenizer to be used to turn sequences of real numbers into token IDs.
    context_length
        Samples context will be limited to this length.
    prediction_length
        Samples labels will be limited to this length.
    drop_prob
        In training mode, observations from a sample will be turned into ``np.nan``,
        i.e. turned into missing values, with this probability.
    min_past
        Data samples will be considered only if there's at least ``min_past``-many
        historical observations.
    mode
        One of ``"training"``, ``"validation"``, or ``"test"``.
    np_dtype
        Numpy float data type.
    """

    def __init__(
        self,
        datasets: list,
        probabilities: List[float],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: Optional[int] = None,
        model_type: str = "seq2seq",
        imputation_method: Optional[MissingValueImputation] = None,
        mode: str = "training",
        np_dtype=np.float32,
    ) -> None:
        super().__init__()

        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")
        assert model_type in ("seq2seq", "causal")

        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob if model_type == "seq2seq" else 0.0
        self.min_past = min_past or prediction_length
        self.model_type = model_type
        self.imputation_method = imputation_method or LeavesMissingValues()
        self.mode = mode
        self.np_dtype = np_dtype

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        assert entry["target"].ndim == 1, f"got {entry['target'].ndim=}, expected 1"

        if self.model_type == "causal":
            # TODO: what missing values???


            # Causal models do not play nice with missing values, so it is
            # recommended to use an imputation method, e.g., LastValueImputation
            entry["target"] = self.imputation_method(entry["target"])

        if mode == "training" and self.drop_prob > 0:
            target = entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice(
                [True, False], size=len(target), p=[drop_p, 1 - drop_p]
            )
            target[mask] = np.nan
            entry["target"] = target

        return entry

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "test", "validation"]

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter(
            "training"
        ) + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        data = split_transform.apply(data, is_train=True)
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            past_target
        )
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        labels[labels_mask == 0] = -100

        if self.model_type == "causal":
            # The InstanceSplitter pads time series on the left to be equal to the
            # context_length. However, certain models (e.g., GPT2) with absolute
            # position embeddings should not be trained with left padding.
            # The following piece of code moves padding from left to right.

            assert input_ids.shape[-1] == entry["past_is_pad"].shape[0]

            # Find the index where padding starts
            pad_start_idx = np.searchsorted(1 - entry["past_is_pad"], 1)
            padded_input_ids, obs_input_ids = torch.tensor_split(
                input_ids, [pad_start_idx], dim=-1
            )
            padded_attention_mask, obs_attention_mask = torch.tensor_split(
                attention_mask, [pad_start_idx], dim=-1
            )

            # Move padding to the right
            input_ids = torch.cat(
                [
                    obs_input_ids,
                    labels,
                    padded_input_ids,
                ],
                axis=-1,
            )
            attention_mask = torch.cat(
                [
                    obs_attention_mask,
                    labels_mask,
                    padded_attention_mask,
                ],
                axis=-1,
            )

            # labels for causal models are same as the input_ids.
            # Internally transformers shifts the labels by one during training.
            labels = input_ids.clone()
            input_ids[~attention_mask] = self.tokenizer.config.pad_token_id
            labels[~attention_mask] = -100

        return {
            # "n_channels": 1, # TODO
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def __iter__(self) -> Iterator:
        preprocessed_datasets = [
            Map(
                partial(self.preprocess_entry, mode=self.mode),
                dataset,
            )
            for dataset in self.datasets
        ]

        if self.mode == "training":
            iterables = [
                self.create_training_data(dataset) for dataset in preprocessed_datasets
            ]
        elif self.mode == "test":
            iterables = [
                self.create_test_data(dataset) for dataset in preprocessed_datasets
            ]
        else:
            iterables = [
                self.create_validation_data(dataset)
                for dataset in preprocessed_datasets
            ]

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(
                itertools.islice(self.probabilities, worker_id, None, num_workers)
            )

        probs = [prob / sum(probs) for prob in probs]

        iterators = list(map(iter, iterables))
        if self.mode == "training":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs)
                try:
                    yield self.to_hf_format(next(iterators[idx]))
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry)






class Chronos_Moment_Dataset(IterableDataset):

    def __init__(
        self,
        moment_data: IterableDataset,
        tokenizer: ChronosTokenizer,
        model_type: str = "seq2seq",
        prompt: bool = False,
        *args,
        **kwargs
    ) -> None:

        super().__init__()

        self.moment_data = moment_data
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.n_channels = moment_data.n_channels
        self.prompt = prompt

        self.length = len([m for m in moment_data]) #* self.n_channels

        if not prompt:
            self.length *= self.n_channels


    def to_hf_format(self, entry: dict) -> dict:

        timeseries, forecast, input_mask = entry

        past_target = torch.tensor(timeseries)    # n_channels x seq_len
        future_target = torch.tensor(forecast)    # n_channels x seq_len

        # TODO check what context_input_transform does
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            past_target
        )
        labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        labels[labels_mask == 0] = -100

        # TODO: debug this
        if self.model_type == "causal":
            # The InstanceSplitter pads time series on the left to be equal to the
            # context_length. However, certain models (e.g., GPT2) with absolute
            # position embeddings should not be trained with left padding.
            # The following piece of code moves padding from left to right.

            assert input_ids.shape[-1] == entry["past_is_pad"].shape[0]

            # Find the index where padding starts
            pad_start_idx = np.searchsorted(1 - entry["past_is_pad"], 1)
            padded_input_ids, obs_input_ids = torch.tensor_split(
                input_ids, [pad_start_idx], dim=-1
            )
            padded_attention_mask, obs_attention_mask = torch.tensor_split(
                attention_mask, [pad_start_idx], dim=-1
            )

            # Move padding to the right
            input_ids = torch.cat(
                [
                    obs_input_ids,
                    labels,
                    padded_input_ids,
                ],
                axis=-1,
            )
            attention_mask = torch.cat(
                [
                    obs_attention_mask,
                    labels_mask,
                    padded_attention_mask,
                ],
                axis=-1,
            )

            # labels for causal models are same as the input_ids.
            # Internally transformers shifts the labels by one during training.
            labels = input_ids.clone()
            input_ids[~attention_mask] = self.tokenizer.config.pad_token_id
            labels[~attention_mask] = -100

        assert len(input_ids.shape) == len(labels.shape) == len(attention_mask.shape) == 2
        assert input_ids.shape[0] == self.n_channels and input_ids.shape[1] == 513

        return {
            # "n_channels": 1, # TODO
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __iter__(self) -> Iterator:

        for data in self.moment_data:
            d = self.to_hf_format(data)

            if self.prompt:
                yield d

            else:
                for channel in range(self.n_channels):
                    yield {k: v[channel] for k, v in d.items()}    # TODO: add assert to check if the shape is correct

    def __len__(self):
        return self.length


class InformerDatasetMultiFile(torch.utils.data.IterableDataset):
    def __init__(
        self,
        batch_size,
        forecast_horizon: Optional[int] = 192,
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
        filename = "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer",
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', or  'imputation'.
        random_seed : int
            Random seed for reproducibility.
        """
        # TODO: need to init?

        self.batch_size = batch_size

        self.seq_len = 512
        self.forecast_horizon = forecast_horizon
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed

        self.filename = filename
        # get data
        self._read_data(filename)


    def _get_borders(self, filename):
        self.train_ratio = 0.6
        self.val_ratio = 0.1
        self.test_ratio = 0.3

        # n_train = math.floor(self.length_timeseries_original * 0.6)
        # n_val = math.floor(self.length_timeseries_original * 0.1)
        # n_test = math.floor(self.length_timeseries_original * 0.3)

        if "ETTm" in filename:
            n_train = 12 * 30 * 24 * 4
            n_val = 4 * 30 * 24 * 4
            n_test = 4 * 30 * 24 * 4

        elif "ETTh" in filename:
            n_train = 12 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24

        else:
            n_train = int(self.train_ratio * self.length_timeseries_original)
            n_test = int(self.test_ratio * self.length_timeseries_original)
            n_val = self.length_timeseries_original - n_train - n_test

        train_end = n_train
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        # TODO: is the indexing correct?
        train = slice(0, train_end)
        val = slice(train_end - self.seq_len, val_end)
        test = slice(test_start, test_end)

        assert train_end > 0 and val_end > 0 and test_start > 0 and test_end > 0 and train_end - self.seq_len > 0

        return train, val, test


    def _read_data(self, filename):
        self.scaler = StandardScaler()
        df = pd.read_csv(filename)
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        df.drop(columns=["date"], inplace=True)
        df = df.infer_objects(copy=False).interpolate(method="cubic")

        data_splits = self._get_borders(filename)

        # TODO: need to std?
        train_data = df[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)
        # df = df.values

        if self.data_split == "train":
            self.data = df[data_splits[0], :]
        elif self.data_split == "val":
            self.data = df[data_splits[1], :]
        elif self.data_split == "test":
            self.data = df[data_splits[2], :]

        self.length_timeseries = self.data.shape[0]


    def __iter__(self):
        if self.task_name == "imputation":
            num = (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        elif self.task_name == "forecasting":
            num = (
                self.length_timeseries - self.seq_len - self.forecast_horizon
            ) // self.data_stride_len + 1
        else:
            raise ValueError("Unknown task name")

        if num < 0:
            raise ValueError("data too short")

        num = num // self.batch_size * self.batch_size
        for index in range(num):

            seq_start = self.data_stride_len * index
            seq_end = seq_start + self.seq_len
            input_mask = np.ones(self.seq_len)

            if self.task_name == "imputation":
                if seq_end > self.length_timeseries:
                    seq_end = self.length_timeseries
                    seq_end = seq_end - self.seq_len

                timeseries = self.data[seq_start:seq_end, :].T

                yield timeseries, input_mask

            else:
                pred_end = seq_end + self.forecast_horizon

                if pred_end > self.length_timeseries:
                    pred_end = self.length_timeseries
                    seq_end = seq_end - self.forecast_horizon
                    seq_start = seq_end - self.seq_len

                assert(seq_start >=0 and seq_end >= 0 and pred_end >= 0)
                timeseries = self.data[seq_start:seq_end, :].T
                forecast = self.data[seq_end:pred_end, :].T

                yield timeseries, forecast, input_mask



@app.command()
@use_yaml_config(param_name="config")
def main(
    training_data_paths: str,
    probability: Optional[str] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    log_steps: int = 500,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-3,
    optim: str = "adamw_torch_fused",
    shuffle_buffer_length: int = 100,
    gradient_accumulation_steps: int = 2,
    model_id: str = "google/t5-efficient-tiny",
    model_type: str = "seq2seq",
    random_init: bool = False,
    tie_embeddings: bool = False,
    output_dir: str = "./output/",
    tf32: bool = True,
    torch_compile: bool = True,
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    dataloader_num_workers: int = 1,
    max_missing_prop: float = 0.9,
    num_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    num_train_epochs: int = 1,
    prompt: bool = False,
    lora: bool = False,
):
    if tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # TF32 floating point format is available only on NVIDIA GPUs
        # with compute capability 8 and above. See link for details.
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        tf32 = False

    if seed is None:
        seed = random.randint(0, 2**32)

    log_on_main(f"Using SEED: {seed}", logger)
    transformers.set_seed(seed=seed)

    raw_training_config = deepcopy(locals())
    output_dir = Path(output_dir)
    training_data_paths = ast.literal_eval(training_data_paths)
    assert isinstance(training_data_paths, list)

    if isinstance(probability, str):
        probability = ast.literal_eval(probability)
    elif probability is None:
        probability = [1.0 / len(training_data_paths)] * len(training_data_paths)
    assert isinstance(probability, list)

    assert len(training_data_paths) == len(probability)

    if dataloader_num_workers > len(training_data_paths):
        log_on_main(
            f"Setting the number of data loader workers to {len(training_data_paths)}, "
            f"instead of {dataloader_num_workers}.",
            logger,
        )
        dataloader_num_workers = len(training_data_paths)

    if isinstance(tokenizer_kwargs, str):
        tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)
    assert isinstance(tokenizer_kwargs, dict)

    assert model_type in ["seq2seq", "causal"]

    output_dir = get_next_path("run", base_dir=output_dir, file_type="")

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(training_data_paths)} datasets "
        f"for training: {training_data_paths}",
        logger,
    )

    log_on_main(
        f"Mixing probabilities: {probability}",
        logger,
    )

    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in training_data_paths
    ]

    log_on_main("Initializing model", logger)

    chronos_config = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_eos_token=use_eos_token,
        model_type=model_type,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


    # moment_file_name = "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv"
    # target = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

    moment_file_name = "/zfsauton/project/public/Mononito/TimeseriesDatasets/forecasting/autoformer/national_illness.csv"
    target = ['% WEIGHTED ILI', '%UNWEIGHTED ILI', 'AGE 0-4', 'AGE 5-24', 'ILITOTAL', 'NUM. OF PROVIDERS', 'OT']


    d = InformerDatasetMultiFile(1,
                                 data_split="train",
                                 forecast_horizon=prediction_length,
                                 filename=moment_file_name,
                                 random_seed=seed)

    shuffled_train_dataset = Chronos_Moment_Dataset(d,
                                                    chronos_config.create_tokenizer(),
                                                    prompt=prompt,
                                                    model_type=model_type)


    d = InformerDatasetMultiFile(1,
                                 data_split="val",
                                 forecast_horizon=prediction_length,
                                 filename=moment_file_name,
                                 random_seed=seed)

    shuffled_val_dataset = Chronos_Moment_Dataset(d,
                                                    chronos_config.create_tokenizer(),
                                                    prompt=prompt,
                                                    model_type=model_type)


    d = InformerDatasetMultiFile(1,
                                 data_split="test",
                                 forecast_horizon=prediction_length,
                                 filename=moment_file_name,
                                 random_seed=seed)

    shuffled_test_dataset = Chronos_Moment_Dataset(d,
                                                    chronos_config.create_tokenizer(),
                                                    prompt=prompt,
                                                    model_type=model_type)


    model = load_model(
        model_id=model_id,
        model_type=model_type,
        vocab_size=n_tokens,
        random_init=random_init,
        tie_embeddings=tie_embeddings,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        prompt=prompt,
        lora=lora,
        n_channels=shuffled_train_dataset.n_channels,
        forecast_length=prediction_length,
    )

    # Add extra items to model config so that it's saved in the ckpt
    model.config.chronos_config = chronos_config.__dict__


    # shuffled_train_dataset = ChronosDataset(
    #     datasets=train_datasets,
    #     probabilities=probability,
    #     tokenizer=chronos_config.create_tokenizer(),
    #     context_length=context_length,
    #     prediction_length=prediction_length,
    #     min_past=min_past,
    #     model_type=model_type,
    #     imputation_method=LastValueImputation() if model_type == "causal" else None,
    #     mode="training",
    # ).shuffle(shuffle_buffer_length=shuffle_buffer_length)

    # Define training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim=optim,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=log_steps if prompt else log_steps * shuffled_train_dataset.n_channels,
        save_strategy="steps",
        save_steps=save_steps if prompt else log_steps * shuffled_train_dataset.n_channels,
        report_to=["tensorboard"],
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        tf32=tf32,  # remove this if not using Ampere GPUs (e.g., A100)
        torch_compile=torch_compile,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        num_train_epochs=num_train_epochs,

        evaluation_strategy='steps',
        load_best_model_at_end=True,
    )




    from torch.optim.lr_scheduler import OneCycleLR
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    total_steps = (len(shuffled_train_dataset) // per_device_train_batch_size + 1) * num_train_epochs
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.3)




    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
        eval_dataset=shuffled_val_dataset,
        optimizers=(optimizer, scheduler),
    )
    log_on_main("Training", logger)


    # for batch in trainer.get_train_dataloader():
    #     print("batch")
    #     outputs = trainer.model.cpu()(**{k: v.to("cpu") for k, v in batch.items()})



    trainer.train()
    print(trainer.evaluate(eval_dataset=shuffled_test_dataset))



    torch_dtype = "bfloat16"
    batch_size = 32
    num_samples = 20
    temperature = None
    top_k = None
    top_p = None


    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map='cuda',
        torch_dtype=torch_dtype,
    )


    pipeline.model.model = trainer.model


    # results = trainer.predict(shuffled_test_dataset)

    # preds = results.predictions
    # labels = results.label_ids



    # for b in shuffled_test_dataset:
    #     r = trainer.prediction_step(trainer.model, b, False)
    #     break



    preds = []
    trues = []
    for data in tqdm(d):
        
        timeseries, forecast, input_mask = data
        
        pred = pipeline.predict(
            torch.tensor(timeseries),
            prediction_length=prediction_length,
            # num_samples=num_samples,
            num_samples=1,
            limit_prediction_length=False,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ).numpy().squeeze()
        
        preds.append(pred.flatten())
        trues.append(forecast.flatten())


    preds = np.array(preds)
    trues = np.array(trues)



    # mae, mse
    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues) ** 2)
    
    print('mae:', mae)
    print('mse:', mse)



    # # EVAL
    # df = pd.read_csv(moment_file_name,
    #     index_col=0,
    #     parse_dates=True,
    #     )

    # dataset = PandasDataset(df, target=target)

    # # training_data, test_gen = split(dataset, offset=-prediction_length)

    # # fine to have length > self.config.context_length, chorons checks and deletes the extra
    # training_data, test_gen = split(dataset,
    #                                 offset=int(0.7 * len(df))) # test idx start
    # distance = prediction_length # TODO
    # test_data = test_gen.generate_instances(prediction_length=prediction_length,
    #                                         windows=len(shuffled_test_dataset)//distance if prompt else len(shuffled_test_dataset) // shuffled_test_dataset.n_channels // distance,
    #                                         # windows=2, # TODO
    #                                         max_history=context_length,
    #                                         # distance=1, # data_stride_len
    #                                         distance=distance, # data_stride_len
    #                                         )



    # def generate_sample_forecasts(
    #     test_data_input: Iterable,
    #     pipeline: ChronosPipeline,
    #     # pipeline: Trainer,
    #     prediction_length: int,
    #     batch_size: int,
    #     num_samples: int,
    #     **predict_kwargs,
    # ):
    #     # Generate forecast samples
    #     sample_forecasts = []
    #     for batch in tqdm(test_data_input):
    #         forecast_samples = []
    #         context = torch.tensor(batch["target"])
    #         forecast_samples.append(
    #             pipeline.predict(
    #                 context,
    #                 prediction_length=prediction_length,
    #                 # num_samples=num_samples,
    #                 num_samples=1,
    #                 limit_prediction_length=False,
    #                 **predict_kwargs,
    #             ).numpy()
    #         )
    #         forecast_samples = np.concatenate(forecast_samples)

    #     # Convert forecast samples into gluonts SampleForecast objects
    #         # if prompt:
    #         forecast_start_date = batch["start"] + 512   # TODO: what is this for?
    #         sample_forecasts.append(
    #             SampleForecast(samples=np.transpose(forecast_samples, (1,2,0)), start_date=forecast_start_date)
    #             # SampleForecast(samples=np.transpose(forecast_samples, (1,0)), start_date=forecast_start_date)
    #         )
    #         # else:
    #         #     for forecast_sample in forecast_samples:
    #         #         forecast_start_date = batch["start"] + 512
    #         #         sample_forecasts.append(
    #         #             SampleForecast(samples=forecast_sample, start_date=forecast_start_date)
    #         #         )

    #     return sample_forecasts




    # sample_forecasts = generate_sample_forecasts(
    #     test_data.input,
    #     pipeline=pipeline,
    #     prediction_length=prediction_length,
    #     batch_size=8,
    #     num_samples=num_samples,
    #     temperature=temperature,
    #     top_k=top_k,
    #     top_p=top_p,
    #     )

    # # if prompt:
    # #     for i in range(len(sample_forecasts)):
    # #         sample_forecasts[i].samples = np.expand_dims(sample_forecasts[i].samples, 0)






    # # mse and mae
    # # for sample in sample_forecasts:
    # #     pass
        






    # metrics = (
    #     evaluate_forecasts(
    #         sample_forecasts,
    #         test_data=test_data,
    #         metrics=[
    #             MASE(),
    #             MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
    #             MAE(),
    #             MSE(),
    #         ],
    #         batch_size=5000,
    #     )
    #     .reset_index(drop=True)
    #     .to_dict(orient="records")
    # )

    # print('test metrics:', metrics)


    if is_main_process():
        model.save_pretrained(output_dir / "checkpoint-final")
        save_training_info(
            output_dir / "checkpoint-final", training_config=raw_training_config
        )





if __name__ == "__main__":

    # TODO:
    # try ts decoder with prompt
    #   - what's the shape of decoder_input_ids? in training the length is 25, in inference it's 1


    for seed in range(5):

        control_randomness(seed=seed)

        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__file__)
        logger.setLevel(logging.INFO)
        # app()

        config = {
            'training_data_paths': "['/zfsauton2/home/mingzhul/time-series-prompt/src/chronos_forecasting/scripts/kernelsynth-data.arrow']",
            'probability': [1.0],
            'context_length': 512,
            'min_past': 60,
            'save_steps': 100_000,
            'log_steps': 500,       # TODO: need to times by n_channels?
            'per_device_train_batch_size': 8,
            'optim': 'adamw_torch_fused',
            'num_samples': 20,
            'shuffle_buffer_length': 100_000,
            'gradient_accumulation_steps': 1,
            'model_id': 'amazon/chronos-t5-small',
            'model_type': 'seq2seq',
            'random_init': False,
            'tie_embeddings': True,
            'output_dir': '/home/scratch/mingzhul/time-series-prompt/output/',
            'tf32': True,
            'torch_compile': True,
            'tokenizer_class': 'MeanScaleUniformBins',
            'tokenizer_kwargs': {'low_limit': -15.0, 'high_limit': 15.0},
            'n_tokens': 4096,
            'warmup_ratio': 0.0,
            'dataloader_num_workers': 1,
            'max_missing_prop': 0.9,

            # 'use_eos_token': False, # TODO: change this
            'use_eos_token': True,
            'max_steps': -1,
            'num_train_epochs': 10,
            # 'lr_scheduler_type': 'linear',
            'learning_rate': 1e-5, # TODO

            'prediction_length': 60,
            'prompt': False,
            'lora': True,
            
            'seed': seed,
        }

        main(**config)


