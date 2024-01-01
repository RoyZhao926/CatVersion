import logging
import os
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from os.path import join
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from .composition import AdapterCompositionBlock, Fuse, Stack, parse_composition
from .configuration import (
    ADAPTER_CONFIG_MAP,
    AdapterConfig,
    AdapterConfigBase,
    AdapterFusionConfig,
    get_adapter_config_hash,
)
from .context import AdapterSetup, ForwardContext
from .hub_mixin import PushAdapterToHubMixin
from .layer import AdapterLayer, AdapterLayerBase
from .loading import AdapterFusionLoader, AdapterLoader, PredictionHeadLoader, WeightsLoader
from .lora import LoRALayer
from .modeling import Adapter, GLOWCouplingBlock, NICECouplingBlock, init_shared_parameters
from .prefix_tuning import PrefixTuningPool, PrefixTuningShim
from .utils import EMBEDDING_FILE, TOKENIZER_PATH, inherit_doc
from .wrappers.configuration import SUBMODEL_NAMES, wrap_config


logger = logging.getLogger(__name__)


class InvertibleAdaptersMixin:
    """Mixin for Transformer models adding invertible adapters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invertible_adapters = nn.ModuleDict(dict())

        # Make sure config is wrapped
        if hasattr(self, "config"):
            self.config = wrap_config(self.config)

    def add_invertible_adapter(self, adapter_name: str):
        """
        Adds an invertible adapter module for the adapter with the given name. If the given adapter does not specify an
        invertible adapter config, this method does nothing.

        Args:
            adapter_name (str): The name of the adapter for which to add an invertible adapter module.
        """
        if adapter_name in self.invertible_adapters:
            raise ValueError(f"Model already contains an adapter module for '{adapter_name}'.")
        embedding_size = getattr(self.config, "embedding_size", self.config.hidden_size)
        adapter_config = self.config.adapters.match(
            adapter_name,
            config_type=AdapterConfig,
            location_key="inv_adapter",
        )
        if adapter_config and adapter_config["inv_adapter"]:
            if adapter_config["inv_adapter"] == "nice":
                inv_adap = NICECouplingBlock(
                    [[embedding_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            elif adapter_config["inv_adapter"] == "glow":
                inv_adap = GLOWCouplingBlock(
                    [[embedding_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            else:
                raise ValueError(f"Invalid invertible adapter type '{adapter_config['inv_adapter']}'.")
            self.invertible_adapters[adapter_name] = inv_adap
            self.invertible_adapters[adapter_name].apply(Adapter.init_bert_weights)

    def delete_invertible_adapter(self, adapter_name: str):
        if adapter_name in self.invertible_adapters:
            del self.invertible_adapters[adapter_name]

    def get_invertible_adapter(self):
        # TODO: Currently no fusion over invertible adapters, takes only very first language adapter position
        if self.config.adapters.active_setup is not None and len(self.config.adapters.active_setup) > 0:
            first_adapter = self.config.adapters.active_setup.first()
            if first_adapter in self.invertible_adapters:
                return self.invertible_adapters[first_adapter]
        return None

    def enable_invertible_adapters(self, adapter_names, return_param=False):

        for adapter_name in adapter_names:
            if adapter_name in self.invertible_adapters: 
                for param in self.invertible_adapters[adapter_name].parameters():
                    param.requires_grad = True
        if return_param:
            return param_list

    def invertible_adapters_forward(self, hidden_states, rev=False):
        # TODO: Currently no fusion over invertible adapters, takes only very first language adapter position
        if self.config.adapters.active_setup is not None and len(self.config.adapters.active_setup) > 0:
            first_adapter = self.config.adapters.active_setup.first()
            if first_adapter in self.invertible_adapters:
                hidden_states = self.invertible_adapters[first_adapter](hidden_states, rev=rev)

        return hidden_states


class InvertibleAdaptersWrapperMixin:
    """
    Mixin for Transformer models supporting invertible adapters in a child module. When applying this mixin, set
    `invertible_adapters_base_name` to the name of the child module that includes `InvertibleAdaptersMixin`.
    """

    invertible_adapters_base_name = ""

    @property
    def invertible_adapters_base(self):
        return getattr(self, self.invertible_adapters_base_name, None)

    @property
    def invertible_adapters(self):
        if self.invertible_adapters_base is not None:
            return self.invertible_adapters_base.invertible_adapters
        return None

    def add_invertible_adapter(self, adapter_name: str):
        """
        Adds an invertible adapter module for the adapter with the given name. If the given adapter does not specify an
        invertible adapter config, this method does nothing.

        Args:
            adapter_name (str): The name of the adapter for which to add an invertible adapter module.
        """
        if self.invertible_adapters_base is not None:
            self.invertible_adapters_base.add_invertible_adapter(adapter_name)

    def delete_invertible_adapter(self, adapter_name: str):
        if self.invertible_adapters_base is not None:
            self.invertible_adapters_base.delete_invertible_adapter(adapter_name)

    def get_invertible_adapter(self):
        if self.invertible_adapters_base is not None:
            return self.invertible_adapters_base.get_invertible_adapter()
        return None

    def enable_invertible_adapters(self, adapter_names, return_param=False):
        if self.invertible_adapters_base is not None:
            self.invertible_adapters_base.enable_invertible_adapters(adapter_names, return_param=False)

    def invertible_adapters_forward(self, hidden_states, rev=False):
        if self.invertible_adapters_base is not None:
            return self.invertible_adapters_base.invertible_adapters_forward(hidden_states, rev=rev)
        return hidden_states


class EmbeddingAdaptersMixin:
    """Mixin for Transformer models adding support for dynamically switching embeddings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loaded_embeddings = {}
        self._active_embedding = "default"

        # Make sure config is wrapped
        if hasattr(self, "config"):
            self.config = wrap_config(self.config)

    def load_embeddings(self, path: str, name: str):
        """
        Load a saved embedding from the given path. If the embedding was saved with a tokenizer it is returned

        Args:
            path: the path to the saved embedding
            name: the name the embedding should be loaded as

        Returns: a tokenizer if it ws saved with the embedding otherwise None

        """
        from ..models.auto.tokenization_auto import AutoTokenizer

        if name in self.loaded_embeddings:
            raise ValueError("An embedding with the name {} already exists".format(name))
        tokenizer = None
        tokenizer_path = os.path.join(path, TOKENIZER_PATH)
        if os.path.isdir(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        embedding_path = os.path.join(path, EMBEDDING_FILE)
        if not os.path.isfile(embedding_path):
            raise FileNotFoundError("No embeddings found at {}".format(embedding_path))
        weights = torch.load(embedding_path)

        self.loaded_embeddings[name] = nn.Embedding.from_pretrained(weights)
        self.set_active_embeddings(name)
        return tokenizer

    def add_embeddings(self, name, tokenizer, reference_embedding=None, reference_tokenizer=None, embedding_dim=None):
        """
        Add a new embedding to the model. If a reference embedding and reference tokenizer are provided tokens in the
        present in both tokenizers are initialized to the embedding in the reference_embedding.

        Args:
            name: the name of the embedding
            tokenizer: the tokenizer determining the vocab of the embedding
            reference_embedding:
                the reference embedding to use for initializing the embeddings of tokens present in the newly created
                embedding
            reference_tokenizer: the tokenizer providing the vocab for the reference embedding
            embedding_dim:
                the dimension of the embeddings (if None the embedding_size, or if this doesn't exist the hidden_size,
                from the config is used)
        """
        if name in self.loaded_embeddings:
            raise ValueError("An embedding with the name {} already exists".format(name))
        if embedding_dim is not None:
            embedding_size = embedding_dim
        else:
            embedding_size = getattr(self.config, "embedding_size", self.config.hidden_size)
        embedding = nn.Embedding(len(tokenizer), embedding_size)
        # Use same initialization as base Transformer model
        embedding.weight.data.normal_(mean=0.0, std=0.02)
        if embedding.padding_idx is not None:
            embedding.weight.data[embedding.padding_idx].zero_()
        embedding.requires_grad_(False)
        if (reference_embedding is not None and reference_tokenizer is None) or (
            reference_tokenizer is not None and reference_embedding is None
        ):
            raise KeyError(
                "Reference embedding and reference tokenizer are required to use initialize embeddings from reference"
                " embedding"
            )
        if reference_embedding is not None and reference_tokenizer is not None:
            tokens = set(tokenizer.get_vocab().keys()) & set(reference_tokenizer.get_vocab().keys())
            reference_vocab = reference_tokenizer.get_vocab()
            vocab = tokenizer.get_vocab()
            for t in tokens:
                idx_reference = reference_vocab[t]
                idx = vocab[t]
                embedding.weight[idx] = (
                    self.loaded_embeddings[reference_embedding].weight[idx_reference].detach().clone()
                )
        embedding.train(False)
        self.loaded_embeddings[name] = embedding
        self.set_active_embeddings(name)

    def delete_embeddings(self, name):
        """
        Deletes the embedding with the given name

        Args:
            name: The name of the embedding that should be deleted

        """
        if name not in self.loaded_embeddings:
            raise ValueError("No embedding with name {}".format(name))
        if self.active_embeddings == name:
            logger.warning("The active embedding is deleted. Setting the default embedding as active.")
            self.set_active_embeddings("default")
        del self.loaded_embeddings[name]

    def save_embeddings(self, path, name, tokenizer=None):
        """
        Saves the embedding with the given name. If a tokenizer is passed as well the tokenizer is saved together with
        the embedding.

        Args:
            path: The path where the embedding should be saved
            name: The name of the embedding that should be saved
            tokenizer: optionally a tokenizer to save with the embedding (default is None)

        """
        if self.active_embeddings == name:
            self.loaded_embeddings[name] = self.get_input_embeddings()
        os.makedirs(path, exist_ok=True)
        embedding_path = os.path.join(path, EMBEDDING_FILE)
        torch.save(self.loaded_embeddings[name].weight, embedding_path)
        if tokenizer:
            tokenizer_path = os.path.join(path, TOKENIZER_PATH)
            tokenizer.save_pretrained(tokenizer_path)

    def set_active_embeddings(self, name):
        """
        Sets the active embedding for the forward pass of the model

        Args:
            name: The name of the embedding that should be used

        """
        self.loaded_embeddings[self.active_embeddings] = self.get_input_embeddings()
        self.set_input_embeddings(self.loaded_embeddings[name])
        self.config.vocab_size = self.loaded_embeddings[name].num_embeddings
        self._active_embedding = name

    @property
    def active_embeddings(self):
        return self._active_embedding


class EmbeddingAdaptersWrapperMixin:
    def load_embeddings(self, path: str, name: str):
        return self.base_model.load_embeddings(path, name)

    def add_embeddings(self, name, tokenizer, reference_embedding=None, reference_tokenizer=None):
        return self.base_model.add_embeddings(name, tokenizer, reference_embedding, reference_tokenizer)

    def delete_embeddings(self, name):
        return self.base_model.delete_embeddings(name)

    def save_embeddings(self, path, name, tokenizer=None):
        return self.base_model.save_embeddings(path, name, tokenizer)

    def set_active_embeddings(self, name):
        return self.base_model.set_active_embeddings(name)

    @property
    def active_embeddings(self):
        return self.base_model.active_embeddings

    @property
    def loaded_embeddings(self):
        return self.base_model.loaded_embeddings


class ModelAdaptersMixin(PushAdapterToHubMixin, ABC):
    """Mixin for transformer models adding support for loading/ saving adapters."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if config.name_or_path and not os.path.exists(config.name_or_path):
            self.model_name = config.name_or_path
        else:
            self.model_name = None
        self.base_model.shared_parameters = nn.ModuleDict()

        # Make sure config is wrapped
        self.config = wrap_config(self.config)

    def _link_prefix_to_pool(self, layer):
        if isinstance(layer, PrefixTuningShim):
            layer.set_pool(self.base_model.prefix_tuning)

    def _init_adapter_modules(self, add_prefix_tuning_pool=True):
        """
        This method initializes adapter modules and fusion modules from the model config.
        """
        # Link all prefix tunings
        if add_prefix_tuning_pool:
            self.base_model.prefix_tuning = PrefixTuningPool(self.config)
            self.apply_to_adapter_layers(lambda i, layer: self._link_prefix_to_pool(layer))

        # Initialize adapters from config
        for adapter_name in self.config.adapters:
            self._add_adapter_weights(adapter_name)
        # Initialize fusion from config
        for fusion_name in self.config.adapters.fusions:
            self.apply_to_adapter_layers(lambda i, layer: layer.add_fusion_layer(fusion_name))

        if isinstance(self, EmbeddingAdaptersMixin):
            self.loaded_embeddings["default"] = self.get_input_embeddings()

    # These methods have to be implemented by every deriving class:

    @abstractmethod
    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        """
        Iterates over all layers of the model.

        This abstract method has to ne implemented by every implementing model.
        """
        pass

    def apply_to_adapter_layers(self, fn):
        """
        Applies a function to all adapter layers of the model.
        """
        for i, layer in self.iter_layers():
            if ( i >= 9 and i < 12 ):
                for module in layer.modules():
                    if isinstance(module, PrefixTuningShim):
                        fn(i, module)
            # for module in layer.modules():
            #     if isinstance(module, AdapterLayerBase):
            #         fn(i, module)

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock], train_embeddings=False, return_param=False):
        """Sets the model into mode for training the given adapters."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        # self.apply_to_adapter_layers(lambda i, layer: layer.enable_adapters(adapter_setup, True, False)) # 设置梯度存在
        for i, layer in self.iter_layers():
            if ( i >= 9 and i < 12 ):
                for module in layer.modules():
                    if isinstance(module, PrefixTuningShim):
                        module.enable_adapters(adapter_setup, True, False) # transformers/adapters/prefix_tuning.py 293 -> 186
            # for module in layer.modules():
            #     if isinstance(module, AdapterLayerBase):
            #         module.enable_adapters(adapter_setup, True, False) # transformers/adapters/prefix_tuning.py 293 -> 186
        
        for adapter_name in adapter_setup:
            if adapter_name in self.base_model.shared_parameters:
                param_list.append(self.base_model.shared_parameters[adapter_name].values())
                for param in self.base_model.shared_parameters[adapter_name].values():
                    param.requires_grad = True

        if isinstance(self, InvertibleAdaptersMixin) or isinstance(self, InvertibleAdaptersWrapperMixin):
            if return_param:
                param_list = self.enable_invertible_adapters(adapter_setup.flatten(),return_param=return_param)  
                #param_list = self.enable_invertible_adapters(adapter_setup.flatten())  
            else:
                self.enable_invertible_adapters(adapter_setup.flatten())  
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup) #毫无作用
        if train_embeddings:
            self.get_input_embeddings().train() 
        if return_param:
            return param_list

    def train_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        warnings.warn(
            "add_fusion() has been deprecated in favor of add_adapter_fusion(). Please use the newer method instead.",
            FutureWarning,
        )
        self.train_adapter_fusion(adapter_setup, unfreeze_adapters=unfreeze_adapters)

    def train_adapter_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        self.train()
        self.freeze_model(True)
        adapter_setup = parse_composition(adapter_setup)
        self.apply_to_adapter_layers(lambda i, layer: layer.enable_adapters(adapter_setup, unfreeze_adapters, True))
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_setup)
        # TODO implement fusion for invertible adapters

    def has_adapters(self):
        if not getattr(self.config, "is_adaptable", None):
            return False
        return len(self.config.adapters.adapters) > 0

    @property
    def has_parallel_adapters(self) -> bool:
        if self.config.adapters.active_setup:
            return self.config.adapters.active_setup.parallel_channels > 1
        else:
            return False

    @property
    def active_adapters(self) -> AdapterCompositionBlock:
        return self.config.adapters.active_setup

    @active_adapters.setter
    def active_adapters(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        self.set_active_adapters(adapter_setup)

    def set_shared_parameters(self, param):
        self.base_model.shared_parameters = param

    def set_active_adapters(
        self, adapter_setup: Union[list, AdapterCompositionBlock], skip_layers: Optional[List[int]] = None
    ):
        """
        Sets the adapter modules to be used by default in every forward pass. If no adapter with the given name is
        found, no module of the respective type will be activated.

        Args:
            adapter_setup (list):
                The list of adapters to be activated by default. Can be a fusion or stacking configuration.
        """
        adapter_setup = parse_composition(adapter_setup, model_type=self.config.model_type)
        if adapter_setup:
            for adapter_name in adapter_setup.flatten():
                if adapter_name not in self.config.adapters.adapters:
                    raise ValueError(
                        f"No adapter with name '{adapter_name}' found. Please make sure that all specified adapters"
                        " are correctly loaded."
                    )

        # Make sure LoRA is reset
        self.reset_adapter()
        self.config.adapters.active_setup = adapter_setup
        self.config.adapters.skip_layers = skip_layers

    def add_adapter(self, adapter_name: str, config=None, overwrite_ok: bool = False, set_active: bool = False):
        """
        Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            config (str or dict or AdapterConfigBase, optional): The adapter configuration, can be either:

                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
            overwrite_ok (bool, optional):
                Overwrite an adapter with the same name if it exists. By default (False), an
            exception is thrown. set_active (bool, optional):
                Set the adapter to be the active one. By default (False),
            the adapter is added but not activated.
        """
        config = AdapterConfigBase.load(config)  # ensure config is ok and up-to-date
        # In case adapter already exists and we allow overwriting, explicitly delete the existing one first
        if overwrite_ok and adapter_name in self.config.adapters:
            self.delete_adapter(adapter_name)
        self.config.adapters.add(adapter_name, config=config)  
        try:
            self._add_adapter_weights(adapter_name) #添加adapter
        except ValueError as ex:
            self.delete_adapter(adapter_name)
            raise ex
        if set_active:
            self.set_active_adapters(adapter_name)

    def _add_adapter_weights(self, adapter_name: str):
        """Helper method that performs the actual parameter additions when adding a new adapter."""
        self.apply_to_adapter_layers(lambda i, layer: layer.add_adapter(adapter_name, i))
        # PHM Layer
        adapter_config = self.config.adapters.match(adapter_name, AdapterConfig, location_key="phm_layer")
        if adapter_config:
            adapter_module = list(self.get_adapter(adapter_name)[0].values())[0]
            # if multiple adapters with same location key exist they are returned as a modulelist
            if isinstance(adapter_module, nn.ModuleList):
                adapter_module = adapter_module[0]
            if adapter_config["shared_phm_rule"] or adapter_config["shared_W_phm"]:
                if self.config.model_type in SUBMODEL_NAMES:
                    hidden_sizes = [
                        getattr(self.config, key).hidden_size for key in SUBMODEL_NAMES[self.config.model_type]
                    ]
                    if all(hidden_sizes[0] == h for h in hidden_sizes):
                        self.base_model.shared_parameters[adapter_name] = init_shared_parameters(
                            adapter_config, hidden_sizes[0], self.device
                        )
                    else:
                        raise ValueError(
                            "The model has different hidden sizes {}. Sharing comapcter weights is only possible if"
                            " the hidden_sizes match.".format(hidden_sizes)
                        )
                else:
                    self.base_model.shared_parameters[adapter_name] = init_shared_parameters(
                        adapter_config, self.config.hidden_size, self.device
                    )
        # Prefix Tuning
        for module in self.modules():
            if isinstance(module, PrefixTuningPool):
                module.confirm_prefix(adapter_name)
        if isinstance(self, InvertibleAdaptersMixin) or isinstance(self, InvertibleAdaptersWrapperMixin):
            self.add_invertible_adapter(adapter_name)

    def add_fusion(self, adapter_names: Union[Fuse, list], adapter_fusion_config=None, override_kwargs=None):
        warnings.warn(
            "add_fusion() has been deprecated in favor of add_adapter_fusion(). Please use the newer method instead.",
            FutureWarning,
        )
        adapter_fusion_config = AdapterFusionConfig.from_dict(adapter_fusion_config).replace(**override_kwargs)
        self.add_adapter_fusion(adapter_names, adapter_fusion_config)

    def add_adapter_fusion(
        self,
        adapter_names: Union[Fuse, list, str],
        config=None,
        overwrite_ok: bool = False,
        set_active: bool = False,
    ):
        """
        Adds AdapterFusion to the model with alll the necessary configurations and weight initializations

        Args:
            adapter_names (Fuse or list or str): AdapterFusion layer to add. Can be either:

                - a ``Fuse`` composition block
                - a list of adapter names to fuse
                - a comma-separated string of adapter names to fuse
            config (str or dict): adapter fusion configuration, can be either:

                - a string identifying a pre-defined adapter fusion configuration
                - a dictionary representing the adapter fusion configuration
                - the path to a file containing the adapter fusion configuration
            overwrite_ok (bool, optional):
                Overwrite an AdapterFusion layer with the same name if it exists. By default (False), an exception is
                thrown.
            set_active (bool, optional):
                Activate the added AdapterFusion. By default (False), the AdapterFusion is added but not activated.
        """
        if isinstance(adapter_names, Fuse):
            adapter_names = adapter_names.children
        elif isinstance(adapter_names, str):
            adapter_names = adapter_names.split(",")

        if isinstance(config, dict):
            config = AdapterFusionConfig.from_dict(config)  # ensure config is ok and up-to-date
        # In case adapter already exists and we allow overwriting, explicitly delete the existing one first
        if overwrite_ok and self.config.adapters.get_fusion(adapter_names) is not None:
            self.delete_adapter_fusion(adapter_names)
        self.config.adapters.add_fusion(adapter_names, config=config)
        self.apply_to_adapter_layers(lambda i, layer: layer.add_fusion_layer(adapter_names))
        if set_active:
            if not isinstance(adapter_names, list):
                adapter_names = adapter_names.split(",")
            self.set_active_adapters(Fuse(*adapter_names))

    def delete_adapter(self, adapter_name: str):
        """
        Deletes the adapter with the specified name from the model.

        Args:
            adapter_name (str): The name of the adapter.
        """
        if adapter_name not in self.config.adapters:
            logger.info("No adapter '%s' found for deletion. Skipping.", adapter_name)
            return
        del self.config.adapters.adapters[adapter_name]
        self.apply_to_adapter_layers(lambda i, layer: layer.delete_adapter(adapter_name))
        # PHM Layer
        if adapter_name in self.base_model.shared_parameters:
            del self.base_model.shared_parameters[adapter_name]
        if isinstance(self, InvertibleAdaptersMixin) or isinstance(self, InvertibleAdaptersWrapperMixin):
            self.delete_invertible_adapter(adapter_name)
        # Reset active adapters if this was the only active adapter
        if self.active_adapters == Stack(adapter_name):
            self.active_adapters = None

    def delete_adapter_fusion(self, adapter_names: Union[Fuse, list, str]):
        """
        Deletes the AdapterFusion layer of the specified adapters.

        Args:
            adapter_names (Union[Fuse, list, str]): AdapterFusion layer to delete.
        """
        if isinstance(adapter_names, Fuse):
            adapter_fusion_name = ",".join(adapter_names.children)
        elif isinstance(adapter_names, list):
            adapter_fusion_name = ",".join(adapter_names)
        elif isinstance(adapter_names, str):
            adapter_fusion_name = adapter_names
        else:
            raise ValueError("Invalid AdapterFusion definition: {}".format(adapter_names))

        if adapter_fusion_name not in self.config.adapters.fusions:
            logger.info("No AdapterFusion '%s' found for deletion. Skipping.", adapter_fusion_name)
            return
        del self.config.adapters.fusions[adapter_fusion_name]
        self.apply_to_adapter_layers(lambda i, layer: layer.delete_fusion_layer(adapter_fusion_name))
        # Reset active adapters if this was the active setup
        if self.active_adapters == adapter_names:
            self.active_adapters = None

    def save_adapter(
        self,
        save_directory: str,
        adapter_name: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves an adapter and its configuration file to a directory so that it can be shared or reloaded using
        `load_adapter()`.

        Args:
            save_directory (str): Path to a directory where the adapter should be saved.
            adapter_name (str): Name of the adapter to be saved.

        Raises:
            ValueError: If the given adapter name is invalid.
        """
        loader = AdapterLoader(self)
        loader.save(save_directory, adapter_name, meta_dict)
        # save additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.save(save_directory, adapter_name)

    def save_adapter_fusion(
        self,
        save_directory: str,
        adapter_names: Union[Fuse, list, str],
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves an AdapterFusion layer and its configuration file to a directory so that it can be shared or reloaded
        using `load_adapter_fusion()`.

        Args:
            save_directory (str): Path to a directory where the AdapterFusion should be saved.
            adapter_names (Union[Fuse, list, str]): AdapterFusion to be saved.

        Raises:
            ValueError: If the given AdapterFusion name is invalid.
        """
        if isinstance(adapter_names, Fuse):
            adapter_fusion_name = ",".join(adapter_names.children)
        elif isinstance(adapter_names, list):
            adapter_fusion_name = ",".join(adapter_names)
        elif isinstance(adapter_names, str):
            adapter_fusion_name = adapter_names
        else:
            raise ValueError("Invalid AdapterFusion definition: {}".format(adapter_names))

        loader = AdapterFusionLoader(self)
        loader.save(save_directory, adapter_fusion_name, meta_dict)
        # save additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.save(save_directory, adapter_fusion_name)

    def load_adapter(
        self,
        adapter_name_or_path: str,
        config: Union[dict, str] = None,
        version: str = None,
        model_name: str = None,
        load_as: str = None,
        source: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        leave_out: Optional[List[int]] = None,
        id2label=None,
        set_active: bool = False,
        **kwargs
    ) -> str:
        """
        Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:

                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (dict or str, optional): The requested configuration of the adapter.
                If not specified, will be either: - the default adapter config for the requested adapter if specified -
                the global default adapter config
            version (str, optional): The version of the adapter to be loaded.
            model_name (str, optional): The string identifier of the pre-trained model.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
                    saved will be used.
            source (str, optional): Identifier of the source(s) from where to load the adapter. Can be:

                - "ah" (default): search on AdapterHub.
                - "hf": search on HuggingFace model hub.
                - None: search on all sources
            leave_out: Dynamically drop adapter modules in the specified Transformer layers when loading the adapter.
            set_active (bool, optional):
                Set the loaded adapter to be the active one. By default (False), the adapter is loaded but not
                activated.

        Returns:
            str: The name with which the adapter was added to the model.
        """
        loader = AdapterLoader(self)
        load_dir, load_name = loader.load(
            adapter_name_or_path,
            config,
            version,
            model_name,
            load_as,
            source=source,
            leave_out=leave_out,
            set_active=set_active,
            **kwargs,
        )
        # load additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.load(
                    load_dir,
                    load_as=load_as,
                    loading_info=kwargs.get("loading_info", None),
                    main_load_name=load_name,
                    id2label=id2label,
                    set_active=set_active,
                )
        return load_name

    def load_adapter_fusion(
        self,
        adapter_fusion_name_or_path: str,
        load_as: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        set_active: bool = False,
        **kwargs
    ) -> str:
        """
        Loads a pre-trained AdapterFusion layer from the local file system.

        Args:
            adapter_fusion_name_or_path (str):
                a path to a directory containing AdapterFusion weights saved using `model.save_adapter_fusion()`.
            load_as (str, optional): Load the AdapterFusion using this name.
                    By default, the name with which the AdapterFusion layer was saved will be used.
            set_active (bool, optional):
                Activate the loaded AdapterFusion. By default (False), the AdapterFusion is loaded but not activated.

        Returns:
            str: The name with which the AdapterFusion was added to the model.
        """

        loader = AdapterFusionLoader(self)
        load_dir, load_name = loader.load(adapter_fusion_name_or_path, load_as, set_active=set_active)
        # load additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.load(
                    load_dir,
                    load_as=load_as,
                    loading_info=kwargs.get("loading_info", None),
                    main_load_name=load_name,
                    set_active=set_active,
                )
        return load_name

    def save_all_adapters(
        self,
        save_directory: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves all adapters of this model together with their configuration to subfolders of the given location.

        Args:
            save_directory (str): Path to a directory where the adapters should be saved.
        """
        os.makedirs(save_directory, exist_ok=True)
        for name in self.config.adapters:
            adapter_config = self.config.adapters.get(name)
            h = get_adapter_config_hash(adapter_config)
            save_path = join(save_directory, name)
            if meta_dict:
                meta_dict.update({"config_id": h})
            else:
                meta_dict = {"config_id": h}
            self.save_adapter(save_path, name, meta_dict=meta_dict, custom_weights_loaders=custom_weights_loaders)

    def save_all_adapter_fusions(
        self,
        save_directory: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves all AdapterFusion layers of this model together with their configuration to subfolders of the given
        location.

        Args:
            save_directory (str): Path to a directory where the AdapterFusion layers should be saved.
        """
        os.makedirs(save_directory, exist_ok=True)
        for name in self.config.adapters.fusions:
            adapter_fusion_config = self.config.adapters.get_fusion(name)
            h = get_adapter_config_hash(adapter_fusion_config)
            save_path = join(save_directory, name)
            if meta_dict:
                meta_dict.update({"config_id": h})
            else:
                meta_dict = {"config_id": h}
            self.save_adapter_fusion(
                save_path, name, meta_dict=meta_dict, custom_weights_loaders=custom_weights_loaders
            )

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model."""
        # first freeze/ unfreeze all model weights
        for param in self.base_model.parameters():
            param.requires_grad = not freeze
        self.model_frozen = freeze

    def forward_context(self, context: ForwardContext, *args, **kwargs):
        """
        This method is called by the ``ForwardContext`` at the beginning of the forward pass.
        """
        # some warnings if we don't use available adapters
        active_adapters = getattr(self, "active_adapters", None) or AdapterSetup.get_context_adapter_setup()
        if not active_adapters:
            if self.has_adapters():
                logger.warning("There are adapters available but none are activated for the forward pass.")
            return

        context.adapters_parallelized = False
        # Check if already parallelized in encoder
        adapter_input_parallelized = kwargs.pop("adapter_input_parallelized", None)
        if adapter_input_parallelized:
            if active_adapters.parallel_channels > 1:
                context.adapters_parallelized = True
        # Add the shared parameters for the active adapters to the context
        context.shared_parameters = {
            name: param
            for name, param in self.base_model.shared_parameters.items()
            if name in active_adapters.flatten()
        }

        context.prefix_states = self.base_model.prefix_tuning(*args, **kwargs)
        # Adapter gating and attention outputs
        context.output_adapter_gating_scores = kwargs.get("output_adapter_gating_scores", False)
        context.output_adapter_fusion_attentions = kwargs.get("output_adapter_fusion_attentions", False)
        context.adapter_gating_scores = defaultdict(dict)
        context.adapter_fusion_attentions = defaultdict(dict)

    def get_fusion_regularization_loss(self):
        reg_loss = None

        target = torch.zeros((self.config.hidden_size, self.config.hidden_size)).fill_diagonal_(1.0).to(self.device)
        for i, layer in self.iter_layers():
            for module in layer.modules():
                if isinstance(module, AdapterLayer):
                    for _, layer_fusion in module.adapter_fusion_layer.items():
                        if hasattr(layer_fusion, "value") and layer_fusion.value.weight.requires_grad:
                            layer_reg_loss = 0.01 * (target - layer_fusion.value.weight).pow(2).sum()
                            if reg_loss is None:
                                reg_loss = layer_reg_loss
                            else:
                                reg_loss += layer_reg_loss

        return reg_loss

    def get_adapter(self, name) -> dict:
        """
        Returns a dictionary with all weights of the adapter with the specified name.

        Args:
            name (str): The adapter name.

        Returns:
            dict: A nested dictionary containing the weights of the adapter. The dictionary is structured as follow:
            {<layer id>: {<module location>: <nn.Module>}}. <layer id> = -1 indicates global/ shared weights.
        """
        destination = defaultdict(dict)

        # global weights are saved at index -1
        if name in self.base_model.shared_parameters:
            destination[-1]["shared"] = self.base_model.shared_parameters[name]
        if (
            isinstance(self, InvertibleAdaptersMixin) or isinstance(self, InvertibleAdaptersWrapperMixin)
        ) and name in self.invertible_adapters:
            destination[-1]["invertible"] = self.invertible_adapters[name]

        # use a custom index to ensure numbering is from 0 to N layers
        for i, (_, layer) in enumerate(self.iter_layers()):
            for module in layer.modules():
                if isinstance(module, AdapterLayerBase):
                    adapter_module = module.get_adapter(name)
                    if adapter_module is not None:
                        # location_key might already be added before -> concat to ModuleList
                        if module.location_key in destination[i]:
                            old_module = destination[i][module.location_key]
                            if isinstance(old_module, nn.ModuleList):
                                old_module.append(adapter_module)
                            else:
                                destination[i][module.location_key] = nn.ModuleList([old_module, adapter_module])
                        else:
                            destination[i][module.location_key] = adapter_module

        return dict(destination)

    def adapter_summary(self, as_dict=False) -> Union[str, dict]:
        """
        Returns a string summary of all adapters currently added to the model. Each entry in the summary table has the
        following attributes:

            - name: the name of the adapter
            - architecture: the architectural base of the adapter
            - #param: the number of parameters of the adapter
            - %param: the number of parameters of the adapter relative to the full model
            - active: whether the adapter is active
            - train: whether the adapter weights are enabled for training
        """
        # table header
        header = ["name", "architecture", "#param", "%param", "active", "train"]
        # rows containing adapter info
        rows = []
        # fill in data for adapters
        for name, config_name in self.config.adapters.adapters.items():
            if config_name in self.config.adapters.config_map:
                config = self.config.adapters.config_map.get(config_name, None)
            else:
                config = ADAPTER_CONFIG_MAP.get(config_name, None)
            if isinstance(config, str):
                config = ADAPTER_CONFIG_MAP[config]
            row = {"name": name, "architecture": config.get("architecture", None) or "bottleneck"}
            weights = self.get_adapter(name)
            row["active"] = self.active_adapters is not None and name in self.active_adapters.flatten()
            # count parameters
            no_params = 0
            train = True
            for _, module_dict in weights.items():
                for _, module in module_dict.items():
                    no_params += sum(p.numel() for p in module.parameters())
                    train &= all(p.requires_grad for p in module.parameters())
            row["#param"] = no_params
            row["train"] = train
            rows.append(row)
        # count no. of parameters in base network
        model_no_params = sum(p.numel() for p in self.base_model.parameters())
        model_no_params -= sum([r["#param"] for r in rows])
        # add %param info
        for row in rows:
            row["%param"] = row["#param"] / model_no_params * 100
        # add full model info
        rows.append(
            {
                "name": "Full model",
                "#param": model_no_params,
                "%param": 100.0,
                "train": not getattr(self.base_model, "model_frozen", False),
            }
        )

        if as_dict:
            return rows
        else:
            # print
            total_length = 80
            header_format = "{:<25}{:<15}{:>12}{:>12}{:>8}{:>8}"
            row_format = "{:<25}{:<15}{:>12,}{:>12.3f}{:>8}{:>8}"
            s = ["=" * total_length]
            s.append(header_format.format(*map(lambda x: x.title(), header)))
            s.append("-" * total_length)
            for row in rows:
                s.append(row_format.format(*[row.get(h, "") for h in header]))
            s.insert(len(s) - 1, "-" * total_length)
            s.append("=" * total_length)
            return "\n".join(s)

    def eject_prefix_tuning(self, name: str):
        """
        Converts the prefix tuning with the given name from the reparameterized form into the flat form.

        Args:
            name (str): The name of the prefix tuning.
        """
        for module in self.modules():
            if isinstance(module, PrefixTuningPool):
                if name in module.prefix_tunings:
                    module.prefix_tunings[name].eject()

    def merge_adapter(self, name: str):
        """
        Merges the weights of the given LoRA module with the Transformer weights as described in the paper.

        Args:
            name (str): LoRA module to merge.
        """
        for module in self.modules():
            if isinstance(module, LoRALayer):
                if name in module.loras:
                    module.merge_adapter(name)

    def reset_adapter(self):
        """
        Resets weights of a LoRA module merged using `model.merge_adapter(name)`.
        """
        for module in self.modules():
            if isinstance(module, LoRALayer):
                module.reset_adapter()


@inherit_doc
class ModelWithHeadsAdaptersMixin(ModelAdaptersMixin):
    """
    Mixin adding support for loading/ saving adapters to transformer models with head(s).
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._convert_to_flex_head = False

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        """
        Iterates over all layers of the model.
        """
        if self.base_model is self:
            return super().iter_layers()
        else:
            return self.base_model.iter_layers()

    def add_adapter(self, adapter_name: str, config=None, overwrite_ok: bool = False, set_active: bool = False):
        """
        Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            config (str or dict, optional): The adapter configuration, can be either:

                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
            overwrite_ok (bool, optional):
                Overwrite an adapter with the same name if it exists. By default (False), an exception is thrown.
            set_active (bool, optional):
                Set the adapter to be the active one. By default (False), the adapter is added but not activated.

        If self.base_model is self, must inherit from a class that implements this method, to preclude infinite
        recursion
        """
        if self.base_model is self:
            super().add_adapter(adapter_name, config, overwrite_ok=overwrite_ok, set_active=set_active)
        else:
            self.base_model.add_adapter(adapter_name, config, overwrite_ok=overwrite_ok, set_active=set_active)

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock], train_embeddings=False):
        """
        Sets the model into mode for training the given adapters. If self.base_model is self, must inherit from a class
        that implements this method, to preclude infinite recursion
        """
        if self.base_model is self:
            super().train_adapter(adapter_setup, train_embeddings)
        else:
            self.base_model.train_adapter(adapter_setup, train_embeddings)

    def train_adapter_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        """
        Sets the model into mode for training of adapter fusion determined by a list of adapter names. If
        self.base_model is self, must inherit from a class that implements this method, to preclude infinite recursion
        """
        if self.base_model is self:
            super().train_adapter_fusion(adapter_setup, unfreeze_adapters=unfreeze_adapters)
        else:
            self.base_model.train_adapter_fusion(adapter_setup, unfreeze_adapters=unfreeze_adapters)

    def save_head(self, save_directory: str, head_name: str = None):
        loader = PredictionHeadLoader(self)
        loader.save(save_directory, name=head_name)

    def load_head(self, save_directory, load_as=None, id2label=None, **kwargs):
        loader = PredictionHeadLoader(self, convert_to_flex_head=self._convert_to_flex_head)
        return loader.load(save_directory, load_as=load_as, id2label=id2label, **kwargs)

    def save_adapter(
        self,
        save_directory: str,
        adapter_name: str,
        with_head: bool = True,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        if with_head:
            if custom_weights_loaders is None:
                custom_weights_loaders = []
            custom_weights_loaders.append(PredictionHeadLoader(self, error_on_missing=False))
        super().save_adapter(
            save_directory,
            adapter_name,
            meta_dict=meta_dict,
            custom_weights_loaders=custom_weights_loaders,
        )

    def load_adapter(
        self,
        adapter_name_or_path: str,
        config: Union[dict, str] = None,
        version: str = None,
        model_name: str = None,
        load_as: str = None,
        source: str = None,
        with_head: bool = True,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        leave_out: Optional[List[int]] = None,
        id2label=None,
        set_active: bool = False,
        **kwargs
    ) -> str:
        if with_head:
            if custom_weights_loaders is None:
                custom_weights_loaders = []
            custom_weights_loaders.append(
                PredictionHeadLoader(
                    self,
                    error_on_missing=False,
                    convert_to_flex_head=self._convert_to_flex_head,
                )
            )
        # Support passing a num_labels for compatibility reasons. Convert to label map here.
        num_labels = kwargs.pop("num_labels", None)
        if num_labels is not None:
            id2label = {i: "LABEL_" + str(i) for i in range(num_labels)}
        return super().load_adapter(
            adapter_name_or_path,
            config=config,
            version=version,
            model_name=model_name,
            load_as=load_as,
            source=source,
            custom_weights_loaders=custom_weights_loaders,
            leave_out=leave_out,
            id2label=id2label,
            set_active=set_active,
            **kwargs,
        )

    def save_all_adapters(
        self,
        save_directory: str,
        with_head: bool = True,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        os.makedirs(save_directory, exist_ok=True)
        for name in self.config.adapters:
            adapter_config = self.config.adapters.get(name)
            h = get_adapter_config_hash(adapter_config)
            save_path = join(save_directory, name)
            if meta_dict:
                meta_dict.update({"config_id": h})
            else:
                meta_dict = {"config_id": h}
            self.save_adapter(
                save_path,
                name,
                meta_dict=meta_dict,
                with_head=with_head,
                custom_weights_loaders=custom_weights_loaders,
            )

    def save_adapter_fusion(
        self,
        save_directory: str,
        adapter_names: Union[Fuse, list, str],
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        with_head: Union[bool, str] = False,
    ):
        """
        Saves an AdapterFusion layer and its configuration file to a directory so that it can be shared or reloaded
        using `load_adapter_fusion()`.

        Args:
            save_directory (str): Path to a directory where the AdapterFusion should be saved.
            adapter_names (Union[Fuse, list, str]): AdapterFusion to be saved.
            with_head (Union[bool, str]):
                If True, will save a head with the same name as the AdapterFusionLayer. If a string, this will be used
                as the name of the head to be saved.

        Raises:
            ValueError: If the given AdapterFusion name is invalid.
        """
        super().save_adapter_fusion(save_directory, adapter_names, meta_dict, custom_weights_loaders)

        if with_head:
            # Make sure to cover the different options for adapter_names
            if isinstance(with_head, str):
                head_name = with_head
            elif isinstance(adapter_names, Fuse):
                head_name = adapter_names.name
            elif isinstance(adapter_names, list):
                head_name = ",".join(adapter_names)
            else:
                head_name = adapter_names
            if head_name not in self.heads:
                raise ValueError("No head with name {} found".format(head_name))
            loader = PredictionHeadLoader(self)
            loader.save(save_directory, head_name)

    def load_adapter_fusion(
        self,
        adapter_fusion_name_or_path: str,
        load_as: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        set_active: bool = False,
        with_head: bool = True,
        **kwargs
    ) -> str:
        if with_head:
            if custom_weights_loaders is None:
                custom_weights_loaders = []
            custom_weights_loaders.append(PredictionHeadLoader(self, error_on_missing=False))
        super().load_adapter_fusion(adapter_fusion_name_or_path, load_as, custom_weights_loaders, set_active)

    def save_all_heads(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        for head_name in self.heads:
            save_path = join(save_directory, head_name)
            self.save_head(save_path, head_name)

    def get_labels(self):
        return list(self.config.id2label.values())

    def get_labels_dict(self):
        return self.config.id2label

    def get_adapter(self, name):
        """
        If self.base_model is self, must inherit from a class that implements this method, to preclude infinite
        recursion
        """
        if self.base_model is self:
            return super().get_adapter(name)
        else:
            return self.base_model.get_adapter(name)
