"""
Utility for validating YAML configurations.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from omegaconf import OmegaConf, ValidationError

logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass

def validate_yaml_syntax(file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """
    Validate the syntax of a YAML file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)

def validate_config_structure(config: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the structure of a configuration against a schema.
    
    Args:
        config: Configuration to validate
        schema: Schema to validate against
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check for required fields
    for key, value in schema.items():
        if key not in config:
            errors.append(f"Missing required field: {key}")
        elif isinstance(value, dict) and isinstance(config[key], dict):
            # Recursively validate nested dictionaries
            valid, nested_errors = validate_config_structure(config[key], value)
            if not valid:
                errors.extend([f"{key}.{error}" for error in nested_errors])
    
    return len(errors) == 0, errors

def validate_model_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a model configuration.
    
    Args:
        config: Model configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    required_fields = ["name", "pretrained_model_name_or_path", "model_type"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Check quantization settings
    if "quantization" in config:
        quant = config["quantization"]
        if "enabled" in quant and quant["enabled"]:
            if "bits" not in quant:
                errors.append("Missing required field: quantization.bits")
            elif not isinstance(quant["bits"], (int, float)):
                errors.append("quantization.bits must be a number")
            elif quant["bits"] not in [4, 8]:
                errors.append("quantization.bits must be 4 or 8")
    
    # Check model loading settings
    if "model_loading" in config:
        loading = config["model_loading"]
        if "use_cache" in loading and not isinstance(loading["use_cache"], bool):
            errors.append("model_loading.use_cache must be a boolean")
        if "trust_remote_code" in loading and not isinstance(loading["trust_remote_code"], bool):
            errors.append("model_loading.trust_remote_code must be a boolean")
    
    # Check tokenizer settings
    if "tokenizer" in config:
        tokenizer = config["tokenizer"]
        if "add_bos_token" in tokenizer and not isinstance(tokenizer["add_bos_token"], bool):
            errors.append("tokenizer.add_bos_token must be a boolean")
        if "use_fast" in tokenizer and not isinstance(tokenizer["use_fast"], bool):
            errors.append("tokenizer.use_fast must be a boolean")
    
    return len(errors) == 0, errors

def validate_generation_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a generation configuration.
    
    Args:
        config: Generation configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check temperature
    if "temperature" in config:
        if not isinstance(config["temperature"], (int, float)):
            errors.append("temperature must be a number")
        elif config["temperature"] <= 0:
            errors.append("temperature must be positive")
    
    # Check top_p
    if "top_p" in config:
        if not isinstance(config["top_p"], (int, float)):
            errors.append("top_p must be a number")
        elif not 0 <= config["top_p"] <= 1:
            errors.append("top_p must be between 0 and 1")
    
    # Check top_k
    if "top_k" in config:
        if not isinstance(config["top_k"], int):
            errors.append("top_k must be an integer")
        elif config["top_k"] < 0:
            errors.append("top_k must be non-negative")
    
    # Check repetition_penalty
    if "repetition_penalty" in config:
        if not isinstance(config["repetition_penalty"], (int, float)):
            errors.append("repetition_penalty must be a number")
        elif config["repetition_penalty"] <= 0:
            errors.append("repetition_penalty must be positive")
    
    # Check max_length and min_length
    if "max_length" in config:
        if not isinstance(config["max_length"], int):
            errors.append("max_length must be an integer")
        elif config["max_length"] <= 0:
            errors.append("max_length must be positive")
    
    if "min_length" in config:
        if not isinstance(config["min_length"], int):
            errors.append("min_length must be an integer")
        elif config["min_length"] < 0:
            errors.append("min_length must be non-negative")
    
    if "max_length" in config and "min_length" in config:
        if isinstance(config["max_length"], int) and isinstance(config["min_length"], int):
            if config["min_length"] > config["max_length"]:
                errors.append("min_length must be less than or equal to max_length")
    
    # Check batch_size
    if "batch_size" in config:
        if not isinstance(config["batch_size"], int):
            errors.append("batch_size must be an integer")
        elif config["batch_size"] <= 0:
            errors.append("batch_size must be positive")
    
    # Check vLLM settings
    if "vllm" in config:
        vllm = config["vllm"]
        if "enabled" in vllm and not isinstance(vllm["enabled"], bool):
            errors.append("vllm.enabled must be a boolean")
        if "tensor_parallel_size" in vllm:
            if not isinstance(vllm["tensor_parallel_size"], int):
                errors.append("vllm.tensor_parallel_size must be an integer")
            elif vllm["tensor_parallel_size"] <= 0:
                errors.append("vllm.tensor_parallel_size must be positive")
        if "gpu_memory_utilization" in vllm:
            if not isinstance(vllm["gpu_memory_utilization"], (int, float)):
                errors.append("vllm.gpu_memory_utilization must be a number")
            elif not 0 < vllm["gpu_memory_utilization"] <= 1:
                errors.append("vllm.gpu_memory_utilization must be between 0 and 1")
    
    return len(errors) == 0, errors

def validate_logging_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a logging configuration.
    
    Args:
        config: Logging configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check enabled
    if "enabled" in config and not isinstance(config["enabled"], bool):
        errors.append("enabled must be a boolean")
    
    # Check project
    if "project" in config and not isinstance(config["project"], str):
        errors.append("project must be a string")
    
    # Check tags
    if "tags" in config:
        # Handle both list and sequence formats from OmegaConf
        if not (isinstance(config["tags"], list) or 
                (hasattr(config["tags"], "__iter__") and not isinstance(config["tags"], (str, dict)))):
            errors.append("tags must be a list")
        else:
            # Check tag items if it's iterable
            try:
                for i, tag in enumerate(config["tags"]):
                    if not isinstance(tag, str):
                        errors.append(f"tags[{i}] must be a string")
            except (TypeError, ValueError):
                errors.append("tags must be a list of strings")
    
    # Check log_interval
    if "log_interval" in config:
        if not isinstance(config["log_interval"], int):
            errors.append("log_interval must be an integer")
        elif config["log_interval"] <= 0:
            errors.append("log_interval must be positive")
    
    # Check metrics
    if "metrics" in config:
        metrics = config["metrics"]
        for metric in ["throughput", "latency", "token_diversity", "memory_usage"]:
            if metric in metrics and not isinstance(metrics[metric], bool):
                errors.append(f"metrics.{metric} must be a boolean")
    
    return len(errors) == 0, errors

def validate_post_processing_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a post-processing configuration.
    
    Args:
        config: Post-processing configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check enabled
    if "enabled" in config and not isinstance(config["enabled"], bool):
        errors.append("enabled must be a boolean")
    
    # Check max_length
    if "max_length" in config:
        if not isinstance(config["max_length"], int):
            errors.append("max_length must be an integer")
        elif config["max_length"] <= 0:
            errors.append("max_length must be positive")
    
    # Check min_length
    if "min_length" in config:
        if not isinstance(config["min_length"], int):
            errors.append("min_length must be an integer")
        elif config["min_length"] < 0:
            errors.append("min_length must be non-negative")
    
    # Check complete_sentences
    if "complete_sentences" in config and not isinstance(config["complete_sentences"], bool):
        errors.append("complete_sentences must be a boolean")
    
    # Check deduplicate
    if "deduplicate" in config and not isinstance(config["deduplicate"], bool):
        errors.append("deduplicate must be a boolean")
    
    return len(errors) == 0, errors

def validate_full_config(config: Dict[str, Any]) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validate a complete configuration.
    
    Args:
        config: Complete configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages_by_section)
    """
    errors = {}
    
    # Validate model configuration
    if "model" in config:
        valid, model_errors = validate_model_config(config["model"])
        if not valid:
            errors["model"] = model_errors
    else:
        errors["model"] = ["Missing model configuration"]
    
    # Validate generation configuration
    if "generation" in config:
        valid, generation_errors = validate_generation_config(config["generation"])
        if not valid:
            errors["generation"] = generation_errors
    else:
        errors["generation"] = ["Missing generation configuration"]
    
    # Validate logging configuration
    if "logging" in config:
        valid, logging_errors = validate_logging_config(config["logging"])
        if not valid:
            errors["logging"] = logging_errors
    else:
        errors["logging"] = ["Missing logging configuration"]
    
    # Validate post-processing configuration
    if "post_processing" in config:
        valid, post_processing_errors = validate_post_processing_config(config["post_processing"])
        if not valid:
            errors["post_processing"] = post_processing_errors
    else:
        errors["post_processing"] = ["Missing post-processing configuration"]
    
    return len(errors) == 0, errors

def validate_config_directory(config_dir: Union[str, Path]) -> Dict[str, Dict[str, List[str]]]:
    """
    Validate all YAML files in a configuration directory.
    
    Args:
        config_dir: Path to the configuration directory
        
    Returns:
        Dictionary of errors by file
    """
    config_dir = Path(config_dir)
    errors = {}
    
    # Walk through the directory
    for root, _, files in os.walk(config_dir):
        for file in files:
            if file.endswith((".yaml", ".yml")):
                file_path = Path(root) / file
                
                # Validate YAML syntax
                valid, error = validate_yaml_syntax(file_path)
                if not valid:
                    errors[str(file_path)] = {"syntax": [error]}
                    continue
                
                # Load the configuration
                try:
                    config = OmegaConf.load(file_path)
                    
                    # Determine the configuration type based on the directory
                    rel_path = file_path.relative_to(config_dir)
                    parts = rel_path.parts
                    
                    if len(parts) > 1:
                        config_type = parts[0]
                        
                        # Validate based on configuration type
                        if config_type == "model":
                            valid, validation_errors = validate_model_config(config)
                        elif config_type == "generation":
                            valid, validation_errors = validate_generation_config(config)
                        elif config_type == "logging":
                            valid, validation_errors = validate_logging_config(config)
                        elif config_type == "post_processing":
                            valid, validation_errors = validate_post_processing_config(config)
                        else:
                            valid, validation_errors = True, []
                        
                        if not valid:
                            errors[str(file_path)] = {"validation": validation_errors}
                except Exception as e:
                    errors[str(file_path)] = {"loading": [str(e)]}
    
    return errors

def check_config_files(config_dir: Union[str, Path] = "config") -> bool:
    """
    Check all configuration files for errors.
    
    Args:
        config_dir: Path to the configuration directory
        
    Returns:
        True if all configurations are valid, False otherwise
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        logger.error(f"Configuration directory not found: {config_dir}")
        return False
    
    errors = validate_config_directory(config_dir)
    
    if errors:
        logger.error("Configuration errors found:")
        for file, file_errors in errors.items():
            logger.error(f"  {file}:")
            for error_type, error_messages in file_errors.items():
                for error in error_messages:
                    logger.error(f"    {error_type}: {error}")
        return False
    
    logger.info("All configuration files are valid")
    return True 