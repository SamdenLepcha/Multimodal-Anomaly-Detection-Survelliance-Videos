import os
from src.layers.bert import BertTokenizer, BertConfig, BertForImageCaptioning
from src.utils.logger import LOGGER as logger

def get_bert_model(args):
    # Build base dir dynamically so it's not hardcoded
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    default_bert_path = os.path.join(BASE_DIR, "models/captioning/bert-base-uncased")

    # Use provided path if exists, otherwise fall back to Hugging Face
    model_path = args.model_name_or_path
    if not os.path.exists(model_path):
        logger.warning(f"⚠️ Local model path not found: {model_path}. Falling back to Hugging Face 'bert-base-uncased'.")
        model_path = "bert-base-uncased"

    # Load pretrained bert and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer
    config = config_class.from_pretrained(
        args.config_name if args.config_name else model_path,
        num_labels=2,
        finetuning_task='image_captioning'
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else model_path,
        do_lower_case=args.do_lower_case
    )

    # Model setup
    config.img_feature_type = 'frcnn'
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = 'classification'
    config.tie_weights = args.tie_weights
    config.freeze_embedding = args.freeze_embedding
    config.label_smoothing = args.label_smoothing
    config.drop_worst_ratio = args.drop_worst_ratio
    config.drop_worst_after = args.drop_worst_after

    # Handle overrides (img_feature_dim etc.)
    update_params = ['img_feature_dim', 'num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
    model_structure_changed = [False] * len(update_params)
    for idx, param in enumerate(update_params):
        arg_param = getattr(args, param)
        config_param = getattr(config, param) if hasattr(config, param) else -1
        if arg_param > 0 and arg_param != config_param:
            logger.info(f"Update config parameter {param}: {config_param} -> {arg_param}")
            setattr(config, param, arg_param)
            model_structure_changed[idx] = True

    # Load model
    if any(model_structure_changed):
        assert config.hidden_size % config.num_attention_heads == 0
        if args.load_partial_weights:
            model = model_class.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=config)
            logger.info("Load partial weights for bert layers.")
        else:
            model = model_class(config=config)
            logger.info("Init model from scratch.")
    else:
        model = model_class.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=config)
        logger.info(f"✅ Loaded pretrained BERT from {model_path}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model total parameters: {total_params:,}')
    return model, config, tokenizer
