import importlib

def load_model(model_name, args):
    try:
        module = importlib.import_module(f"models.{model_name}")
        model_class = getattr(module, model_name)
        return model_class(args)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Invalid model name: {model_name}") from e