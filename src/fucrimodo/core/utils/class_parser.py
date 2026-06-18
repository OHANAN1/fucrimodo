def convert_class_to_writeable_dict(stage_params):
    for key, value in stage_params.items():
        if isinstance(value, float) or isinstance(value, int):
            stage_params[key] = value

        elif isinstance(value, list):
            if len(value) == 0:
                stage_params[key] = "empty list"
                continue
            if isinstance(value[0], tuple):
                value_dict = {}
                for i, item in enumerate(value):
                    if isinstance(item[0], str):
                        value_dict[item[0]] = item[1]
                    else:
                        value_dict[f"item_{id}"] = item
                stage_params[key] = value_dict

            elif isinstance(
                    value[0], int
                ) or isinstance(
                    value[0], float
                ):
                stage_params[key] = value
                continue

            value_dict = {}
            for item in value:
                if hasattr(item, "__dict__"):
                    value_dict[item.__class__.__name__] = str(
                        item.__dict__
                    )
                else:
                    value_dict[item.__class__.__name__] = item

            stage_params[key] = value_dict

        elif hasattr(value, "__dict__"):
            stage_params[key] = {
                value.__class__.__name__: str(value.__dict__)
            }
