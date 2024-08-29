def get_device(config):
    if "local_rank" not in config:
        return 0 
    local_rank = config.local_rank
    return local_rank

# def distributed():
#     return "DIST_WORKERS" in CONFIG_MAP and CONFIG_MAP["DIST_WORKERS"] > 1
