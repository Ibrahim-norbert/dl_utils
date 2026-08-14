# Constants shared across the dl_utils package.
# Import from here (not from dl_utils directly) to avoid circular imports.

# Additional Configuration Variables
RES: list[float] = [0.025, 0.01, 0.01]
HR_SHAPE: list[int] = [11416, 25916, 27499]
AXES: list[str] = ['z', 'y', 'x']

LM_RES: list[float] = [0.5, 0.129, 0.129]


MOBIE_LABEL_KEY = "label_id"
LABEL_KEY = "labels"
EMBED_DICT_EMBED = "Embeddings"
EMBED_KEY = EMBED_DICT_EMBED
PREDICTED_LABEL_KEY = "pred_labels"
GT_LABEL_KEY = "gt"
LOSS_KEY = "loss"
EMBED_DICT_TEXTUREMASK = "Texture_mask"



TOKEN_KEY = "token"
FEATURES_KEY = "features"
AVG_TOKEN_FEATURES_KEY = "avg_token_feature"
MASKED_FEATURES_KEY = "masked_features"
MASKED_AVG_TOKEN_FEATURES_KEY = "masked_avg_token_feature"
