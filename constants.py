# Constants shared across the dl_utils package.
# Import from here (not from dl_utils directly) to avoid circular imports.

# Additional Configuration Variables
RES: list[float] = [0.025, 0.01, 0.01]
HR_SHAPE: list[int] = [11416, 25916, 27499]
AXES: list[str] = ['z', 'y', 'x']

LM_RES: list[float] = [0.5, 0.129, 0.129]

NUCL_N5 = None
CELL_N5 = None

RAW_FILE = None
NUCL_FILE = None
CELL_FILE = None

TIME_POINT_NUCL = None
SECTION_RAW = None

RAW_DATA = None
SECTION_NUCL = None
NUCL_DATA = None
CELL_DATA = None

PATCH_VOLS_N5 = None
PATCH_VOLS = None
POSITIONS = None

# Global Variables
PATH = None
CELL2NUCL_TSV = None
NUCL_DEF_TSV = None
CELL_DEF_TSV = None
NUCL_TABLE = None
CELL_TABLE = None

act_shape = None
RES_DIF = None

LM_VOL = None
RAW_IMG = None
MASK_IMG = None
LM_DF = None

NUCL_DICT = None
CELL_DICT = None

OLD_LABEL_KEY = "label_id"
NUCLEUS_LABEL_KEY = OLD_LABEL_KEY
LABEL_KEY = "labels"
EMBED_DICT_EMBED = "Embeddings"
EMBED_KEY = EMBED_DICT_EMBED
PREDICTED_LABEL_KEY = "pred labels"
GT_LABEL_KEY = "gt"
LOSS_KEY = "loss"
EMBED_DICT_TEXTUREMASK = "Texture_mask"

TOKEN_KEY = "token"
FEATURES_KEY = "features"
AVG_TOKEN_FEATURES_KEY = "avg_token_feature"
MASKED_FEATURES_KEY = "masked_features"
MASKED_AVG_TOKEN_FEATURES_KEY = "masked_avg_token_feature"
