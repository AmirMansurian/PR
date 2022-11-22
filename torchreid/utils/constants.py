GLOBAL = "globl"
FOREGROUND = "foreg"
BACKGROUND = "backg"
CONCAT_PARTS = "conct"
PARTS = "parts"
BN_GLOBAL = "bn_globl"
BN_FOREGROUND = "bn_foreg"
BN_BACKGROUND = "bn_backg"
BN_CONCAT_PARTS = "bn_conct"
BN_PARTS = "bn_parts"
PIXELS = "pixls"


def get_test_embeddings_names(parts_names, test_embeddings):
    test_embeddings_names = []
    if GLOBAL in test_embeddings or BN_GLOBAL in test_embeddings:
        test_embeddings_names.append("global")
        # test_embeddings_names.append(("global", "gb"))
    if FOREGROUND in test_embeddings or BN_FOREGROUND in test_embeddings:
        test_embeddings_names.append("foreground")
        # test_embeddings_names.append(("foreground", "fg"))
    if CONCAT_PARTS in test_embeddings or BN_CONCAT_PARTS in test_embeddings:
        test_embeddings_names.append("concatenated")
        # test_embeddings_names.append(("concatenated", "cc"))
    if PARTS in test_embeddings or BN_PARTS in test_embeddings:
        test_embeddings_names = test_embeddings_names + parts_names
    return test_embeddings_names