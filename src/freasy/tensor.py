import numpy as np


def load_tensor(n, arcs, pos_source):
    """
    Creates a tensor from a list of dependency arcs.
    """
    # 3-dimensional tensor: [dependent, head, source_language]
    # TODO Currently does not support labels.
    single_source_tensor = np.zeros((n+1, n+1, len(arcs)), dtype=float)
    sources = []
    multi_source_matrix = np.zeros((n+1, n+1))

    # iterate through arcs
    for lang_index, (lang, lang_arcs) in enumerate(arcs.items()):
        # we do this for all single languages
        if lang != "ALL":
            sources.append(lang)  # store the languages in a particular order
            for arc in lang_arcs[pos_source]:
                single_source_tensor[arc.dependent, arc.head, lang_index] = arc.weight  # fill the tensor with weights
        else:
            for arc in lang_arcs[pos_source]:
                # multisource gets special treatment
                multi_source_matrix[arc.dependent, arc.head] = arc.weight

    return single_source_tensor, multi_source_matrix, sources
