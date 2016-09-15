from collections import defaultdict
import csv
import scipy.spatial.distance as dist
import sys


def read_iso_mappings(filename):
    """
    Loads the ISO 639-2/3 mappings.
    """
    mappings_to = defaultdict(list)  # one two-letter ISO code can have multiple three-letter ones assigned
    mappings_back = defaultdict()

    with open(filename) as file:
        for line in file:
            line = line.strip()
            if line:
                iso3, iso2 = line.split()
                mappings_to[iso2].append(iso3)
                mappings_back[iso3] = iso2

    return mappings_to, mappings_back


def read_wals_csv(filename):
    """
    Loads the WALS database CSV.
    """
    wals_data = defaultdict(list)

    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            # only use the entries that have an ISO code
            if row[1]:
                wals_data[row[1]].append(row)  # TODO There are multiple entries possibly, this reflects below...

    return wals_data


def get_distributions_from_wals(target_sentence_sample, list_of_source_languages, identifier, iso_to, iso_back, wals):
    """
    Predicts source language appropriateness for a given target sentence sample using WALS and langid.
    """
    # get all tokens from the sample in a list
    target_tokens = [token for target_sentence in target_sentence_sample for token, _ in target_sentence]
    target_string = " ".join(target_tokens)

    # guess target language with langid on those tokens
    target_language_guess = identifier.classify(target_string)[0]

    # do the WALS magic to match the guessed target with an appropriate source
    target_names = iso_to[target_language_guess]

    closest_source = None
    min_distance = sys.float_info.max
    distribution_of_sources = []

    # evaluate each source language for appropriateness
    for source_language in list_of_source_languages:
        source_names = iso_to[source_language]  # get its 3-letter code from WALS

        # there can be different source names and target names in WALS; we choose the score of the closest pair
        for target_name in target_names:
            for source_name in source_names:
                # the vectors come from WALS, and there can be more than one per code
                target_vectors = wals[target_name]
                source_vectors = wals[source_name]

                min_distance_for_this_source = sys.float_info.max

                for target_vector in target_vectors:
                    for source_vector in source_vectors:

                        assert len(target_vector) == len(source_vector), "Vectors must be equal in length!"

                        # some languages have "holes" in the vectors, and we compare only on non-empty fields
                        fair_tv = []
                        fair_sv = []
                        for i in range(len(target_vector)):
                            if target_vector[i] and source_vector[i]:
                                fair_tv.append(target_vector[i])
                                fair_sv.append(source_vector[i])

                        # finally we calculate the Hamming distance as closeness metric
                        distance = dist.hamming(fair_tv, fair_sv)

                        if distance < min_distance:  # TODO Include = and append() instead?
                            min_distance = distance
                            closest_source = source_name
                        if distance < min_distance_for_this_source:
                            min_distance_for_this_source = distance

                # only include those where something changed, as Hamming distance is in [0, 1]
                if min_distance_for_this_source != sys.float_info.max:
                    distribution_of_sources.append((iso_back[source_name], min_distance_for_this_source))

    return iso_back[closest_source], distribution_of_sources
