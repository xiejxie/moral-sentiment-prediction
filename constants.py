CATEGORIES = {
    "care": 0,
    "harm": 1,
    "fairness": 2,
    "cheating": 3,
    "loyalty": 4,
    "betrayal": 5,
    "authority": 6,
    "subversion": 7,
    "purity": 8,
    "degradation": 9
}

CATEGORY_LABELS = [y[0] for y in sorted(CATEGORIES.items(), key=lambda x:x[1])]

def convert_to_polar(input):
    # Moral virtues map to 0
    # Moral vices map to 1
    return input % 2

def convert_to_labels(input):
    if 2 in set(input):
        inv_categories = {v: k for k, v in CATEGORIES.items()}
    else:
        inv_categories = {
            0: "positive",
            1: "negative"
        }
    return [inv_categories[x] for x in input]