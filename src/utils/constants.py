class BoolPlaceholder(object):
    def __bool__(self):
        raise ValueError("Bool placeholder is not set.")


BOOL_PLACEHOLDER = BoolPlaceholder()


class ListPlaceholder(object):
    def __len__(self):
        raise ValueError("List placeholder is not set.")

    def __getitem__(self, index):
        raise ValueError("List placeholder is not set.")


LIST_PLACEHOLDER = ListPlaceholder()

FLOAT_EPSILON = 1e-6
